"""E385-style continuation on one transition-uniform accumulated replay buffer."""
from copy import deepcopy
from dataclasses import asdict
import math
from pathlib import Path
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .contracts import MODEL_ROOT, PROTOCOL, TrainingConfig, dump, sha256, state_contract, under
from .data import Transitions, episode
from .local_iql import (_actor_terms, _run_epoch, apply_normalizer, expectile_loss,
                        gather_action_values, soft_update)
from .train import make_networks, validate_bundle
from .online_common import (OnlineConfig, atomic_torch_save, checkpoint_identity,
                            freeze_json, read, seal, verified)


def trace_transitions(trace, psnr, mode, *, require_sampling=True):
    if trace.get('state_contract') != state_contract(mode):
        raise ValueError('online trace state contract mismatch')
    if require_sampling and trace.get('action_mode') != 'policy_categorical':
        raise ValueError('online training requires current-policy sampling traces')
    rows = trace['decisions']
    scalar = episode(rows, psnr, 'scalar5' if mode == 'scalar5' else 'sea7')
    states = torch.tensor([r['state'] for r in rows[::2]], dtype=torch.float32)
    dim = len(state_contract(mode)['names'])
    if states.shape != (50, dim) or not torch.isfinite(states).all():
        raise ValueError('missing/nonfinite online state')
    tail = states if mode == 'scalar5' else states[:, -7:]
    if not torch.allclose(tail, scalar['state'], atol=1e-7, rtol=1e-6):
        raise ValueError('online pre-action state differs from replay reconstruction')
    for i, row in enumerate(rows[::2]):
        if not torch.allclose(states[i], torch.tensor(rows[2*i+1]['state']),atol=2e-6,rtol=1e-6):
            raise ValueError('CFG full feature states differ')
        eligible = bool(scalar['actor_mask'][i])
        if row['policy_queried'] != eligible or float(row['actor_mask']) != float(eligible):
            raise ValueError('forced row entered actor sampling/loss')
        if eligible:
            p = row['p_skip']
            if not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1:
                raise ValueError('invalid policy probability')
            expected = p if row['action'] == 'reuse' else 1-p
            if not math.isclose(row['behavior_action_probability'], expected, abs_tol=1e-7):
                raise ValueError('behavior action probability does not match sampled action')
    scalar['state'] = states
    scalar['next_state'] = torch.cat((states[1:], states[-1:]))
    return scalar


def fixed_support(bundle, config=OnlineConfig()):
    """Freeze train-only discretionary states stratified by step and K, weighted by population."""
    idx = bundle['train_indices']
    t = bundle['tensors']
    idx = idx[t['actor_mask'][idx] > .5]
    if not len(idx):
        raise ValueError('no free-action training states')
    state = t['state'][idx]
    steps = (state[:, -4] * 49).round().long() // 10
    budgets = (state[:, -3] * 50).round().long() // 8
    groups = steps * 7 + budgets
    unique = groups.unique(sorted=True)
    if config.fixed_state_limit < len(unique):
        raise ValueError('fixed state budget cannot represent all strata')
    generator = torch.Generator().manual_seed(config.seed)
    selected, weights = [], []
    per_group = config.fixed_state_limit // len(unique)
    for group in unique:
        positions = idx[groups == group]
        count = min(len(positions), per_group)
        chosen = positions[torch.randperm(len(positions), generator=generator)[:count]]
        selected.append(chosen)
        weights.append(torch.full((count,), len(positions)/count, dtype=torch.float64))
    indices = torch.cat(selected)
    w = torch.cat(weights)
    return dict(indices=indices, states=t['state'][indices].clone(), weights=w/w.sum(),
                description='train-only actor-free step-decile/K-bin stratified sample; population weights')


def select_checkpoint(probabilities, actions, weights, config=OnlineConfig()):
    p = torch.as_tensor(probabilities, dtype=torch.float64)
    a = torch.as_tensor(actions)
    w = torch.as_tensor(weights, dtype=torch.float64)
    if (p.ndim != 2 or p.shape != a.shape or p.shape[0] != config.joint_epochs or
            w.shape != (p.shape[1],) or not len(w) or not torch.isfinite(p).all() or
            not torch.isfinite(w).all() or not (w > 0).all() or
            not ((p >= 0) & (p <= 1)).all() or not ((a == 0) | (a == 1)).all()):
        raise ValueError('malformed fixed-state epoch outputs')
    w = w / w.sum()
    agreement = ((a[1:] == a[:-1]).double() * w).sum(1)
    drift = ((p[1:] - p[:-1]).abs() * w).sum(1)
    rows = []
    for e in range(config.selection_start, config.selection_end+1):
        left, incoming = e-3, e-2
        rows.append(dict(epoch=e, agreement_previous=float(agreement[left]),
            agreement_incoming=float(agreement[incoming]),
            actor_agreement=min(float(agreement[left]), float(agreement[incoming])),
            probability_drift=max(float(drift[left]), float(drift[incoming]))))
    selected = max(rows, key=lambda r:(r['actor_agreement'], -r['probability_drift'], r['epoch']))
    return dict(selected_epoch=selected['epoch'], selected=selected, candidates=rows,
                criterion='maximize min of two preceding adjacent actor argmax agreements; '
                          'tie: lower max probability drift, then later epoch')


class UniformReplay:
    def __init__(self, offline, online, seed=42):
        self.offline_count = len(offline['action'])
        self.tensors = {k: torch.cat([offline[k], *[x[k] for x in online]]) for k in offline}
        self.size = len(self.tensors['action'])
        self.generator = torch.Generator().manual_seed(seed)
        self.offline_draws = self.online_draws = 0

    def sample(self, count):
        idx = torch.randint(self.size, (count,), generator=self.generator)
        old = int((idx < self.offline_count).sum())
        self.offline_draws += old
        self.online_draws += count-old
        return {k:v[idx] for k,v in self.tensors.items()}


class OnlineTrainer:
    def __init__(self, payload, config=OnlineConfig(), device='cpu', *, restore=False):
        self.config, self.device = config, torch.device(device)
        if payload.get('schema') != 'ours4wan21_iql_checkpoint_v1' or payload['protocol'] != PROTOCOL:
            raise ValueError('not a resident Wan21 checkpoint')
        self.payload = deepcopy(payload)
        self.mode = payload['state']['mode']
        if payload['state'] != state_contract(self.mode):
            raise ValueError('state contract mismatch')
        self.normalizer = {k:v.detach().cpu().float().clone() for k,v in payload['normalizer'].items()}
        self.mc, self.nets = make_networks(self.mode, TrainingConfig(), self.device)
        if asdict(self.mc) != payload['model_config']:
            raise ValueError('architecture mismatch')
        for k, net in self.nets.items():
            net.load_state_dict(payload[k], strict=True)
            if not all(torch.isfinite(v).all() for v in net.parameters()):
                raise ValueError('nonfinite network weights')
        if (set(self.normalizer) != {'mean','std'} or any(v.shape != (self.mc.input_dim,)
                or not torch.isfinite(v).all() for v in self.normalizer.values()) or
                not (self.normalizer['std'] > 0).all()):
            raise ValueError('invalid frozen normalizer')
        c, n = config, self.nets
        self.optimizers = [torch.optim.AdamW(params, lr=lr, weight_decay=c.weight_decay,
                    betas=(.9,.999), eps=1e-8) for params,lr in (
            (n['value_net'].parameters(),c.critic_lr),
            (list(n['q1_net'].parameters())+list(n['q2_net'].parameters()),c.critic_lr),
            (n['policy_net'].parameters(),c.actor_lr))]
        if restore:
            if 'online' not in payload or payload['online']['config'] != c.payload():
                raise ValueError('online continuation configuration changed')
            saved = payload['optimizer_states']
            if not isinstance(saved,list) or len(saved) != 3:
                raise ValueError('missing Wan21 optimizer states')
            for opt, state, lr in zip(self.optimizers,saved,(c.critic_lr,c.critic_lr,c.actor_lr)):
                opt.load_state_dict(state)
                for group in opt.param_groups:
                    group.update(lr=lr,weight_decay=c.weight_decay,betas=(.9,.999),eps=1e-8)
        self.restored = restore

    def normalized(self, x):
        return apply_normalizer(x.float(),self.normalizer).to(self.device)

    def update(self, replay, *, actor):
        c,n = self.config,self.nets
        batch = replay.sample(c.batch_size)
        s,ns = self.normalized(batch['state']),self.normalized(batch['next_state'])
        a,r,d,m = [batch[k].to(self.device) for k in ('action','reward','done','actor_mask')]
        vo,qo,po = self.optimizers
        with torch.no_grad():
            target = torch.minimum(gather_action_values(n['target_q1'](s),a),
                                   gather_action_values(n['target_q2'](s),a))
        vl = expectile_loss(target-n['value_net'](s),c.tau)
        vo.zero_grad(set_to_none=True); vl.backward()
        vg = torch.nn.utils.clip_grad_norm_(n['value_net'].parameters(),c.grad_clip_norm)
        vo.step()
        with torch.no_grad():
            target = r+c.gamma*(1-d)*n['value_net'](ns)
        ql = sum(F.mse_loss(gather_action_values(n[k](s),a),target) for k in ('q1_net','q2_net'))
        qp = list(n['q1_net'].parameters())+list(n['q2_net'].parameters())
        qo.zero_grad(set_to_none=True);ql.backward()
        qg = torch.nn.utils.clip_grad_norm_(qp,c.grad_clip_norm);qo.step()
        for key in ('q1','q2'):
            soft_update(n['target_'+key],n[key+'_net'],c.target_rho)
        metrics = dict(v_loss=float(vl),q_loss=float(ql),value_grad_norm=float(vg),q_grad_norm=float(qg),
                       pi_loss=0.,actor_examples=0.,policy_grad_norm=0.)
        if actor:
            pl, stats = _actor_terms(state=s,action=a,actor_mask=m,value_net=n['value_net'],
                q1_net=n['q1_net'],q2_net=n['q2_net'],policy_net=n['policy_net'],
                beta=c.beta,weight_max=c.weight_max)
            metrics.update(stats)
            if pl is not None:
                po.zero_grad(set_to_none=True);pl.backward()
                pg = torch.nn.utils.clip_grad_norm_(n['policy_net'].parameters(),c.grad_clip_norm)
                po.step();metrics['policy_grad_norm']=float(pg)
        if not all(math.isfinite(x) for x in metrics.values()):
            raise ValueError('nonfinite training metric')
        return metrics

    @torch.no_grad()
    def outputs(self, states):
        p,a = [],[]
        for x in states.split(1024):
            logits = self.nets['policy_net'](self.normalized(x))
            p.append(logits.softmax(-1)[:,1].cpu());a.append(logits.argmax(-1).cpu())
        return torch.cat(p),torch.cat(a)

    def checkpoint(self, replay, online):
        payload = deepcopy(self.payload)
        payload.update({k:{name:v.detach().cpu().clone() for name,v in net.state_dict().items()}
                        for k,net in self.nets.items()})
        payload.update(epoch=online['joint_epoch'],
            offline_epoch=payload.get('offline_epoch',payload.get('epoch')),
            normalizer=self.normalizer, optimizer_states=[o.state_dict() for o in self.optimizers],
            torch_rng_state=torch.get_rng_state(),
            cuda_rng_states=torch.cuda.get_rng_state_all() if self.device.type=='cuda' else [],
            online=dict(**online,config=self.config.payload(),replay_rng=replay.generator.get_state(),
                        python_rng=random.getstate(),offline_draws=replay.offline_draws,
                        online_draws=replay.online_draws))
        return payload


def restore_rng(payload, replay, device):
    if 'online' in payload:
        torch.set_rng_state(payload['torch_rng_state'])
        if str(device).startswith('cuda') and payload['cuda_rng_states']:
            torch.cuda.set_rng_state_all(payload['cuda_rng_states'])
        replay.generator.set_state(payload['online']['replay_rng'])
        random.setstate(payload['online']['python_rng'])


def train_round(parent_path, offline_path, online_paths, support_path, output, weights,
                round_index, config=OnlineConfig(), device='cuda', *, allow_smoke=False):
    output,weights = Path(output),under(weights,MODEL_ROOT)
    identity = dict(parent=checkpoint_identity(parent_path),offline=sha256(offline_path),
                    online=[sha256(p) for p in online_paths],support=sha256(support_path),
                    round=round_index,config=config.payload())
    if verified(output,identity):
        selected=read(output/'selected.json')
        if sha256(selected['path']) != selected['sha256']:
            raise ValueError('selected checkpoint corrupt')
        return Path(selected['path'])
    if len(online_paths) != round_index:
        raise ValueError('must include all completed online rounds')
    output.mkdir(parents=True,exist_ok=True);weights.mkdir(parents=True,exist_ok=True)
    (output/'README.md').write_text('Per-epoch online metrics and selection; checkpoints are stored under models/.\n')
    (weights/'README.md').write_text('Online warmup and joint epoch checkpoints with optimizer/RNG; selected.pt continues the next round.\n')
    freeze_json(output/'identity.json',identity)
    b=torch.load(offline_path,map_location='cpu',weights_only=False)
    parent=torch.load(parent_path,map_location='cpu',weights_only=False)
    mode=parent['state']['mode'];validate_bundle(b,mode)
    if parent.get('smoke_only') and not allow_smoke:
        raise ValueError('smoke checkpoint cannot enter production')
    if parent['dataset_manifest'] != b['manifest']:
        raise ValueError('checkpoint does not belong to this frozen offline dataset')
    if round_index>1 and parent.get('online',{}).get('round') != round_index-1:
        raise ValueError('wrong parent round')
    online=[]
    for r,path in enumerate(online_paths,1):
        data=torch.load(path,map_location='cpu',weights_only=False)
        if data['round']!=r or data['state']!=state_contract(mode):
            raise ValueError('wrong round or state in replay')
        if len(data['tensors']['action']) != config.trajectories_per_round*50:
            raise ValueError('incomplete online round')
        online.append(data['tensors'])
    old={k:v[b['train_indices']] for k,v in b['tensors'].items()}
    replay=UniformReplay(old,online,config.seed)
    batches=math.ceil(replay.size/config.batch_size)
    support=torch.load(support_path,map_location='cpu',weights_only=False)
    last=output/'latest.json'
    start=0;history=[];probs=[];actions=[]
    payload=parent
    if last.exists():
        latest=read(last)
        if sha256(latest['path'])!=latest['sha256']:
            raise ValueError('resume checkpoint corrupt')
        payload=torch.load(latest['path'],map_location='cpu',weights_only=False)
        if payload['online']['identity']!=identity:
            raise ValueError('resume lineage changed')
        start=payload['online']['completed_epochs']
        history=payload['online']['history'];probs=payload['online']['probabilities'];actions=payload['online']['actions']
    trainer=OnlineTrainer(payload,config,device,restore=start>0 or round_index>1)
    torch.manual_seed(config.seed);random.seed(config.seed)
    restore_rng(payload,replay,device)
    if start:
        replay.offline_draws=payload['online']['offline_draws'];replay.online_draws=payload['online']['online_draws']
    for count in range(start+1,config.critic_warmup_epochs+config.joint_epochs+1):
        actor=count>config.critic_warmup_epochs
        rows=[trainer.update(replay,actor=actor) for _ in range(batches)]
        means={k:sum(r[k] for r in rows)/batches for k in rows[0]}
        e=count-config.critic_warmup_epochs
        history.append(dict(phase='joint' if actor else 'warmup',epoch=e if actor else count,updates=batches,**means))
        if actor:
            p,a=trainer.outputs(support['states']);probs.append(p);actions.append(a)
        online_state=dict(round=round_index,identity=identity,completed_epochs=count,
            joint_epoch=max(e,0),history=history,probabilities=probs,actions=actions,
            replay_count=replay.size,offline_count=replay.offline_count,
            optimizer_restored_from_parent=round_index>1)
        saved=trainer.checkpoint(replay,online_state)
        path=weights/(f'epoch_{e:02d}.pt' if actor else f'warmup_{count:02d}.pt')
        atomic_torch_save(saved,path)
        dump(last,dict(path=str(path),sha256=sha256(path)))
        dump(output/'epoch_metrics.json',history)
        print(f'round {round_index}: {history[-1]["phase"]} epoch {history[-1]["epoch"]} complete',flush=True)
    selection=select_checkpoint(torch.stack(probs),torch.stack(actions),support['weights'],config)
    source=weights/f'epoch_{selection["selected_epoch"]:02d}.pt'
    chosen=torch.load(source,map_location='cpu',weights_only=False)
    chosen['online']['selection']=selection
    chosen['online']['computed_joint_epochs']=config.joint_epochs
    target=weights/'selected.pt';atomic_torch_save(chosen,target)
    dump(output/'selection.json',selection)
    dump(output/'selected.json',dict(path=str(target),sha256=sha256(target),source=str(source)))
    dump(output/'metrics.json',dict(round=round_index,replay_transitions=replay.size,
        expected_online_fraction=1-replay.offline_count/replay.size,batches_per_epoch=batches,
        joint_updates_computed=config.joint_epochs*batches,
        critic_warmup_updates=config.critic_warmup_epochs*batches,
        selected_joint_updates=selection['selected_epoch']*batches,
        computed_offline_draws=replay.offline_draws,computed_online_draws=replay.online_draws,
        selected_epoch=selection['selected_epoch']))
    seal(output,['identity.json','selection.json','selected.json','epoch_metrics.json','metrics.json'],identity=identity)
    return target
