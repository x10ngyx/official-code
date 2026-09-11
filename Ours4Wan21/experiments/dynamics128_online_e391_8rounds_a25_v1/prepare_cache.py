"""Verify e391 offline feature cache and exercise live feature -> replay -> IQL."""
from pathlib import Path
import sys,json,random,time
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
import torch
from ours4wan21.contracts import EXP_ROOT,MODEL_ROOT,create_result,dump,sha256,state_contract
from ours4wan21.feature_cache import load_feature_index,load_features
from ours4wan21.online_common import read,seal,online_config
from ours4wan21.online_training import trace_transitions,OnlineTrainer,UniformReplay,fixed_support
from ours4wan21.latent_features import LatentFeatureHistory
from ours4wan21.runtime import Controller
from ours4wan21.policy import Policy

NAME='ours21_dynamics128_e391_online_offline800_8rounds_a25_v1'
MODE='sea7_dynamics_raw_sea128';GROUP='dynamics_raw_sea128'
CACHE=EXP_ROOT/'ours21_random3000_12groups_v1_sea7_dynamics_raw_sea128_cache'
FEATURES=EXP_ROOT/'ours21_random3000_12groups_v1_features'
CHECKPOINT=MODEL_ROOT/'ours21_random3000_12groups_v1_sea7_dynamics_raw_sea128/checkpoints/epoch_391.pt'
SETUP=EXP_ROOT/(NAME+'_setup')

def main():
    torch.set_num_threads(1)
    marker=read(CACHE/'COMPLETE.json');digest=sha256(CACHE/'transitions.pt')
    if marker['status']!='complete' or marker['sha256']!=digest:raise ValueError('offline dynamics cache corrupt/incomplete')
    selection=read(EXP_ROOT/'ours21_random3000_12groups_v1_sea7_dynamics_raw_sea128_analysis/checkpoint_selection.json')
    if selection['checkpoint_epoch']!=391 or sha256(CHECKPOINT)!=selection['checkpoint_sha256']:raise ValueError('wrong Dynamics128 e391')
    parent=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
    bundle=torch.load(CACHE/'transitions.pt',map_location='cpu',weights_only=False)
    manifest=read(CACHE/'manifest.json');index=load_feature_index(FEATURES)
    if parent['epoch']!=391 or parent['state']!=state_contract(MODE) or parent['dataset_manifest']!=manifest or bundle['manifest']!=manifest:
        raise ValueError('checkpoint/cache state or provenance mismatch')
    if manifest['latent_feature_cache']['index_sha256']!=sha256(FEATURES/'index.json'):raise ValueError('wrong source feature index')
    if len(index['rows'])!=3000 or bundle['tensors']['state'].shape!=(150000,135):raise ValueError('incomplete dynamics cache')
    sea=read(EXP_ROOT/'ours21_random3000_12groups_v1_sea7_cache/manifest.json')
    if manifest['sources']!=sea['sources'] or manifest['prompt_splits']!=sea['prompt_splits']:raise ValueError('baseline/timing prior source population differs')
    counts={key:len(bundle[key+'_indices']) for key in ('train','val','test')}
    if counts!=dict(train=120000,val=15000,test=15000):raise ValueError('unexpected data split')
    checked=[]
    rng=random.Random(42)
    for split in ('train','evaluation','test'):
        eligible=[i for i,s in enumerate(manifest['sources']) if s['split']==split]
        for i in sorted(rng.sample(eligible,4)):
            source=manifest['sources'][i];trace=next(Path(x) for x in source['files'] if x.endswith('/trace.json'))
            if sha256(trace)!=source['files'][str(trace)]:raise ValueError('source trace changed')
            actions=bundle['tensors']['action'][50*i:50*(i+1)]
            features=load_features(FEATURES,dict(source,split=source['source_split']),trace,actions,index=index)[GROUP]
            if not torch.equal(features,bundle['tensors']['state'][50*i:50*(i+1),:128]):raise ValueError('feature cache/transition leading dimensions differ')
            checked.append(dict(trajectory_id=source['trajectory_id'],split=split,feature_file=index['rows'][source['trajectory_id']]))
    # Real e391 policy and real causal feature extractor, synthetic small latents only.
    policy=Policy(CHECKPOINT,device='cpu',state_mode=MODE);policy.set_sampling(20260813)
    c=Controller(policy,29);sigmas=torch.linspace(.99,0.,51);c.set_scheduler_sigmas(sigmas)
    history=LatentFeatureHistory([GROUP]);generator=torch.Generator().manual_seed(42);features=[]
    for step in range(50):
        latent=torch.randn((16,2,4,4),generator=generator)*.2
        expected=history.observe(latent.half(),step,sigmas[step])[GROUP][0]
        c.observe_latent(latent,step)
        token=torch.arange(32).float().reshape(1,8,4)+1+step*.1
        for branch in ('cond','uncond'):
            reuse=c.plan_step(branch=branch,step_index=step,num_steps=50,feature=token,grid_size=torch.tensor([2,2,2]))
            if reuse:c.reuse_residual(branch,step)
            else:c.record_recompute(branch,step,torch.ones_like(token))
            if not torch.equal(torch.tensor(c.decisions[-1]['state'][:128]),expected):raise ValueError('live/cache extraction differs')
        history.commit(int(reuse));features.append(expected)
    trace=c.summary();transitions=trace_transitions(trace,22.,MODE)
    if transitions['state'].shape!=(50,135) or int(transitions['action'].sum())!=29:raise ValueError('invalid online dynamics replay')
    if not torch.equal(transitions['state'][:,:128],torch.stack(features)):raise ValueError('online replay lost dynamics')
    cfg=online_config(rounds=8,prompt_pool_size=800,profile='aggressive_a2_a3_v1')
    trainer=OnlineTrainer(parent,cfg,'cpu');rows={k:v[bundle['train_indices'][:500]] for k,v in bundle['tensors'].items()}
    replay=UniformReplay(rows,[transitions]);before={k:v.clone() for k,v in trainer.nets['policy_net'].state_dict().items()}
    warmup=trainer.update(replay,actor=False)
    if any(not torch.equal(before[k],v) for k,v in trainer.nets['policy_net'].state_dict().items()):raise ValueError('warmup altered actor')
    joint=trainer.update(replay,actor=True)
    if all(torch.equal(before[k],v) for k,v in trainer.nets['policy_net'].state_dict().items()):raise ValueError('joint did not update actor')
    for k,v in parent['normalizer'].items():
        if not torch.equal(v,trainer.normalizer[k]):raise ValueError('normalizer changed')
    saved=trainer.checkpoint(replay,dict(round=1,joint_epoch=1));OnlineTrainer(saved,cfg,'cpu',restore=True)
    support=fixed_support(bundle,cfg)
    if support['states'].shape[1]!=135:raise ValueError('fixed support is not dynamics state')
    create_result(SETUP,'# Dynamics128 e391 online cache preflight\n\ncache_audit.json verifies existing offline cache and CPU live feature/replay/IQL integration. offline_cache and feature_cache link immutable existing caches. Synthetic integration does not produce production replay or a new checkpoint. Online rounds build real 135D replay from current-policy traces; no baseline latents enter feature extraction.\n')
    (SETUP/'offline_cache').symlink_to(CACHE,target_is_directory=True);(SETUP/'feature_cache').symlink_to(FEATURES,target_is_directory=True)
    dump(SETUP/'cache_audit.json',dict(status='pass',checkpoint=str(CHECKPOINT.resolve()),checkpoint_sha256=sha256(CHECKPOINT),offline_epoch=391,
        mode=MODE,state_width=135,dynamics_width=128,transitions=150000,splits=counts,offline_cache_reused=True,
        same_offline_sources_and_prompt_splits_as_baseline_reference=True,
        sources={str(p.resolve()):sha256(p) for p in (CHECKPOINT,CACHE/'transitions.pt',CACHE/'manifest.json',CACHE/'COMPLETE.json',FEATURES/'index.json',FEATURES/'COMPLETE.json')},
        feature_transition_checks=checked,feature_check_scope='12 deterministic source rows, 4 per split; full bundle/index SHA and exact checkpoint manifest binding checked',
        online_integration=dict(kind='CPU real e391 policy, synthetic small latents and terminal reward; no production quality result',
            steps=50,cfg_decisions=100,skip_budget=29,state_width=135,offline_live_features_exact=True,replay_features_preserved=True,
            critic_warmup_metrics=warmup,joint_metrics=joint,normalizer_frozen=True,checkpoint_restore=True),
        fixed_support_states=list(support['states'].shape),profile=cfg.payload()))
    seal(SETUP,['cache_audit.json'],identity=dict(checkpoint_sha256=sha256(CHECKPOINT),mode=MODE))
    print(json.dumps(dict(status='pass',checkpoint_epoch=391,state_width=135,cache_shape=list(bundle['tensors']['state'].shape),splits=counts,cache_rebuilt=False)),flush=True)

if __name__=='__main__':main()
