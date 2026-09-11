"""Portable offline IQL over explicit, pre-action CNN+G1 trajectory tensors."""
from dataclasses import asdict, replace
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
from .contracts import DIM, PROTOCOL, FEATURE, FORCED, scalar_state
from .model import ARCH, networks, encode_normalized
from .policy import SCHEMA
from .shared import MODELS, result_dir, under, write, sha256, implementation_hashes
from ours4wan21.contracts import TrainingConfig, training_config
from ours4wan21.local_iql import _run_epoch, required_hard_budget_action


def validate_data(data):
    if data.get('schema') != 'ours4wan22_cnn_G1_dataset_v1' or data.get('protocol') != PROTOCOL or data.get('feature_contract') != FEATURE:
        raise ValueError('dataset protocol/feature contract mismatch')
    states,actions,rewards = data['states'],data['actions'],data['terminal_psnr']
    n = len(states)
    if states.shape != (n,50,DIM) or not n or not torch.isfinite(states).all():
        raise ValueError('states must be finite [trajectories,50,18439] raw pre-action features')
    if actions.shape != (n,50) or not ((actions == 0)|(actions == 1)).all():
        raise ValueError('binary [trajectories,50] actions required')
    if rewards.shape != (n,) or not torch.isfinite(rewards).all() or (rewards < 0).any():
        raise ValueError('finite terminal absolute RGB PSNR required')
    if (len(data['prompt_ids']) != n or len(data['split']) != n
            or not {'train','val'} <= set(data['split']) <= {'train','val','test'}):
        raise ValueError('explicit train/val/test trajectory split required')
    assignment = {}
    masks = torch.zeros(n,50)
    for i,(pid,split) in enumerate(zip(data['prompt_ids'],data['split'])):
        if pid in assignment and assignment[pid] != split:
            raise ValueError('prompt leakage across splits')
        assignment[pid] = split
        k = int(actions[i].sum())
        used = consecutive = 0
        for step in range(50):
            scalar = states[i,step,-7:]
            expected = scalar_state(step,k,used,consecutive,step not in (0,32),float(scalar[0]),float(scalar[1]))
            if not torch.allclose(scalar,expected,atol=2e-6,rtol=1e-6):
                raise ValueError('state/action history or scalar convention mismatch')
            if step in (0,32) and torch.count_nonzero(states[i,step,:-7]):
                raise ValueError('stage-start latent features must be zero')
            required,_ = required_hard_budget_action(step_index=step,used_skips=used,skip_budget=k,num_steps=50,forced_steps=FORCED)
            action = int(actions[i,step])
            if required is not None and action != required:
                raise ValueError('infeasible recorded action')
            masks[i,step] = float(required is None)
            used += action
            consecutive = consecutive+1 if action else 0
    return masks


class Transitions(Dataset):
    def __init__(self, x, data, mask, split):
        self.x,self.data,self.mask = x,data,mask
        self.indices = [(i,t) for i,s in enumerate(data['split']) if s == split for t in range(50)]
    def __len__(self):
        return len(self.indices)
    def __getitem__(self,index):
        i,t = self.indices[index]
        return dict(state=self.x[i,t],next_state=self.x[i,min(t+1,49)],
            action=self.data['actions'][i,t],done=torch.tensor(float(t==49)),
            reward=self.data['terminal_psnr'][i] if t==49 else torch.tensor(0.),actor_mask=self.mask[i,t])


def run(args):
    data = torch.load(args.dataset,map_location='cpu',weights_only=False)
    mask = validate_data(data)
    train_ids = [i for i,s in enumerate(data['split']) if s == 'train']
    train = data['states'][train_ids].reshape(-1,DIM).double()
    normalizer = dict(mean=train.mean(0).float(),std=train.std(0,unbiased=False).clamp_min(1e-6).float())
    del train
    x = encode_normalized(data['states'],normalizer)
    weights = under(args.weights,MODELS)
    if weights.exists():
        raise FileExistsError(weights)
    output = result_dir(args.output,'# Ours4Wan22 IQL training\n\nconfig.json and epoch_metrics.jsonl record training; model_weights links to checkpoints under models/.')
    weights.mkdir(parents=True)
    (weights/'README.md').write_text('# CNN+G1 IQL checkpoints\n\nIndependent actor/Q1/Q2/V and EMA targets, optimizer, normalizer and protocol metadata.\n')
    (output/'model_weights').symlink_to(weights,target_is_directory=True)
    torch.manual_seed(42)
    cfg = replace(training_config('aggressive_a1_v1'),epochs=args.epochs,batch_size=args.batch_size,num_layers=2)
    nets = networks(args.device)
    opts = tuple(torch.optim.AdamW(p,lr=cfg.lr,weight_decay=cfg.weight_decay) for p in
        (nets['value_net'].parameters(),list(nets['q1_net'].parameters())+list(nets['q2_net'].parameters()),nets['policy_net'].parameters()))
    loaders = {s:DataLoader(Transitions(x,data,mask,s),batch_size=cfg.batch_size,shuffle=s=='train',num_workers=0) for s in ('train','val')}
    metadata = dict(schema=SCHEMA,architecture=ARCH,feature_contract=FEATURE,protocol=PROTOCOL,
        group='G1',normalizer=normalizer,smoke_only=bool(args.smoke_only),train_config=asdict(cfg),dataset_sha256=sha256(args.dataset),
        implementation_sha256=implementation_hashes())
    write(output/'config.json',{k:v for k,v in metadata.items() if k!='normalizer'})
    for epoch in range(1,cfg.epochs+1):
        row = dict(epoch=epoch)
        for split in ('train','val'):
            row[split] = _run_epoch(loader=loaders[split],**nets,args=cfg,
                device=torch.device(args.device),optimizers=opts if split=='train' else None)
        with (output/'epoch_metrics.jsonl').open('a') as f:
            import json
            f.write(json.dumps(row,allow_nan=False)+'\n')
        payload = dict(**metadata,epoch=epoch,**{k:v.state_dict() for k,v in nets.items()},
                       optimizers=[o.state_dict() for o in opts],rng_state=torch.get_rng_state())
        torch.save(payload,weights/f'epoch_{epoch:03d}.pt')
    write(output/'COMPLETE.json',dict(status='complete',epochs=cfg.epochs,checkpoint=str(weights/f'epoch_{cfg.epochs:03d}.pt'),
        checkpoint_selection='not_selected; final epoch is not a validation-selected checkpoint'))
    return output
