import sys,time,types
from pathlib import Path
import torch
from common import FORMAL,PROTOCOL
sys.path.insert(0,str(FORMAL))
from model import CNN,ARCH,GROUPS,dimension,encode_normalized
from features import History,CONTRACT,pool
from ours4wan21.latent_features import sea_filter
from ours4wan21.policy import Policy as BasePolicy
from ours4wan21 import runtime as rt

class SelectedHistory(History):
    def __init__(self,group):super().__init__();self.group=group
    @torch.no_grad()
    def observe(self,latent,step,sigma):
        if self.pending or step!=self.step+1 or not 0<=step<50:raise ValueError('history order')
        if not 0<=sigma<=1 or (self.sigma is not None and sigma>=self.sigma):raise ValueError('sigma order')
        x=latent.detach().half().float()
        if x.ndim==4:x=x.unsqueeze(0)
        if tuple(x.shape)!=(1,16,21,60,104) or not torch.isfinite(x).all():raise ValueError('latent contract')
        self.current=x;self.step=step;self.sigma=sigma;self.pending=True
        if self.cache is None:return x.new_zeros(dimension(self.group)-7)
        fields=[x,self.previous,self.cache]
        if self.group in ('G3','G4'):fields=[sea_filter(z,sigma) for z in fields]
        if self.group in ('G2','G4'):fields=[fields[0]-fields[1],fields[0]-fields[2]]
        parts=[pool(z) for z in fields]
        value=torch.cat((torch.cat([p[0] for p in parts]),torch.cat([p[1] for p in parts])))
        if not torch.isfinite(value).all():raise ValueError('nonfinite feature')
        return value

class Policy(BasePolicy):
    def __init__(self,checkpoint,device='cuda',state_mode=None):
        p=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if p.get('schema')!='ours21_three_branch_cnn_checkpoint_v1' or p['architecture']!=ARCH or p['feature_contract']!=CONTRACT or p['protocol']!=PROTOCOL or p['smoke_only']:raise ValueError('CNN checkpoint contract')
        self.group=p['group']
        if state_mode is not None and state_mode!='cnn_'+self.group:raise ValueError('group mismatch')
        self.mode='sea7' # Exact-K scalar state; raw CNN feature is supplied separately.
        self.device=torch.device(device);self.net=CNN(self.group,2).to(self.device).eval();self.net.load_state_dict(p['policy_net'])
        self.normalizer={k:v.to(self.device) for k,v in p['normalizer'].items()}
        if set(self.normalizer)!={'mean','std'} or any(v.shape!=(dimension(self.group),) or not torch.isfinite(v).all() for v in self.normalizer.values()) or not (self.normalizer['std']>0).all():raise ValueError('normalizer')
        if not all(torch.isfinite(v).all() for v in self.net.parameters()):raise ValueError('weights')
        # Same operation scope as the validated static estimate; no hidden feature FLOPs.
        macs=12290304 if GROUPS[self.group]==3 else 12093696
        self.flops_profile=dict(flops_per_call=2*macs,convention='2_flops_per_mac',scope='CNN+MLP Conv/Linear only; excludes normalization, activations, pooling and all feature extraction',breakdown={'conv_linear':2*macs})
        self._event_pairs=[];self.current_feature=None
        if self.device.type=='cuda':
            with torch.no_grad(),torch.autocast(device_type='cuda',enabled=False):self.net(torch.zeros(1,dimension(self.group),device=self.device))
            for _ in range(48):
                pair=(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True))
                for e in pair:e.record(torch.cuda.current_stream(self.device))
                self._event_pairs.append(pair)
            self._event_pairs[-1][1].synchronize()
        self.reset_measurements();self.set_sampling(None)
    @torch.no_grad()
    def choose(self,scalar):
        if self.current_feature is None:raise ValueError('feature not observed')
        index=len(self._measurements)
        if index>=48:raise ValueError('too many policy calls')
        start,end=self._event_pairs[index] if self.device.type=='cuda' else (None,None)
        begun=time.perf_counter()
        with torch.autocast(device_type=self.device.type,enabled=False):
            raw=torch.cat((self.current_feature,scalar)).to(self.device).unsqueeze(0)
            x=encode_normalized(raw,self.normalizer).float()
            if start is not None:start.record(torch.cuda.current_stream(self.device))
            tick=time.perf_counter();logits=self.net(x);host=time.perf_counter()-tick
            if end is not None:end.record(torch.cuda.current_stream(self.device))
            if not torch.isfinite(logits).all():raise ValueError('nonfinite logits')
            action=int(logits[0].argmax().item());prob=float(logits[0].softmax(-1)[1].item())
        self._measurements.append(dict(network_host_span_seconds=host,decision_wall_seconds=time.perf_counter()-begun))
        return action,prob

class Controller(rt.Controller):
    def reset(self):
        super().reset();self.feature_history=SelectedHistory(self.policy.group);self.policy.current_feature=None;self.saved_features=[]
    def observe_latent(self,latent,step):
        tick=time.perf_counter()
        with torch.autocast(device_type=latent.device.type,enabled=False):
            self.policy.current_feature=self.feature_history.observe(latent,step,float(self.scheduler_sigmas[step])).cpu()
            self.saved_features.append(self.policy.current_feature.clone())
        self.feature_wall_seconds.append(time.perf_counter()-tick)
        self.current_latent_feature=None # inherited scalar state constructor must receive only SEA7
    def feature_overhead(self):
        return dict(group=self.policy.group,calls=len(self.feature_wall_seconds),wall_seconds=sum(self.feature_wall_seconds),per_step_wall_seconds=self.feature_wall_seconds,tflops=None,scope='selected-group feature construction and CPU trace copy including queue waits; nested in DiT/generate',tflops_scope='FFT/statistics/pooling not counted')
    def summary(self):
        out=super().summary()
        if len(self.saved_features)!=50:raise ValueError('incomplete latent feature trace')
        out.update(policy_family='three_branch_cnn',cnn_group=self.policy.group,architecture=ARCH,feature_contract=CONTRACT,latent_feature_file='latent_features.pt',policy_input_layout='concat(latent_features[step], decision.state SEA7); train normalizer then FP16->FP32')
        return out

def apply_policy(pipeline,policy,budget):
    integration=rt.integration_functions();controller=Controller(policy,budget);pipeline.model.seacache_controller=controller
    def forward(model,x,*args,**kwargs):
        if kwargs.get('seacache_branch')=='cond':
            if len(x)!=1:raise ValueError('batch1 required')
            controller.observe_latent(x[0],kwargs['seacache_step_index'])
        return integration.seacache_forward(model,x,*args,**kwargs)
    pipeline.model.forward=types.MethodType(forward,pipeline.model)
    def generate(self,input_prompt,size=(832,480),frame_num=81,shift=5.,sample_solver='unipc',sampling_steps=50,guide_scale=5.,n_prompt='',seed=42,offload_model=False):
        if (tuple(size),frame_num,shift,sample_solver,sampling_steps,guide_scale,seed,offload_model)!=((832,480),81,5.,'unipc',50,5.,42,False):raise ValueError('fixed Wan21 protocol')
        if self.t5_cpu or self.rank!=0 or self.sp_size!=1 or self.param_dtype!=torch.bfloat16:raise ValueError('resident single-GPU BF16 protocol')
        return integration.t2v_generate(self,input_prompt,size=size,frame_num=frame_num,shift=shift,sample_solver=sample_solver,sampling_steps=sampling_steps,guide_scale=guide_scale,n_prompt=n_prompt,seed=seed,offload_model=offload_model)
    pipeline.generate=types.MethodType(generate,pipeline);return controller
