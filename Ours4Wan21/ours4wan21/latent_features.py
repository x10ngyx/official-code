"""Versioned, causal latent feature groups shared by offline IQL and live rollout.

Only current/past candidate latents and committed actions enter this module.
No baseline, future action, reward, prompt ID or final quality is accepted.
"""
from __future__ import annotations
import math
import torch
import torch.nn.functional as F

VERSION = 'wan21_latent_groups_v1'
EPS = 1e-6
GROUPS = {
    'dynamics_raw_sea128': (128, 'Raw + same-current-filter SEA speed, turning, curvature, signed acceleration'),
    'cache_update192': (192, 'Recent update versus last recompute update: magnitude, direction and drift alignment'),
    'local_drift1024': (1024, '2x4x4 local signed mean and RMS of current-to-cache drift'),
    'spatial_gradient96': (96, 'Three-axis current/cache gradient mismatch and alignment'),
    'channel_geometry240': (240, 'Off-diagonal channel correlations and their cache-relative change'),
    'distribution256': (256, 'Per-channel location, spread, quantiles, skew, kurtosis and tail changes'),
    'spectral_drift512': (512, '32-band current-to-cache Fourier residual relative RMS'),
    'spectral_phase512': (512, '32-band phase-only disagreement with cache'),
    'spectral_shape576': (576, '32-band normalized energy migration + entropy/concentration changes'),
    'spectral_dynamics1024': (1024, '32-band acceleration and cached-update innovation spectra'),
}

def contract(group: str) -> dict:
    if group not in GROUPS:
        raise ValueError(f'Unknown latent group {group!r}; choose {list(GROUPS)}')
    return dict(version=VERSION, group=group, dim=GROUPS[group][0], channels=16,
                steps=50, stage_boundaries=[0], input_quantization='float16_then_float32',
                state_layout='feature_then_sea7', epsilon=EPS,
                sea=dict(power_const=1., power_exp=3., eps=1e-16, norm='mean',
                         history_filter='all_at_current_sigma'),
                bands='temporal_abs_index_bins_0_2_4__spatial_radius_squared_j2_over128_v1')

def rms(x): return x.square().mean((2,3,4)).sqrt().clamp_min(EPS)
def dot(a,b): return (a*b).mean((2,3,4))
def cosine(a,b): return (dot(a,b)/(rms(a)*rms(b))).clamp(-1,1)
def slog(x): return x.sign()*torch.log1p(x.abs())
def stack(xs): return torch.stack(xs,2).flatten(1)

def sea_filter(x, sigma):
    sigma=max(EPS,min(1-EPS,float(sigma)));a=1-sigma;gain=None
    for axis,n in enumerate(x.shape[2:]):
        freq=torch.fft.fftfreq(n,device=x.device,dtype=torch.float32).abs()
        power=1./(freq.pow(3)+1e-16);v=a*power/(a*a*power+sigma*sigma+1e-16)
        shape=[1,1,1];shape[axis]=n;v=v.reshape(shape);gain=v if gain is None else gain*v
    gain=gain/gain.mean()
    return torch.fft.ifftn(torch.fft.fftn(x.float(),dim=(2,3,4))*gain,dim=(2,3,4)).real

def band_matrix(shape,device):
    t,h,w=shape
    ft=(torch.fft.fftfreq(t,device=device).abs()*t).round()
    ti=torch.bucketize(ft,torch.tensor([0,2,4],device=device))
    fy=torch.fft.fftfreq(h,device=device);fx=torch.fft.fftfreq(w,device=device)
    ri=torch.bucketize(fy[:,None].square()+fx[None,:].square(),torch.arange(1,8,device=device).float().square()/128)
    ids=(ti[:,None,None]*8+ri[None]).flatten()
    return F.one_hot(ids,num_classes=32).float()

class LatentFeatureHistory:
    """observe(t) before deciding, then commit(actual_action); one call per step.

    Uses bounded history, reset at step0. Offline callers MUST replay all
    preceding actions, including forced recomputes. A fresh object per video.
    """
    def __init__(self, groups):
        self.groups=tuple(groups)
        if not self.groups or len(set(self.groups))!=len(self.groups):
            raise ValueError('Choose one or more distinct feature groups')
        for group in self.groups: contract(group)
        self.last_step=-1;self.pending=False;self._matrix=None;self._shape=None
        self.reset_stage()

    def reset_stage(self):
        self.previous=None;self.previous2=None;self.previous_sigma=None;self.previous_ds=None
        self.previous_velocity=None;self.cache=None;self.cache_velocity=None;self.last_action=None

    @torch.no_grad()
    def observe(self, latent, step, sigma):
        if self.pending or step!=self.last_step+1 or not 0<=step<50:
            raise ValueError('History requires consecutive observe/commit pairs for steps0..49')
        if not math.isfinite(float(sigma)) or not 0<=float(sigma)<=1:
            raise ValueError('Invalid scheduler sigma')
        if step == 0: self.reset_stage()
        x=latent.detach()
        if x.ndim==4:x=x.unsqueeze(0)
        if x.ndim!=5 or x.shape[0]!=1 or x.shape[1]!=16:
            raise ValueError('Expected single-video latent [1,16,T,H,W]')
        # Archives are FP16; explicitly match their precision on the live path.
        x=x.to(torch.float16).float().clone()
        if not torch.isfinite(x).all():raise ValueError('Non-finite latent feature input')
        if self.previous_sigma is not None and float(sigma)>=self.previous_sigma:
            raise ValueError('Expected strictly decreasing denoising sigma within a stage')
        self.current=x;self.current_sigma=float(sigma);self.current_step=int(step)
        ds=1. if self.previous is None else max(self.previous_sigma-float(sigma),EPS)
        v=torch.zeros_like(x) if self.previous is None else (x-self.previous)/ds
        old=torch.zeros_like(x) if self.previous_velocity is None else self.previous_velocity
        if self.last_action==0:self.cache_velocity=v
        self.current_velocity=v;self.current_ds=ds;self.pending=True
        if self.cache is None:
            return {g:x.new_zeros((1,GROUPS[g][0])) for g in self.groups}
        c=self.cache;cv=self.cache_velocity
        if cv is None:raise RuntimeError('Cached update is unavailable')
        drift=x-c;acc=v-old;innovation=v-cv;ref=rms(c);cr=rms(cv);out={}
        def dynamics(u,old_u,reference):
            return [torch.log1p(rms(u)/reference),cosine(u,old_u),
                    torch.log1p(rms(u-old_u)/rms(old_u)),slog((u-old_u).mean((2,3,4))/reference)]
        if 'dynamics_raw_sea128' in self.groups:
            sx=sea_filter(x,sigma);sp=sea_filter(self.previous,sigma)
            sv=(sx-sp)/ds
            so=torch.zeros_like(sv) if self.previous2 is None else (sp-sea_filter(self.previous2,sigma))/max(self.previous_ds,EPS)
            sr=rms(sea_filter(c,sigma));out['dynamics_raw_sea128']=stack(dynamics(v,old,ref)+dynamics(sv,so,sr))
        if 'cache_update192' in self.groups:
            out['cache_update192']=stack([torch.log1p(rms(v)/ref),torch.log1p(cr/ref),torch.log1p(rms(innovation)/cr),cosine(v,cv),cosine(innovation,drift),cosine(cv,drift),cosine(acc,cv),torch.log1p(rms(acc)/cr),slog(innovation.mean((2,3,4))/ref),slog(dot(v,cv)/ref.square()),torch.log1p(rms(drift)/ref),cosine(innovation,x)])
        if 'local_drift1024' in self.groups:
            mean=F.adaptive_avg_pool3d(drift,(2,4,4)).flatten(2)/ref[:,:,None]
            rr=F.adaptive_avg_pool3d(drift.square(),(2,4,4)).flatten(2).sqrt()/ref[:,:,None]
            out['local_drift1024']=torch.stack((slog(mean),torch.log1p(rr)),2).flatten(1)
        if 'spatial_gradient96' in self.groups:
            fields=[]
            for axis in (2,3,4):
                if x.shape[axis]<2:fields.extend([torch.zeros_like(ref),torch.zeros_like(ref)]);continue
                gx=torch.diff(x,dim=axis);gc=torch.diff(c,dim=axis)
                fields.extend([torch.log1p(rms(gx-gc)/rms(gc)),cosine(gx,gc)])
            out['spatial_gradient96']=stack(fields)
        if 'channel_geometry240' in self.groups:
            def corr(z):
                z=z.flatten(2);z=z-z.mean(2,keepdim=True)
                z=z/z.square().mean(2,keepdim=True).sqrt().clamp_min(EPS)
                return (z@z.transpose(1,2)/z.shape[2]).clamp(-1,1)
            now=corr(x);cached=corr(c);ij=torch.triu_indices(16,16,offset=1,device=x.device)
            out['channel_geometry240']=torch.cat((now[:,ij[0],ij[1]],(now-cached)[:,ij[0],ij[1]]),1)
        if 'distribution256' in self.groups:
            def stats(z):
                z=z.flatten(2);mean=z.mean(2);std=z.std(2,unbiased=False).clamp_min(EPS)
                zn=(z-mean[:,:,None])/std[:,:,None]
                q=torch.quantile(z,torch.tensor([.05,.5,.95],device=z.device),dim=2)
                return torch.stack((slog(mean/ref),torch.log1p(std/ref),slog(q[0]/ref),slog(q[1]/ref),slog(q[2]/ref),slog(zn.pow(3).mean(2)),torch.log1p(zn.pow(4).mean(2)),(zn.abs()>2).float().mean(2)),2)
            now=stats(x);cached=stats(c);out['distribution256']=torch.cat((now,now-cached),2).flatten(1)
        spectral=any(g.startswith('spectral_') for g in self.groups)
        if spectral:
            if self._shape!=tuple(x.shape[2:]) or self._matrix is None or self._matrix.device!=x.device:
                self._shape=tuple(x.shape[2:]);self._matrix=band_matrix(self._shape,x.device)
            mat=self._matrix;fx=torch.fft.fftn(x,dim=(2,3,4));fc=torch.fft.fftn(c,dim=(2,3,4))
            energy=lambda z:z.abs().square().flatten(2)@mat
            if 'spectral_drift512' in self.groups:
                out['spectral_drift512']=torch.log1p((energy(fx-fc)/energy(fc).clamp_min(EPS)).sqrt()).flatten(1)
            if 'spectral_phase512' in self.groups:
                valid=(fx.abs()>EPS)&(fc.abs()>EPS)
                disagreement=1-((fx/fx.abs().clamp_min(EPS))*(fc/fc.abs().clamp_min(EPS)).conj()).real.clamp(-1,1)
                out['spectral_phase512']=((disagreement*valid).flatten(2)@mat/((valid.float().flatten(2)@mat).clamp_min(1))).flatten(1)
            if 'spectral_shape576' in self.groups:
                en=energy(fx);ec=energy(fc);pn=en/en.sum(2,keepdim=True).clamp_min(EPS);pc=ec/ec.sum(2,keepdim=True).clamp_min(EPS)
                entropy=lambda p:-(p*p.clamp_min(1e-12).log()).sum(2)/math.log(32)
                hn,hc=entropy(pn),entropy(pc);kn=pn.topk(4,dim=2).values.sum(2);kc=pc.topk(4,dim=2).values.sum(2)
                out['spectral_shape576']=torch.cat(((pn-pc).flatten(1),stack([hn,hn-hc,kn,kn-kc])),1)
            if 'spectral_dynamics1024' in self.groups:
                fa=torch.fft.fftn(acc,dim=(2,3,4));fo=torch.fft.fftn(old,dim=(2,3,4));fi=torch.fft.fftn(innovation,dim=(2,3,4));fv=torch.fft.fftn(cv,dim=(2,3,4))
                out['spectral_dynamics1024']=torch.cat([torch.log1p((energy(a)/energy(b).clamp_min(EPS)).sqrt()).flatten(1) for a,b in ((fa,fo),(fi,fv))],1)
        for group,z in out.items():
            if z.shape!=(1,GROUPS[group][0]) or not torch.isfinite(z).all():raise ValueError(f'Invalid {group} feature')
        return out

    def commit(self, action):
        if not self.pending or action not in (0,1):raise ValueError('Commit one actual binary action after observe')
        if self.current_step in (0,49) and action!=0:raise ValueError('Native forced step must recompute')
        if action==0:self.cache=self.current
        self.previous2=self.previous;self.previous=self.current;self.previous_sigma=self.current_sigma
        self.previous_velocity=self.current_velocity;self.previous_ds=self.current_ds
        self.last_action=action;self.last_step=self.current_step;self.pending=False
