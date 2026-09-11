import sys
from pathlib import Path
import torch
import torch.nn.functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from ours4wan21.latent_features import sea_filter
from model import GROUPS

CONTRACT=dict(version='three_branch_cnn_features_v1',input_shape=[16,21,60,104],
    input_quantization='float16_then_float32',direct_pool=[4,8,8],temporal_mean_pool=[8,8],
    temporal_variance_pool=[8,8],variance_correction=0,variance_before_spatial_pool=True,
    layout='all_roles_3d_then_each_role_mean_var_2d_then_sea7',
    sea_history='all_at_current_sigma',first_step='zero_new_features',
    normalizer='train_only_per_coordinate_population_mean_std_floor_1e-6',
    normalized_cache_dtype='float16',model_input_dtype='float32')

def pool(z):
    a=F.adaptive_avg_pool3d(z,(4,8,8)).flatten()
    b=torch.cat((F.adaptive_avg_pool2d(z.mean(2),(8,8)),F.adaptive_avg_pool2d(z.var(2,unbiased=False),(8,8))),1).flatten()
    return a,b

class History:
    def __init__(self):self.previous=None;self.cache=None;self.step=-1;self.pending=False;self.sigma=None
    @torch.no_grad()
    def observe(self,latent,step,sigma):
        if self.pending or step!=self.step+1 or not 0<=step<50:raise ValueError('consecutive observe/commit required')
        if not 0<=sigma<=1 or (self.sigma is not None and sigma>=self.sigma):raise ValueError('sigma order')
        x=latent.detach().to(torch.float16).float()
        if x.ndim==4:x=x.unsqueeze(0)
        if tuple(x.shape)!=(1,16,21,60,104) or not torch.isfinite(x).all():raise ValueError('latent shape/finite contract')
        self.current=x;self.step=step;self.sigma=sigma;self.pending=True
        if self.cache is None:return {g:x.new_zeros(r*6144) for g,r in GROUPS.items()}
        fields=[x,self.previous,self.cache];filtered=[sea_filter(z,sigma) for z in fields]
        variants=[fields,[x-self.previous,x-self.cache],filtered,[filtered[0]-filtered[1],filtered[0]-filtered[2]]]
        out={}
        for g,zs in zip(GROUPS,variants):
            parts=[pool(z) for z in zs]
            out[g]=torch.cat((torch.cat([p[0] for p in parts]),torch.cat([p[1] for p in parts])))
            if not torch.isfinite(out[g]).all():raise ValueError('feature nonfinite')
        return out
    def commit(self,action):
        if not self.pending or action not in (0,1) or (self.step in (0,49) and action):raise ValueError('invalid action/commit')
        if action==0:self.cache=self.current
        self.previous=self.current;self.pending=False
