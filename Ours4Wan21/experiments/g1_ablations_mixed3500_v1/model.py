"""SEA7 MLP, larger-spatial G1 CNN, and original G1 features with a flat MLP."""
import sys
from pathlib import Path
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from ours4wan21.local_iql import build_mlp
GROUPS={'SEA7':0,'G1_CNN_16x20':3,'G1_MLP':3}
ARCH={
 'SEA7':dict(version='sea7_mlp_v1',input_dim=7,hidden=[256,256,256],activation='SiLU',layernorm=True,dropout=0.),
 'G1_CNN_16x20':dict(version='g1_cnn_16x20_v1',channels=[32,32,64],fusion_dim=128,head_hidden=[256,256],independent_iql_encoders=True,spatial_stride=2,temporal_stride=1,dropout=0.,direct_pool=[4,16,20],statistics_pool=[16,20],final_3d_pool=[2,2,2],final_2d_pool=[2,2]),
 'G1_MLP':dict(version='g1_flat_mlp_v1',input_dim=18439,hidden=[256,256,256],activation='SiLU',layernorm=True,dropout=0.,feature_pool=[4,8,8])}
def dimension(group):return {'SEA7':7,'G1_CNN_16x20':92167,'G1_MLP':18439}[group]
class CNN(torch.nn.Module):
    """Common constructor name retained for the shared IQL training loop."""
    def __init__(self,group,output_dim):
        super().__init__();self.group=group;self.out=output_dim
        if group!='G1_CNN_16x20':self.net=build_mlp(dimension(group),output_dim,256,3,0.);return
        self.n3=48*4*16*20;self.n2=96*16*20
        self.c3=nn.Sequential(nn.Conv3d(48,32,1),nn.SiLU(),nn.Conv3d(32,32,3,padding=1),nn.SiLU(),nn.Conv3d(32,64,3,stride=(1,2,2),padding=1),nn.SiLU(),nn.AdaptiveAvgPool3d((2,2,2)),nn.Flatten())
        self.c2=nn.Sequential(nn.Conv2d(96,32,1),nn.SiLU(),nn.Conv2d(32,32,3,padding=1),nn.SiLU(),nn.Conv2d(32,64,3,stride=2,padding=1),nn.SiLU(),nn.AdaptiveAvgPool2d((2,2)),nn.Flatten())
        self.fuse=nn.Sequential(nn.Linear(768,128),nn.SiLU());self.head=build_mlp(135,output_dim,256,2,0.)
    def forward(self,x):
        if self.group!='G1_CNN_16x20':y=self.net(x)
        else:
            a=self.c3(x[:,:self.n3].reshape(-1,48,4,16,20));b=self.c2(x[:,self.n3:self.n3+self.n2].reshape(-1,96,16,20))
            y=self.head(torch.cat((self.fuse(torch.cat((a,b),1)),x[:,-7:]),1))
        return y.squeeze(-1) if self.out==1 else y

def encode_normalized(raw,normalizer):
    z=((raw.float()-normalizer['mean'])/normalizer['std']).half()
    if not torch.isfinite(z).all():raise ValueError('normalized FP16 cache overflow/nonfinite')
    return z
class FrozenPolicy:
    def __init__(self,path,device='cpu'):
        p=torch.load(path,map_location='cpu',weights_only=False)
        if p['schema']!='ours21_g1_ablation_checkpoint_v1' or p['architecture']!=ARCH[p['group']]:raise ValueError('schema/architecture mismatch')
        self.device=torch.device(device);self.net=CNN(p['group'],2).to(device);self.net.load_state_dict(p['policy_net']);self.net.eval();self.normalizer={k:v.to(device) for k,v in p['normalizer'].items()}
    @torch.no_grad()
    def __call__(self,raw):return self.net(encode_normalized(raw.to(self.device),self.normalizer).float())
