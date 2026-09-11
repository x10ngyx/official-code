import json
from pathlib import Path

def calculate(roles):
    layers=[]
    def add(name, cin, cout, kernel, positions):
        layers.append(dict(name=name,macs=cin*cout*kernel*positions,parameters=cin*cout*kernel+cout))
    if roles is None:
        add('linear1',135,256,1,1);add('linear2',256,256,1,1);add('linear3',256,256,1,1);add('head',256,2,1,1)
        norm_params=3*2*256
    else:
        add('3d_stem',16*roles,32,1,4*8*8)
        add('3d_conv1',32,32,27,4*8*8)
        add('3d_conv2_stride122',32,64,27,4*4*4)
        # 3D pooled to 2x2x2: flatten512.
        add('2d_stem_mean_var',32*roles,32,1,8*8)
        add('2d_conv1',32,32,9,8*8)
        add('2d_conv2_stride22',32,64,9,4*4)
        # 2D pooled to 2x2: flatten256. Fuse 512+256.
        add('fusion',768,128,1,1)
        add('linear1_with_sea7',135,256,1,1);add('linear2',256,256,1,1);add('head',256,2,1,1)
        norm_params=2*2*256
    macs=sum(x['macs'] for x in layers)
    return dict(layers=layers,macs=macs,conv_linear_flops=2*macs,parameters=sum(x['parameters'] for x in layers)+norm_params)

out=Path('/mnt/hdd/xiongyuxiang/tmp/exp/ours21_cnn_compute_estimate_v1');out.mkdir(exist_ok=True)
r={k:calculate(v) for k,v in [('old_mlp',None),('G1_G3_three_states',3),('G2_G4_two_deltas',2)]}
for v in r.values():v['mac_ratio_vs_old']=v['macs']/r['old_mlp']['macs']
r['scope']='single-sample actor forward, Conv/Linear only, 2 FLOPs per MAC; excludes bias additions, activations, normalization, pooling, backward, optimizer, feature extraction and IO. Parameters include biases and MLP LayerNorm. CNN is an illustrative completion of unfrozen branch design, not measured latency.'
(out/'estimate.json').write_text(json.dumps(r,indent=2))
(out/'README.md').write_text('# Static CNN arithmetic estimate\n\nestimate.json includes layerwise MACs/parameters and explicit scope. Candidate network only; no training or GPU benchmark.\n')
print(json.dumps(r,indent=2))
