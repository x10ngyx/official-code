"""Execute the exact formal Wan21 CNN, including all independent IQL encoders."""
from copy import deepcopy
from .shared import load

reference = load('formal_cnn', 'Ours4Wan21/experiments/cnn_mixed3500_v1/model.py')
ARCH = reference.ARCH
encode_normalized = reference.encode_normalized


class CNN(reference.CNN):
    def __init__(self, output_dim=2):
        super().__init__('G1', output_dim)


def networks(device='cpu'):
    nets = {name: CNN(1 if name == 'value_net' else 2).to(device)
            for name in ('value_net', 'q1_net', 'q2_net', 'policy_net')}
    for name in ('q1', 'q2'):
        nets['target_' + name] = deepcopy(nets[name + '_net']).eval().requires_grad_(False)
    return nets
