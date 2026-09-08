import argparse
import sys
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import Mock
PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from generation import load_prompts
from metrics import flops_for_calls
from protocol import prepare_resident, validate_wan21_t2v_1_3b_args


class ProtocolTests(unittest.TestCase):
    def test_standard_vbench200_prompt_manifest(self):
        rows=load_prompts(SimpleNamespace(prompt=None,prompts=PROJECT.parent/'Vbench200/prompts.jsonl'))
        self.assertEqual(len(rows),200)
        self.assertTrue(all(r['prompt']==r['prompt_en'] for r in rows))

    def test_resident_preparation_moves_both_large_models(self):
        p=SimpleNamespace(t5_cpu=False,device='cuda:0',model=Mock(),text_encoder=SimpleNamespace(model=Mock()))
        prepare_resident(p)
        p.model.to.assert_called_once_with('cuda:0')
        p.text_encoder.model.to.assert_called_once_with('cuda:0')
        p.t5_cpu=True
        with self.assertRaises(ValueError):prepare_resident(p)

    def test_fixed_protocol_rejects_offload(self):
        args=SimpleNamespace(task='t2v-1.3B',size='832*480',frame_num=81,sample_steps=50,
                             sample_solver='unipc',sample_shift=5.,sample_guide_scale=5.,base_seed=42,
                             offload_model=False,t5_cpu=False,t5_fsdp=False,dit_fsdp=False,
                             ulysses_size=1,ring_size=1,use_prompt_extend=False)
        validate_wan21_t2v_1_3b_args(args)
        args.offload_model=True
        with self.assertRaises(ValueError):validate_wan21_t2v_1_3b_args(args)

    def test_probe_cost_and_taylor_cached_arithmetic_are_not_zero(self):
        p={'per_model_forward':{'estimated_full_flops':310.,'estimated_always_on_flops':10.},
           'input':{'transformer_blocks':30,'hidden_dim':4,'seq_len':3}}
        allfull=[{'blocks_executed':30} for _ in range(100)]
        self.assertAlmostEqual(flops_for_calls(allfull,p,'baseline')*1e12,31000.)
        probe=[{'blocks_executed':30 if i<20 else 1} for i in range(100)]
        self.assertAlmostEqual(flops_for_calls(probe,p,'dicache')*1e12,7800.)
        cache=[{'blocks_executed':30 if i<2 else 0} for i in range(100)]
        # 2 full calls, then 98 order-zero cached calls with three residual additions.
        self.assertAlmostEqual(flops_for_calls(cache,p,'taylorseer')*1e12,2*310+98*(10+30*(5*12+24)))
        with self.assertRaises(ValueError):flops_for_calls(probe,p,'magcache')


if __name__=='__main__':unittest.main()
