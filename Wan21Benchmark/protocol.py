"""Resident Wan21 preparation shared only by newly integrated method packages."""
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[3]
PACKAGES = Path(__file__).resolve().parent.parent
EXP_ROOT = Path('/all/yiran07-disk3/huteng_data/exp')
sys.path.insert(0,str(PACKAGES/'ComponentMetrics'))
from fixed_protocol import validate_wan21_t2v_1_3b_args


def source_lock(root):
    lock=json.loads((PACKAGES/'DiCache4Wan21/upstream_lock.json').read_text())['wan21']
    for name,sha in lock['compatibility_files'].items():
        if hashlib.sha256((Path(root)/name).read_bytes()).hexdigest()!=sha:
            raise ValueError('Wan21 source mismatch: '+name)


def external(path):
    path=Path(path).expanduser().resolve()
    if not path.is_relative_to(EXP_ROOT.resolve()):
        raise ValueError(f'output must be below {EXP_ROOT}')
    return path


def checkpoint(path):
    path=Path(path).expanduser().resolve(strict=True)
    if not path.is_relative_to((ROOT/'models').resolve()) or 'Wan2.1-T2V-1.3B' not in str(path):
        raise ValueError('checkpoint must be models/Wan2.1-T2V-1.3B')
    return path


def prepare_resident(pipeline):
    if pipeline.t5_cpu:
        raise ValueError('fixed protocol requires t5_cpu=False')
    pipeline.text_encoder.model.to(pipeline.device)
    pipeline.model.to(pipeline.device)
    # WanVAE is constructed on pipeline.device by the native Wan21 initializer.


@contextmanager
def resident_construction(wan_module):
    original=wan_module.WanT2V.__init__
    @wraps(original)
    def init(self,*args,**kwargs):
        original(self,*args,**kwargs)
        prepare_resident(self)
    wan_module.WanT2V.__init__=init
    try:
        yield
    finally:
        wan_module.WanT2V.__init__=original
