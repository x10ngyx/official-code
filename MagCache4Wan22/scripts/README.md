# scripts

prepare_wan22.sh creates a pristine pinned upstream checkout. validate_source.py checks source/package hashes. run_t2v_a14b.sh/.py run a baseline, original MagCache or original calibration under the fixed protocol.

`audit_baseline.py` checks common MagCache/SeaCache/TeaCache upstream hashes,
SeaCache cache-disabled sampler/model AST equivalence, the modulated-norm helper
refactor and unchanged attention/T5/VAE/scheduler/config dependencies. It does
not certify equality of unmeasured full A14B videos.
