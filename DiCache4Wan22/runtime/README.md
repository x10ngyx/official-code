# Runtime

`official.py` compiles the original DiCache cache-block AST into the native Wan2.2 forward and owns expert/CFG lifecycle. `generation.py` is the common fixed-protocol generation path. `inference_timing.py` records generate and component times, including probe execution. `batch.py`/`suite.py` provide persistent workers, strict resume and quality evaluation. `scan.py` selects parameters using measured latency. `common.py`/`prompts.py` validate sources, inputs and output layout.
