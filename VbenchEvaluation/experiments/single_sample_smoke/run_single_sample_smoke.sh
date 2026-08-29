#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 SAMPLE_VIDEO RESULT_ROOT VBENCH_SOURCE" >&2
  exit 2
fi

SMOKE_SAMPLE_VIDEO=$(readlink -f "$1")
SMOKE_RESULT_ROOT=$(readlink -m "$2")
SMOKE_VBENCH_SOURCE=$(readlink -f "$3")
SMOKE_SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SMOKE_EVAL_DIR=$(cd -- "$SMOKE_SCRIPT_DIR/../.." && pwd)
SMOKE_PROJECT_DIR=$(cd -- "$SMOKE_EVAL_DIR/.." && pwd)
SMOKE_FULL_INFO_SOURCE="$SMOKE_PROJECT_DIR/Vbench200/VBench200_full_info.json"
SMOKE_PYTHON=${WAN22_PYTHON:-python}
SMOKE_LOCKED_COMMIT=fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VBENCH_CACHE_DIR=${VBENCH_CACHE_DIR:-$(readlink -f "$SMOKE_EVAL_DIR/weights")}
export PYTHONPATH="$SMOKE_VBENCH_SOURCE${PYTHONPATH:+:$PYTHONPATH}"

[[ -f "$SMOKE_SAMPLE_VIDEO" ]] || { echo "missing sample: $SMOKE_SAMPLE_VIDEO" >&2; exit 1; }
command -v "$SMOKE_PYTHON" >/dev/null 2>&1 || { echo "missing Python: $SMOKE_PYTHON" >&2; exit 1; }
[[ -f "$SMOKE_FULL_INFO_SOURCE" ]] || { echo "missing full-info: $SMOKE_FULL_INFO_SOURCE" >&2; exit 1; }
[[ -f "$SMOKE_VBENCH_SOURCE/vbench/__init__.py" ]] || { echo "invalid VBench source: $SMOKE_VBENCH_SOURCE" >&2; exit 1; }
[[ "$(git -C "$SMOKE_VBENCH_SOURCE" rev-parse HEAD)" == "$SMOKE_LOCKED_COMMIT" ]] || {
  echo "VBench source is not at locked commit $SMOKE_LOCKED_COMMIT" >&2
  exit 1
}

mkdir -p "$SMOKE_RESULT_ROOT/input" "$SMOKE_RESULT_ROOT/output"
SMOKE_ONE_ROW_INFO="$SMOKE_RESULT_ROOT/smoke_full_info.json"
SMOKE_METADATA="$SMOKE_RESULT_ROOT/sample_metadata.json"

SMOKE_FULL_INFO_SOURCE="$SMOKE_FULL_INFO_SOURCE" \
SMOKE_ONE_ROW_INFO="$SMOKE_ONE_ROW_INFO" \
SMOKE_METADATA="$SMOKE_METADATA" \
SMOKE_SAMPLE_VIDEO="$SMOKE_SAMPLE_VIDEO" \
SMOKE_VBENCH_SOURCE="$SMOKE_VBENCH_SOURCE" \
SMOKE_LOCKED_COMMIT="$SMOKE_LOCKED_COMMIT" \
"$SMOKE_PYTHON" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

source = Path(os.environ["SMOKE_FULL_INFO_SOURCE"])
records = json.loads(source.read_text(encoding="utf-8"))
record = next(row for row in records if "temporal_flickering" in row["dimension"])
record = dict(record)
record["dimension"] = ["temporal_flickering"]
Path(os.environ["SMOKE_ONE_ROW_INFO"]).write_text(
    json.dumps([record], ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
sample = Path(os.environ["SMOKE_SAMPLE_VIDEO"])
digest = hashlib.sha256(sample.read_bytes()).hexdigest()
metadata = {
    "purpose": "official VBench single-video execution-path smoke test",
    "benchmark_result": False,
    "dataset": "Vbench200",
    "selected_prompt": record["prompt_en"],
    "dimension": "temporal_flickering",
    "sample_source": str(sample),
    "sample_sha256": digest,
    "prompt_matches_sample": False,
    "device": "cpu",
    "vbench_source": os.environ["SMOKE_VBENCH_SOURCE"],
    "vbench_commit": os.environ["SMOKE_LOCKED_COMMIT"],
}
Path(os.environ["SMOKE_METADATA"]).write_text(
    json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(record["prompt_en"])
PY

SMOKE_PROMPT=$("$SMOKE_PYTHON" -c \
  'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))[0]["prompt_en"])' \
  "$SMOKE_ONE_ROW_INFO")
SMOKE_STAGED_VIDEO="$SMOKE_RESULT_ROOT/input/$SMOKE_PROMPT-0.mp4"
if [[ -e "$SMOKE_STAGED_VIDEO" || -L "$SMOKE_STAGED_VIDEO" ]]; then
  echo "refusing to replace existing staged video: $SMOKE_STAGED_VIDEO" >&2
  exit 1
fi
ln -s "$SMOKE_SAMPLE_VIDEO" "$SMOKE_STAGED_VIDEO"

ffprobe -v error \
  -show_entries format=filename,duration,size:stream=index,codec_type,codec_name,width,height,r_frame_rate,nb_frames \
  -of json "$SMOKE_SAMPLE_VIDEO" > "$SMOKE_RESULT_ROOT/sample_ffprobe.json"
env | rg '^(OPENBLAS_NUM_THREADS|OMP_NUM_THREADS|MKL_NUM_THREADS|NUMEXPR_NUM_THREADS|VBENCH_CACHE_DIR|PYTHONPATH)=' \
  | sort > "$SMOKE_RESULT_ROOT/runtime_environment.txt"

"$SMOKE_PYTHON" "$SMOKE_EVAL_DIR/evaluate_vbench.py" \
  --videos-dir "$SMOKE_RESULT_ROOT/input" \
  --output-dir "$SMOKE_RESULT_ROOT/output" \
  --full-info "$SMOKE_ONE_ROW_INFO" \
  --dimensions temporal_flickering \
  --name-prefix single_sample_smoke \
  --device cpu \
  2>&1 | tee "$SMOKE_RESULT_ROOT/run.log"
