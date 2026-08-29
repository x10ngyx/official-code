#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 SAMPLE_VIDEO RESULT_ROOT VBENCH_SOURCE" >&2
  exit 2
fi

WEIGHTED_SAMPLE_VIDEO=$(readlink -f "$1")
WEIGHTED_RESULT_ROOT=$(readlink -m "$2")
WEIGHTED_VBENCH_SOURCE=$(readlink -f "$3")
WEIGHTED_SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WEIGHTED_EVAL_DIR=$(cd -- "$WEIGHTED_SCRIPT_DIR/../.." && pwd)
WEIGHTED_PROJECT_DIR=$(cd -- "$WEIGHTED_EVAL_DIR/.." && pwd)
WEIGHTED_FULL_INFO_SOURCE="$WEIGHTED_PROJECT_DIR/Vbench200/VBench200_full_info.json"
WEIGHTED_PYTHON=${WAN22_PYTHON:-python}
WEIGHTED_PYTHON_DEPS=${PYTHON_DEPS:-$WEIGHTED_RESULT_ROOT/python_deps}
WEIGHTED_LOCKED_COMMIT=fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490
WEIGHTED_RUN_ROOT="$WEIGHTED_RESULT_ROOT/weighted_clip"

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VBENCH_CACHE_DIR=${VBENCH_CACHE_DIR:-$(readlink -f "$WEIGHTED_EVAL_DIR/weights")}
export PYTHONPATH="$WEIGHTED_PYTHON_DEPS:$WEIGHTED_VBENCH_SOURCE${PYTHONPATH:+:$PYTHONPATH}"

[[ -f "$WEIGHTED_SAMPLE_VIDEO" ]] || { echo "missing sample: $WEIGHTED_SAMPLE_VIDEO" >&2; exit 1; }
command -v "$WEIGHTED_PYTHON" >/dev/null 2>&1 || { echo "missing Python: $WEIGHTED_PYTHON" >&2; exit 1; }
[[ -d "$WEIGHTED_PYTHON_DEPS/clip" ]] || { echo "missing isolated openai-clip dependency: $WEIGHTED_PYTHON_DEPS" >&2; exit 1; }
[[ "$(git -C "$WEIGHTED_VBENCH_SOURCE" rev-parse HEAD)" == "$WEIGHTED_LOCKED_COMMIT" ]] || {
  echo "VBench source is not at locked commit $WEIGHTED_LOCKED_COMMIT" >&2
  exit 1
}
[[ -f "$VBENCH_CACHE_DIR/clip_model/ViT-B-32.pt" ]] || {
  echo "missing local CLIP weight: $VBENCH_CACHE_DIR/clip_model/ViT-B-32.pt" >&2
  exit 1
}
[[ ! -e "$WEIGHTED_RUN_ROOT" ]] || {
  echo "refusing to replace existing weighted run: $WEIGHTED_RUN_ROOT" >&2
  exit 1
}

mkdir -p "$WEIGHTED_RUN_ROOT/input" "$WEIGHTED_RUN_ROOT/output"
WEIGHTED_DERIVED_VIDEO="$WEIGHTED_RUN_ROOT/source_clip_smoke_16f.mp4"
ffmpeg -v error -threads 1 -i "$WEIGHTED_SAMPLE_VIDEO" -an \
  -vf 'scale=320:180' -frames:v 16 -c:v libx264 -preset veryfast -crf 18 \
  "$WEIGHTED_DERIVED_VIDEO"

WEIGHTED_ONE_ROW_INFO="$WEIGHTED_RUN_ROOT/smoke_full_info.json"
WEIGHTED_FULL_INFO_SOURCE="$WEIGHTED_FULL_INFO_SOURCE" \
WEIGHTED_ONE_ROW_INFO="$WEIGHTED_ONE_ROW_INFO" \
"$WEIGHTED_PYTHON" - <<'PY'
import json
import os
from pathlib import Path

records = json.loads(Path(os.environ["WEIGHTED_FULL_INFO_SOURCE"]).read_text(encoding="utf-8"))
record = next(row for row in records if "background_consistency" in row["dimension"])
record = dict(record)
record["dimension"] = ["background_consistency"]
Path(os.environ["WEIGHTED_ONE_ROW_INFO"]).write_text(
    json.dumps([record], ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(record["prompt_en"])
PY

WEIGHTED_PROMPT=$("$WEIGHTED_PYTHON" -c \
  'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))[0]["prompt_en"])' \
  "$WEIGHTED_ONE_ROW_INFO")
ln -s "$WEIGHTED_DERIVED_VIDEO" "$WEIGHTED_RUN_ROOT/input/$WEIGHTED_PROMPT-0.mp4"

ffprobe -v error \
  -show_entries format=filename,duration,size:stream=index,codec_type,codec_name,width,height,r_frame_rate,nb_frames \
  -of json "$WEIGHTED_DERIVED_VIDEO" > "$WEIGHTED_RUN_ROOT/sample_ffprobe.json"
sha256sum "$WEIGHTED_SAMPLE_VIDEO" "$WEIGHTED_DERIVED_VIDEO" \
  "$VBENCH_CACHE_DIR/clip_model/ViT-B-32.pt" > "$WEIGHTED_RUN_ROOT/SHA256SUMS"
"$WEIGHTED_PYTHON" -m pip show openai-clip > "$WEIGHTED_RUN_ROOT/openai_clip_dependency.txt"

"$WEIGHTED_PYTHON" "$WEIGHTED_EVAL_DIR/evaluate_vbench.py" \
  --videos-dir "$WEIGHTED_RUN_ROOT/input" \
  --output-dir "$WEIGHTED_RUN_ROOT/output" \
  --full-info "$WEIGHTED_ONE_ROW_INFO" \
  --dimensions background_consistency \
  --name-prefix single_sample_weighted_smoke \
  --device cpu \
  2>&1 | tee "$WEIGHTED_RUN_ROOT/run.log"
