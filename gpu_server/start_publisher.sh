#!/usr/bin/env bash
# Start the Fast SAM 3D Body 2-camera publisher on the GPU PC.
#   ./start_publisher.sh                    # serials from ./cameras.txt (main camera first)
#   SERIALS=SN_MAIN,SN_SIDE ./start_publisher.sh --record
set -e
cd "$(dirname "$0")"
ENV=${ENV:-$HOME/miniconda3/envs/fast_sam_3d_body}
PY=${PY:-$ENV/bin/python}
# The env pins LD_LIBRARY_PATH to its pip nvjitlink (conda env var, applied only by `conda activate`).
# Without it the shell's /usr/local/cuda-12.2/lib64 breaks `import torch` (undefined __nvJitLinkComplete_12_4).
export LD_LIBRARY_PATH="$ENV/lib/python3.11/site-packages/nvidia/nvjitlink/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
SERIALS=${SERIALS:-$(grep -v '^#' cameras.txt 2>/dev/null | tr -d ' \n')}
if [ -z "$SERIALS" ]; then
    echo "no camera serials: run '$PY run_fastsam_multiview.py --list' and write 'SN_MAIN,SN_SIDE' into cameras.txt" >&2
    exit 1
fi
if ss -ltnH "sport = :5556" | grep -q .; then
    echo "port 5556 is already in use (a publisher is running?): pgrep -af run_fastsam" >&2
    exit 1
fi
exec "$PY" run_fastsam_multiview.py --serials "$SERIALS" "$@"
