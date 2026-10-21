#!/usr/bin/env bash
# Build, run the s_memtime kernels, then ATT-trace the latency kernels and the
# cross-wave kernels (one process per kernel) and print issue cycles.
#
#   ./run.sh                       # gfx942 or gfx950, detected with rocminfo
#   ATT_LIB=/dir/with/librocprof-trace-decoder.so ./run.sh
set -euo pipefail
cd "$(dirname "$0")"

export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}
ARCH=${ARCH:-$(rocminfo | awk 'match($0, /gfx9[0-9a-f]+/) { a = substr($0, RSTART, RLENGTH) } END { print a }')}
ATT_LIB=${ATT_LIB:-$(dirname "$(command -v rocprofv3)")/../lib}
if ! ls "$ATT_LIB"/librocprof-trace-decoder.so* >/dev/null 2>&1; then
  echo "librocprof-trace-decoder.so not found in $ATT_LIB; set ATT_LIB" >&2
  exit 1
fi
FLAGS=()
[[ $ARCH == gfx950 ]] && FLAGS+=(-DGFX950)

hipcc --offload-arch="$ARCH" -O2 "${FLAGS[@]}" -o coexec_bench coexec_bench.hip

./coexec_bench tput
./coexec_bench x | tail -n +2

rm -rf att
mkdir att
xk="x_mfma_fma x_mfma_pkfma x_mfma_pkaddf16 x_mfma_dot"
[[ $ARCH == gfx950 ]] && xk+=" x_x2_fma x_x2_pkfma x_x2_dot x_x2fp8_x2fp8 x_x2scale_x2scale x_x2fp4_x2fp4 x_x2scalefp4_x2scalefp4"
kernels="$(./coexec_bench lat --att | sed -n 's/^ran //p') $xk"
for k in $kernels; do
  if ! rocprofv3 --att --att-library-path "$ATT_LIB" --att-target-cu 1 \
      -d "att/$k" -o run -- ./coexec_bench "$k" --att >"att/$k.log" 2>&1; then
    echo "ATT capture failed for $k; see att/$k.log" >&2
  fi
done

echo
python3 analyze_att.py $(for k in $kernels; do echo "att/$k"; done)
