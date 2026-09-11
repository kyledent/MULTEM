#!/usr/bin/env bash
# check_stat_paths.sh -- show that the statistics toggles change nothing, field by field.
#
#   bash tools/check_stat_paths.sh <test_multem binary> [test=T3] [outdir]
#   DEVICE=host bash tools/check_stat_paths.sh ...            (default: device)
#
# Runs the binary with MULTEM_STAT_DUMP=1 under each statistics setting and requires every [stat]
# line to match the default path. Output arrays cannot show this -- the program is not deterministic
# run to run -- but the statistics are computed on the host and must be bit-identical.
#
# The dump prints every field get_statistic() writes. That matters: an earlier version printed a
# subset, and a copy that missed three fields passed this very comparison.
#
# Exit 1 on any difference; 2 if a run printed no [stat] lines (a check that never ran).
set -uo pipefail

BIN="${1:?usage: $0 <test_multem binary> [test] [outdir]}"
T="${2:-T3}"
OUT="${3:-$(mktemp -d)}"
DEVICE="${DEVICE:-device}"
mkdir -p "${OUT}"

declare -A ARMS=(
    [off]=""
    [skip]="MULTEM_SKIP_REDUNDANT_STATS=1"
    [fast]="MULTEM_FAST_SETUP=1"
    [fast+skip]="MULTEM_FAST_SETUP=1 MULTEM_SKIP_REDUNDANT_STATS=1"
)

for arm in off skip fast fast+skip; do
    # shellcheck disable=SC2086
    env MULTEM_STAT_DUMP=1 ${ARMS[$arm]} "${BIN}" --test "${T}" --device "${DEVICE}" --precision float \
        --output "${OUT}" 2>&1 | grep -a '^\[stat\]' > "${OUT}/stat_${arm}.txt"
    n=$(wc -l < "${OUT}/stat_${arm}.txt")
    echo "${arm}: ${n} [stat] lines"
    [ "${n}" -gt 0 ] || { echo "FATAL: ${arm} printed no [stat] lines -- nothing was compared"; exit 2; }
done

rc=0
for arm in skip fast fast+skip; do
    if cmp -s "${OUT}/stat_off.txt" "${OUT}/stat_${arm}.txt"; then
        echo "off vs ${arm}: IDENTICAL"
    else
        echo "off vs ${arm}: DIFFER"
        diff "${OUT}/stat_off.txt" "${OUT}/stat_${arm}.txt" | head -8
        rc=1
    fi
done
exit ${rc}
