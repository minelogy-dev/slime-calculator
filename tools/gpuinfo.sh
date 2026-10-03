#!/bin/bash
#
# gpuinfo.sh -- print a formatted snapshot of the GPU state.
#
# ASCII output only.  No GPU compute is started, so this is safe to run at any
# time, including while another job owns the device.
#
#   usage: gpuinfo.sh [options]
#     -1, --oneline     one machine-readable line (for loops / logs)
#     -w, --wait [SEC]  wait until the standard test state is reached
#     -c, --check       exit 1 if the standard test state is NOT met
#     -t, --temp N      required start temperature      (default $SLIME_START_TEMP or 35)
#         --sm N        required SM clock in MHz        (default $SLIME_SM_MHZ or 2340)
#         --mem N       required memory clock in MHz    (default $SLIME_MEM_MHZ or 9001)
#     -h, --help
#
# The "standard test state" is this project's measurement convention: the GPU is
# cooled to the start temperature (35 degC) and the clocks are locked (SM 2340 MHz,
# memory 9001 MHz), so that two measurements taken at different times are comparable.
#
set -uo pipefail

ONELINE=0; DO_WAIT=0; WAIT_SEC=600; DO_CHECK=0
T_TEMP="${SLIME_START_TEMP:-35}"
T_SM="${SLIME_SM_MHZ:-2340}"
T_MEM="${SLIME_MEM_MHZ:-9001}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -1|--oneline) ONELINE=1; shift ;;
        -w|--wait)    DO_WAIT=1; shift
                      [[ ${1:-} =~ ^[0-9]+$ ]] && { WAIT_SEC="$1"; shift; } ;;
        -c|--check)   DO_CHECK=1; shift ;;
        -t|--temp)    T_TEMP="$2"; shift 2 ;;
        --sm)         T_SM="$2"; shift 2 ;;
        --mem)        T_MEM="$2"; shift 2 ;;
        -h|--help)    sed -n '2,22p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi not found" >&2; exit 3
fi

snap() {
    nvidia-smi --query-gpu=name,driver_version,temperature.gpu,clocks.current.sm,clocks.current.memory,\
power.draw,power.limit,utilization.gpu,memory.used,memory.total \
        --format=csv,noheader,nounits 2>/dev/null | head -1
}

# strip leading/trailing whitespace only (keeps spaces inside the device name)
trim() { local s="$1"; s="${s#"${s%%[![:space:]]*}"}"; s="${s%"${s##*[![:space:]]}"}"; printf '%s' "$s"; }

read_snap() {
    local line; line="$(snap)"
    [[ -z "$line" ]] && { echo "cannot query GPU" >&2; exit 3; }
    IFS=',' read -r NAME DRV TEMP SM MEM PWR PLIM UTIL MUSD MTOT <<<"$line"
    NAME="$(trim "$NAME")"; DRV="$(trim "$DRV")"; TEMP="$(trim "$TEMP")"; SM="$(trim "$SM")"
    MEM="$(trim "$MEM")";   PWR="$(trim "$PWR")"; PLIM="$(trim "$PLIM")"; UTIL="$(trim "$UTIL")"
    MUSD="$(trim "$MUSD")"; MTOT="$(trim "$MTOT")"
}

if [[ $DO_WAIT -eq 1 ]]; then
    deadline=$(( $(date +%s) + WAIT_SEC ))
    while :; do
        read_snap
        [[ "${TEMP:-999}" -le "$T_TEMP" ]] 2>/dev/null && break
        [[ $(date +%s) -ge $deadline ]] && break
        sleep 3
    done
fi

read_snap

# temperature: "start at T" means the GPU must be at or below T (colder is fine).
# clocks: the host locks them, so they must match exactly.
if [[ "${TEMP:-9999}" -le "$T_TEMP" ]] 2>/dev/null; then S_TEMP="OK"; else S_TEMP="NOT MET (want <= $T_TEMP)"; fi
if [[ "$SM"  == "$T_SM"  ]]; then S_SM="OK";  else S_SM="NOT MET (want $T_SM)"; fi
if [[ "$MEM" == "$T_MEM" ]]; then S_MEM="OK"; else S_MEM="NOT MET (want $T_MEM)"; fi
if [[ "$S_TEMP" == "OK" && "$S_SM" == "OK" && "$S_MEM" == "OK" ]]; then
    STATE="READY (standard test state)"
else
    STATE="NOT READY (standard test state)"
fi

if [[ $ONELINE -eq 1 ]]; then
    printf 'gpu=%s driver=%s temp=%sC/%sC sm=%sMHz/%s mem=%sMHz/%s power=%sW/%sW util=%s%% vram=%sMiB/%sMiB state=%s\n' \
        "${NAME// /_}" "$DRV" "$TEMP" "$T_TEMP" "$SM" "$T_SM" "$MEM" "$T_MEM" "$PWR" "$PLIM" "$UTIL" "$MUSD" "$MTOT" \
        "$([[ "$STATE" == READY* ]] && echo READY || echo NOT_READY)"
else
    hr='--------------------------------------------------------------------------'
    echo "$hr"
    echo " GPU status"
    echo "$hr"
    printf '  %-14s %s\n'  "device"        "$NAME"
    printf '  %-14s %s\n'  "driver"        "$DRV"
    printf '  %-14s %s\n'  "temperature"   "${TEMP} C"
    printf '  %-14s %s\n'  "SM clock"      "${SM} MHz"
    printf '  %-14s %s\n'  "memory clock"  "${MEM} MHz"
    printf '  %-14s %s\n'  "power"         "${PWR} W / ${PLIM} W"
    printf '  %-14s %s\n'  "utilization"   "${UTIL} %"
    printf '  %-14s %s\n'  "memory"        "${MUSD} MiB / ${MTOT} MiB"
    local_thr="$(nvidia-smi --query-gpu=clocks_throttle_reasons.sw_power_cap,\
clocks_throttle_reasons.hw_slowdown,clocks_throttle_reasons.hw_thermal_slowdown \
        --format=csv,noheader 2>/dev/null | head -1)"
    [[ -n "$local_thr" ]] && printf '  %-14s %s\n' "throttling" "$local_thr"
    printf '  %-14s %s\n'  "state"         "$STATE"
    echo "$hr"
fi

if [[ $DO_CHECK -eq 1 && "$STATE" != READY* ]]; then
    exit 1
fi
exit 0
