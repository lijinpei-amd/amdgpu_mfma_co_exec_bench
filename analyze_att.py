#!/usr/bin/env python3
"""Report instruction issue cycles relative to MFMAs from ATT output.

Reads the ui_output_agent_*_dispatch_* directories that rocprofv3 --att writes.

lat_* kernels loop over `s_nop` drains, each followed by a probe group that
starts with an MFMA and ends at the next `s_nop`. For every instruction in a
group, prints the issue cycle (time + stall) minus the MFMA's issue cycle,
aggregated over all reps and all traced waves. Each wave's first rep runs on a
cold instruction cache and is skipped.

x_* and tput_* kernels are streams. Waves on one SIMD whose lifetimes overlap
(one workgroup) are merged by issue cycle, then three histograms are printed:
MFMA-to-MFMA issue gaps, probe-to-probe gaps, and each probe's issue cycle
minus the latest MFMA issue (only for probes issued while that workgroup's
MFMA stream is running).

Usage: analyze_att.py ATT_DIR [ATT_DIR ...]
"""

import bisect
import collections
import glob
import json
import os
import sys

_PROBES = ("v_fma_f32", "v_pk_", "v_dot")
_MFMAS = ("v_mfma", "v_smfmac")

Wave = collections.namedtuple("Wave", "begin end insts")


def _load_dispatch(path):
    """Returns (kernel_name, {(se, simd): [Wave]}).

    Wave.insts is a list of (issue, stall, asm).
    """
    with open(os.path.join(path, "code.json")) as f:
        code = json.load(f)["code"] or []
    asm = {row[2]: row[0] for row in code}
    kernel = code[0][3] if code else "?"
    simds = collections.defaultdict(list)
    for wf in sorted(glob.glob(os.path.join(path, "se*_sm*_sl*_wv*.json"))):
        se, sm = os.path.basename(wf).split("_")[:2]
        with open(wf) as f:
            wave = json.load(f)["wave"]
        insts = [(t + stall, stall, asm.get(idx, "?"))
                 for t, _cat, stall, _dur, idx in wave["instructions"]]
        simds[(int(se[2:]), int(sm[2:]))].append(
            Wave(wave["begin"], wave["end"], insts))
    return kernel, simds


def _groups(insts):
    """Yields each probe group: an MFMA up to (not including) the next s_nop.

    A group not closed by an s_nop is dropped.
    """
    group = []
    for inst in insts:
        if inst[2].startswith("s_nop"):
            if group:
                yield group
            group = []
        elif group or inst[2].startswith(_MFMAS):
            group.append(inst)


def _sessions(waves):
    """Merges waves with overlapping lifetimes; yields their sorted insts."""
    waves = sorted(waves)
    i = 0
    while i < len(waves):
        end, members = waves[i].end, [waves[i]]
        i += 1
        while i < len(waves) and waves[i].begin <= end:
            end = max(end, waves[i].end)
            members.append(waves[i])
            i += 1
        yield sorted(x for w in members for x in w.insts)


def _hist(vals, limit=12):
    c = collections.Counter(vals)
    top = sorted(c.items(), key=lambda kv: -kv[1])[:limit]
    return "  ".join(f"{k}:{v}" for k, v in sorted(top)) or "-"


def analyze_stream(kernel, path, simds):
    mfma_gap, probe_gap, since_mfma = [], [], []
    n_in = n_out = 0
    for waves in simds.values():
        for merged in _sessions(waves):
            mfmas = [t for t, _, a in merged if a.startswith(_MFMAS)]
            probes = [t for t, _, a in merged if a.startswith(_PROBES)]
            mfma_gap += [b - a for a, b in zip(mfmas, mfmas[1:])]
            probe_gap += [b - a for a, b in zip(probes, probes[1:])]
            for t in probes:
                if mfmas and mfmas[0] <= t < mfmas[-1]:
                    since_mfma.append(t - mfmas[bisect.bisect_right(mfmas, t) - 1])
                    n_in += 1
                else:
                    n_out += 1
    print(f"== {kernel}  ({os.path.basename(path)}, {len(simds)} SIMDs)")
    print(f"  MFMA->MFMA gap      {_hist(mfma_gap)}")
    print(f"  probe->probe gap    {_hist(probe_gap)}")
    print(f"  probes inside MFMA stream: {n_in}, outside: {n_out}")
    low = f" (min {min(since_mfma)})" if since_mfma else ""
    print(f"  probe - last MFMA{low}   {_hist(since_mfma)}")
    print()


def analyze_lat(kernel, path, simds):
    waves = [w for ws in simds.values() for w in ws]
    # position -> asm, list of (delta from MFMA issue, stall)
    rows = collections.OrderedDict()
    n = 0
    for wave in waves:
        for group in list(_groups(wave.insts))[1:]:
            t0 = group[0][0]
            n += 1
            for pos, (issue, stall, text) in enumerate(group):
                rows.setdefault(pos, (text, []))[1].append((issue - t0, stall))
    print(f"== {kernel}  ({os.path.basename(path)}, {len(waves)} waves, "
          f"{n} reps)")
    print(f"  {'pos':>3}  {'issue@ (min/med/max)':>20}  {'stall med':>9}  asm")
    for pos, (text, vals) in rows.items():
        d = sorted(v[0] for v in vals)
        s = sorted(v[1] for v in vals)
        print(f"  {pos:>3}  {d[0]:>6}/{d[len(d)//2]:>5}/{d[-1]:>5}  "
              f"{s[len(s)//2]:>9}  {text}")
    print()


def analyze(path):
    """Returns False if the dispatch holds no traced waves."""
    kernel, simds = _load_dispatch(path)
    if not simds:
        return False
    if kernel.startswith("lat_"):
        analyze_lat(kernel, path, simds)
    else:
        analyze_stream(kernel, path, simds)
    return True


def main(argv):
    if len(argv) < 2:
        sys.exit(__doc__)
    for root in argv[1:]:
        dirs = sorted(glob.glob(os.path.join(root, "**", "ui_output_agent_*"),
                                recursive=True))
        if not any([analyze(d) for d in dirs]):
            print(f"== {root}: no ATT data\n")


if __name__ == "__main__":
    main(sys.argv)
