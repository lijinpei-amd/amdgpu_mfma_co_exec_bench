# XDL → PK_F32 issue-timing benchmark

Measures when VALU, PK_F32 and DOT instructions can issue after an MFMA on MI308X (gfx942) and MI350X (gfx950), and checks the timing against the shader programming guide on real hardware.

```
./run.sh > results_<arch>.txt 2>&1   # gfx942 or gfx950, detected with rocminfo; ~5 min (gfx942), ~10 min (gfx950)
ATT_LIB=/dir/with/librocprof-trace-decoder.so ./run.sh   # if the decoder isn't next to rocprofv3
```

- `coexec_bench.hip` contains three kinds of kernel. `-DGFX950` adds the MI350 XDL2x kernels.
  - `lat_*`: a 32-trip loop of {drain; MFMA; probe} in one wave, for ATT.
    - `lat_<mfma>_same`: the gap to the next identical MFMA (C).
    - `lat_<mfma>_q_{fma,pk,pk16,dot}`: the first `v_fma_f32`, `v_pk_fma_f32`, `v_pk_add_f16` and `v_dot2_f32_f16`.
    - The remaining `lat_*` kernels cover probe types, other PK opcodes, filler ladders and family switches.
  - `tput_*`: s_memtime-timed streams in one wave. `tput_win_*` replays `SIPreEmitPeephole`'s output from before and after its unpack-window fix.
  - `x_*`: two waves on one SIMD. One wave issues MFMAs and the other issues probes.
- `analyze_att.py` reads the `rocprofv3 --att` output and prints each instruction's issue cycle (time + stall) relative to the MFMA.
- Raw output:
  - [`results_gfx942.txt`](results_gfx942.txt): MI308X, gfx942, PTL off
  - [`results_gfx942_ptl.txt`](results_gfx942_ptl.txt): MI308X, PTL on
  - [`results_gfx950.txt`](results_gfx950.txt): MI350X, gfx950

The probe operands never depend on the MFMA, so the delays below come from the hardware issue rules, not from data dependencies. All values are cycles after the MFMA issues at cycle 0, measured with ATT. Each value has min = median = max over 124 reps (31 per wave × 4 waves). Earlier MI350X runs occasionally had a late rep, affecting the max only, which never reproduced. Each wave's first rep runs on a cold instruction cache and is skipped.

## MI350X (gfx950)

| Current MFMA (C=16) | Next same-family MFMA | Other-family MFMA | Plain VALU | First PK_F32 |
|---|---:|---:|---:|---:|
| Legacy XDL `16x16x16_bf16` | 16 = C | 20 = C+4 | 4 | **24 = C+8** |
| XDL2x `16x16x32_bf16` | 16 = C | 20 = C+4 | 8 | **28 = C+12** |

- **The PK_F32 rule holds for every shape, data type, and PK opcode tested.**
  - Legacy XDL: 4x4x4 (C=8) → 16, and 32x32x8 (C=32) → 40, so C+8.
  - XDL2x:
    - 32x32x16 BF16 (C=32) → 44
    - I8 16x16x64 (C=16) → 28
    - F8F6F4 fp8 (C=32) → 44
    - F8F6F4 fp4 (C=16) → 28

    All of these are C+12.
  - AGPR accumulators give the same results.
  - `v_pk_{fma,mul,add,mov}_f32` are all identical.
- **The window is measured in absolute cycles.** Independent VALU fillers between the MFMA and the PK don't move the PK earlier than 24 / 28.
- **VALU co-executes with XDL.** A plain VALU, including `v_nop`, issues in the next slot: cycle 4 after legacy XDL and cycle 8 after XDL2x (12 after fp8 F8F6F4). This holds both within one wave and from another wave.
- **SGEMM and DGEMM block all VALU until C+8.**
  - `v_mfma_f32_16x16x4_f32` (C=32) → 40
  - `v_mfma_f64_16x16x4_f64` (C=64) → 72
- **Throughput**, in s_memtime cycles per MFMA (the `min cyc/iter` column, i.e. the fastest wave):

  | Stream | Cycles/MFMA |
  |---|---:|
  | MFMA only | 16.2 |
  | MFMA + 1 or 2 FMA | 16.2 (free) |
  | legacy + PK | **28.2** = C+8+4 |
  | XDL2x + PK | **32.2** = C+12+4 |
  | legacy/XDL2x alternating | 20.1 = C+4 |

## MI308X (gfx942): legacy XDL only; VALU co-exec needs PTL

Whether VALU co-executes with XDL depends on the PTL setting:

- With PTL disabled, which is this machine's default, no VALU co-executes. See [`results_gfx942.txt`](results_gfx942.txt).
- After `sudo amd-smi set -g 0 --ptl-status 1`, non-packed VALU co-executes. See [`results_gfx942_ptl.txt`](results_gfx942_ptl.txt).

`amd-smi static` then reports `PTL_FORMAT: VECTOR,F8`.

| Current MFMA | Plain VALU, PTL off | Plain VALU, PTL on | First PK_F32 (either) |
|---|---:|---:|---:|
| `4x4x4_16b_bf16` (C=8) | 8 = C | **4** | 16 = C+8 |
| `16x16x16_bf16` (C=16) | 16 = C | **4** | 24 = C+8 |
| `32x32x8_bf16` (C=32) | 32 = C | **4** | 40 = C+8 |
| SGEMM / DGEMM `16x16x4` (C=32) | 40 = C+8 | 40 = C+8 | 40 = C+8 |

- **PK_F32 issues at C+8**, and PTL does not change it.
- **PTL off: no VALU co-executes.** No VALU instruction, not even `v_nop`, issues inside an XDL's C window, from the same wave or another wave. Only SALU co-issues, at cycle 4.
- **PTL on: the timings match MI350's legacy-XDL row.** This holds for every XDL data type, and for probes from the same wave and from another wave.
- **Throughput**, in s_memtime cycles per MFMA (the `min cyc/iter` column):

  | Stream | PTL off | PTL on |
  |---|---:|---:|
  | MFMA only | 16.2 | 16.2 |
  | MFMA + 1 FMA | 20.2 | **16.2** |
  | MFMA + 2 FMA | 24.2 | **16.2** |
  | MFMA + PK | 28.2 | 28.2 |

## Per-MFMA issue model (MI308X with PTL on, and MI350X)

These numbers come from the `lat_<mfma>_*` sweep and are in 4-clock issue cycles. V is the first VALU slot and P the first PK_F32 slot.

| MFMA class | V | P |
|---|---|---|
| XDL (legacy, XDL2x, SMFMAC) | 1 + ⌊widest A/B source in bytes / 16⌋ | NumPasses + 2, +1 for XDL2x (A ≥ 16 bytes) |
| SGEMM / DGEMM | NumPasses + 2 | NumPasses + 2 |

- **Scaled MFMAs** follow the table when they issue behind another MFMA. Issued into an idle MFMA pipe they start one slot late; see [Scaled MFMAs](#scaled-mfmas-mi350x).
- **Packed 16-bit ops** follow V. **DOT ops** follow P, both the VOP3P form (`v_dot2_f32_f16`) and the VOP2 accumulate form (`v_dot2c_f32_f16`, `v_dot4c_i32_i8`, `v_dot2c_f32_bf16`). DOTs also match PK_F32 when an MFMA follows them (it issues one slot later) and when they come from another wave (`x_mfma_dot`, `x_x2_dot`).
- **NumPasses** is the LLVM sched-model pass count, which matched the measured C for every MFMA.
- **Agreement:** apart from scaled MFMAs issued into an idle pipe, the model matches every measured cell on both chips.
- **LLVM:** the model is implemented as `AMDGPU::getVALUIssueCycle` / `getPackedVALUIssueCycle`, and `SIPreEmitPeephole` uses it to choose which PKs to unpack.

## Scaled MFMAs (MI350X)

`v_mfma_scale_*` carries an LD_SCALE. Whether it costs a slot depends on what the MFMA pipe is doing when the wave reaches it:

- **Behind another MFMA:** if the wave reaches the scaled MFMA at least 2 slots before an earlier MFMA frees the pipe, it stalls, and the LD_SCALE overlaps the stall. The scaled MFMA then times exactly like the unscaled one. This holds after XDL, XDL2x, scaled, SGEMM and DGEMM MFMAs.
- **Into an idle pipe:** otherwise the MFMA starts one slot late. V, P and the next MFMA all move 1 slot later, counted from the scaled MFMA's issue.

Next-MFMA gap (`lat_*_same`, the first MFMA issues into an idle pipe; `lat_x2fp8_x2scale`), in clocks:

| MFMA | Unscaled | Scaled, into idle pipe | Scaled, behind an MFMA |
|---|---:|---:|---:|
| 16x16x128 fp8 | 32 | 36 | 32 |
| 16x16x128 fp6 / fp4 | 16 | 20 | 16 |
| 32x32x64 fp8 | 64 | 68 | |
| 32x32x64 fp4 | 32 | 36 | |

V / P in slots, counted from the scaled MFMA's issue (`lat_x2scale*_q_*` into an idle pipe; `lat_<prev>_<scaled>_*` behind an MFMA). pk16 follows V, and VOP3P DOT and DOTnC follow P:

| Scaled MFMA | Into idle pipe | Behind unscaled | Behind scaled |
|---|---:|---:|---:|
| 16x16x128 fp8 | 4 / 12 | 3 / 11 | 3 / 11 |
| 16x16x128 fp6 | 3 / 8 | | |
| 16x16x128 fp4 | 3 / 8 | 2 / 7 | 2 / 7 |

- **Boundary:** `lat_x2fp8_fill<N>_x2scale_pk` puts N VALU fillers between an fp8 MFMA (pipe free at 32) and a scaled fp8 MFMA. With N ≤ 3 the wave reaches the scaled MFMA at 24 or earlier, stalls, and the PK issues at 76, 11 slots after the scaled MFMA. With N = 4 it arrives at 28 and issues at once, and with N = 5 at 32. For N ≥ 4 the PK is 12 slots after the scaled MFMA. The fp4 ladder (pipe free at 16) agrees.
- **Throughput:** streams run at the unscaled rate, since only the first MFMA issues into an idle pipe. One wave: 32.11 vs 32.09 clocks per fp8 MFMA, 16.17 vs 16.16 for fp6 and fp4, and 32.09 when fp8 and scaled fp8 alternate. Two waves both streaming MFMAs on one SIMD (`x_x2*_x2*`; the second wave mostly runs after the first, so its cyc/inst is twice the SIMD rate): 64.14 vs 64.11 per inst for fp8, 32.40 vs 32.22 for fp4.
- **LLVM:** `getVALUIssueCycle` / `getPackedVALUIssueCycle` model the behind-an-MFMA case. `SIPreEmitPeephole` uses P − V + 1, which is the same in both cases.

Unpack-window fix (`tput_win_*`), in cycles per MFMA:

| Stream | Before fix | After fix |
|---|---:|---:|
| legacy XDL + 4 PK, MI308X PTL on / MI350X | 36.2 | 32.2 |
| legacy XDL + 4 PK, MI308X PTL off | 36.2 | 40.2 |
| XDL2x + 4 PK, MI350X | 40.2 | 36.2 |
| F8F6F4 XDL2x (fp8) + 6 PK, MI350X | 56.2 | 52.2 |
| SGEMM + 4 PK, MI308X / MI350X | 68.2 | 56.2 |
| DGEMM + 4 PK, MI308X / MI350X | 68.2 / 104.2 | 56.2 / 88.2 |
| legacy XDL + 4 `v_pk_add_f16`, MI308X PTL on / MI350X | 20.2 | 20.2 |
| legacy XDL + 4 `v_pk_add_f16`, MI308X PTL off | 32.2 | 32.2 |
| legacy XDL + 4 `v_dot2_f32_f16`, all | 40.2 | 40.2 |

The pass never unpacks packed 16-bit ops or DOTs, so those rows don't change.

Unpack-window boundary (`tput_win3_*`, `tput_win5_*`), MI350X, in cycles per MFMA. Each stream has one PK more than the fixed pass unpacks. "More" also unpacks that last PK:

| Stream | V / P | Before fix | After fix | More |
|---|---|---:|---:|---:|
| legacy XDL 16x16x16 + 3 PK | 1 / 6 | 32.2 | 28.2 | 28.2 |
| legacy XDL 32x32x8 + 5 PK | 1 / 10 | 48.2 | 44.2 | 44.2 |
| XDL2x 16x16x32 + 3 PK | 2 / 7 | 36.2 | 32.2 | 32.2 |
| XDL2x 32x32x16 + 5 PK | 2 / 11 | 52.2 | 48.2 | 48.2 |
| F8F6F4 XDL2x (fp8) + 5 PK | 3 / 11 | 52.2 | 48.2 | 52.2 |

- **Model:** the k-th unpacked pair issues at V + 2k − 2 and V + 2k − 1, a packed PK no earlier than P, and the next MFMA one slot after the last VALU. Every cell is 4 × (the next MFMA's slot) + 0.2.
- **Boundary:** unpacking the last PK ties or, for fp8, loses a slot, so the fixed pass's P − V + 1 window is the largest that never hurts.

## ATT notes

- The gfx9 decoder loses the PC at `s_memtime`. `--att` mode branches around the timestamps.
- `--kernel-include-regex` matched nothing, so `run.sh` runs one kernel per process.
- The `librocprof-trace-decoder.so.0.2` shipped in the ROCm 7.14 wheels fails with `Error loading decoder: 37`, and rocprofv3 then hangs in its signal handler. Decoder 0.1.7 works; pass its directory as `ATT_LIB`.
