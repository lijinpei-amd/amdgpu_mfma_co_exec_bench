// Micro benchmark for MFMA (XDL) -> VALU / PK_F32 co-execution on gfx942 and
// gfx950. Build with -DGFX950 for gfx950 to add the MI350 XDL2x kernels.
//
// Three kinds of kernels:
//   lat_*  : a 32-trip loop of { drain; MFMA; probe...; s_nop 0 } in one
//            wave. Meant to be captured with rocprofv3 --att; analyze_att.py
//            reports the issue cycle of every probe instruction relative to
//            the MFMA, skipping the first trip (cold instruction cache).
//   tput_* : one wave, 256 back-to-back iterations of a pattern bracketed by
//            s_memtime. Reports cycles per iteration without needing a trace.
//   x_*    : two waves per SIMD, one streaming MFMAs and one streaming probes,
//            to see whether a probe from another wave co-executes.
//
// No probe reads or writes a register the MFMA writes, and no instruction
// writes a register another one reads, so any delay seen after an MFMA comes
// from the issue rules, not from data hazards.
//
// Usage: coexec_bench [lat|tput|x|KERNEL_NAME] [--att]
//   --att skips s_memtime (the ATT decoder loses the PC at s_memtime) and
//   launches every kernel once.

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(x)                                                              \
  do {                                                                        \
    hipError_t e_ = (x);                                                      \
    if (e_ != hipSuccess) {                                                   \
      fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x,               \
              hipGetErrorString(e_));                                         \
      exit(1);                                                                \
    }                                                                         \
  } while (0)

// ---------------------------------------------------------------------------
// Instruction snippets. Register map (all clobbered):
//   v[0:3]   legacy A/B inputs        v[4:7]   PK_F32 / VALU sources
//   v[8:11], v[24:27], v[28:31], v[48:51]   4-VGPR accumulators
//   v[32:47] 16-VGPR accumulator      v[12:23] PK_F32 / VALU destinations
//   v[56:63] XDL2x A/B inputs         v[64:79] F8F6F4 A/B inputs
//   v[80:95] second 16-VGPR accumulator
//   Read-only shared sources: v20 is the SMFMAC index and v80/v81 are the
//   scaled-MFMA scales. Don't pair SM_* with PK_MOV/SEQ6_*, which write v20,
//   or X2_SCALE* with an MFMA accumulating into v[80:95]. The 32x32 scaled
//   MFMAs accumulate into v[32:47] and v[48:63] instead.
// ---------------------------------------------------------------------------
#define S(x) x "\n"

// 64 wait states (~256 cycles): lets the previous MFMA retire so every rep
// starts clean.
#define DRAIN S("s_nop 15") S("s_nop 15") S("s_nop 15") S("s_nop 15")

// Legacy XDL (MI300 opcodes). Passes from the CDNA3/CDNA4 ISA guides.
#define MFMA_4X4    S("v_mfma_f32_4x4x4_16b_bf16 v[8:11], v[0:1], v[2:3], v[8:11]")  // 2
#define MFMA_16     S("v_mfma_f32_16x16x16_bf16 v[8:11], v[0:1], v[2:3], v[8:11]")   // 4
#define MFMA_16_B   S("v_mfma_f32_16x16x16_bf16 v[24:27], v[0:1], v[2:3], v[24:27]")
#define MFMA_16_C   S("v_mfma_f32_16x16x16_bf16 v[28:31], v[0:1], v[2:3], v[28:31]")
#define MFMA_16_D   S("v_mfma_f32_16x16x16_bf16 v[48:51], v[0:1], v[2:3], v[48:51]")
#define MFMA_32     S("v_mfma_f32_32x32x8_bf16 v[32:47], v[0:1], v[2:3], v[32:47]")   // 8
#define MFMA_F16    S("v_mfma_f32_16x16x16_f16 v[8:11], v[0:1], v[2:3], v[8:11]")    // 4
#define MFMA_I8     S("v_mfma_i32_16x16x32_i8 v[8:11], v[0:1], v[2:3], v[8:11]")     // 4
#define MFMA_FP8    S("v_mfma_f32_16x16x32_fp8_fp8 v[8:11], v[0:1], v[2:3], v[8:11]")
#define MFMA_XF32   S("v_mfma_f32_16x16x8_xf32 v[8:11], v[0:1], v[2:3], v[8:11]")    // gfx942 only
// SGEMM / DGEMM.
#define MFMA_F32    S("v_mfma_f32_16x16x4_f32 v[8:11], v0, v1, v[8:11]")
#define MFMA_F64    S("v_mfma_f64_16x16x4_f64 v[32:39], v[0:1], v[2:3], v[32:39]")
#define MFMA_F64_4X4 S("v_mfma_f64_4x4x4_4b_f64 v[32:33], v[0:1], v[2:3], v[32:33]")
#define MFMA_F64_4X4_B S("v_mfma_f64_4x4x4_4b_f64 v[34:35], v[0:1], v[2:3], v[34:35]")
// AGPR accumulator form: C/D live in the separate AccVGPR file.
#define MFMA_16_ACC S("v_mfma_f32_16x16x16_bf16 a[0:3], v[0:1], v[2:3], a[0:3]")

// MI350 XDL2x (gfx950 only).
#define X2_16       S("v_mfma_f32_16x16x32_bf16 v[8:11], v[56:59], v[60:63], v[8:11]")
#define X2_16_B     S("v_mfma_f32_16x16x32_bf16 v[24:27], v[56:59], v[60:63], v[24:27]")
#define X2_16_C     S("v_mfma_f32_16x16x32_bf16 v[28:31], v[56:59], v[60:63], v[28:31]")
#define X2_16_D     S("v_mfma_f32_16x16x32_bf16 v[48:51], v[56:59], v[60:63], v[48:51]")
#define X2_32       S("v_mfma_f32_32x32x16_bf16 v[32:47], v[56:59], v[60:63], v[32:47]")
#define X2_I8       S("v_mfma_i32_16x16x64_i8 v[8:11], v[56:59], v[60:63], v[8:11]")
#define X2_I8_B     S("v_mfma_i32_16x16x64_i8 v[24:27], v[56:59], v[60:63], v[24:27]")
#define X2_FP8      S("v_mfma_f32_16x16x128_f8f6f4 v[8:11], v[64:71], v[72:79], v[8:11]")
#define X2_FP8_B    S("v_mfma_f32_16x16x128_f8f6f4 v[24:27], v[64:71], v[72:79], v[24:27]")
#define X2_FP4      S("v_mfma_f32_16x16x128_f8f6f4 v[8:11], v[64:67], v[72:75], v[8:11] cbsz:4 blgp:4")
#define X2_FP4_B    S("v_mfma_f32_16x16x128_f8f6f4 v[24:27], v[64:67], v[72:75], v[24:27] cbsz:4 blgp:4")
#define X2_32_B     S("v_mfma_f32_32x32x16_bf16 v[80:95], v[56:59], v[60:63], v[80:95]")
#define X2_16_ACC   S("v_mfma_f32_16x16x32_bf16 a[0:3], v[56:59], v[60:63], a[0:3]")
#define X2_F16      S("v_mfma_f32_16x16x32_f16 v[8:11], v[56:59], v[60:63], v[8:11]")
#define X2_F16_B    S("v_mfma_f32_16x16x32_f16 v[24:27], v[56:59], v[60:63], v[24:27]")
#define X2_32F16    S("v_mfma_f32_32x32x16_f16 v[32:47], v[56:59], v[60:63], v[32:47]")
#define X2_32F16_B  S("v_mfma_f32_32x32x16_f16 v[80:95], v[56:59], v[60:63], v[80:95]")
#define X2_32I8     S("v_mfma_i32_32x32x32_i8 v[32:47], v[56:59], v[60:63], v[32:47]")
#define X2_32I8_B   S("v_mfma_i32_32x32x32_i8 v[80:95], v[56:59], v[60:63], v[80:95]")
#define X2_FP6      S("v_mfma_f32_16x16x128_f8f6f4 v[8:11], v[64:69], v[72:77], v[8:11] cbsz:2 blgp:2")
#define X2_FP6_B    S("v_mfma_f32_16x16x128_f8f6f4 v[24:27], v[64:69], v[72:77], v[24:27] cbsz:2 blgp:2")
#define X2_F8F4     S("v_mfma_f32_16x16x128_f8f6f4 v[8:11], v[64:71], v[72:75], v[8:11] blgp:4")
#define X2_F8F4_B   S("v_mfma_f32_16x16x128_f8f6f4 v[24:27], v[64:71], v[72:75], v[24:27] blgp:4")
#define X2_32FP8    S("v_mfma_f32_32x32x64_f8f6f4 v[32:47], v[64:71], v[72:79], v[32:47]")
#define X2_32FP8_B  S("v_mfma_f32_32x32x64_f8f6f4 v[80:95], v[64:71], v[72:79], v[80:95]")
#define X2_32FP4    S("v_mfma_f32_32x32x64_f8f6f4 v[32:47], v[64:67], v[72:75], v[32:47] cbsz:4 blgp:4")
#define X2_32FP4_B  S("v_mfma_f32_32x32x64_f8f6f4 v[80:95], v[64:67], v[72:75], v[80:95] cbsz:4 blgp:4")
#define X2_SCALE    S("v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[64:71], v[72:79], v[8:11], v80, v81")
#define X2_SCALE_B  S("v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[64:71], v[72:79], v[24:27], v80, v81")
#define X2_SCALE_C  S("v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[64:71], v[72:79], v[28:31], v80, v81")
#define X2_SCALE_D  S("v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[64:71], v[72:79], v[48:51], v80, v81")
#define X2_SCALE_FP6   S("v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[64:69], v[72:77], v[8:11], v80, v81 cbsz:2 blgp:2")
#define X2_SCALE_FP6_B S("v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[64:69], v[72:77], v[24:27], v80, v81 cbsz:2 blgp:2")
#define X2_SCALE_FP6_C S("v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[64:69], v[72:77], v[28:31], v80, v81 cbsz:2 blgp:2")
#define X2_SCALE_FP6_D S("v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[64:69], v[72:77], v[48:51], v80, v81 cbsz:2 blgp:2")
#define X2_SCALE_FP4   S("v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[64:67], v[72:75], v[8:11], v80, v81 cbsz:4 blgp:4")
#define X2_SCALE_FP4_B S("v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[64:67], v[72:75], v[24:27], v80, v81 cbsz:4 blgp:4")
#define X2_SCALE_FP4_C S("v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[64:67], v[72:75], v[28:31], v80, v81 cbsz:4 blgp:4")
#define X2_SCALE_FP4_D S("v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[64:67], v[72:75], v[48:51], v80, v81 cbsz:4 blgp:4")
#define X2_SCALE32     S("v_mfma_scale_f32_32x32x64_f8f6f4 v[32:47], v[64:71], v[72:79], v[32:47], v80, v81")
#define X2_SCALE32_B   S("v_mfma_scale_f32_32x32x64_f8f6f4 v[48:63], v[64:71], v[72:79], v[48:63], v80, v81")
#define X2_SCALE32_FP4   S("v_mfma_scale_f32_32x32x64_f8f6f4 v[32:47], v[64:67], v[72:75], v[32:47], v80, v81 cbsz:4 blgp:4")
#define X2_SCALE32_FP4_B S("v_mfma_scale_f32_32x32x64_f8f6f4 v[48:63], v[64:67], v[72:75], v[48:63], v80, v81 cbsz:4 blgp:4")
#define SM_16BF16   S("v_smfmac_f32_16x16x64_bf16 v[8:11], v[56:59], v[64:71], v20")
#define SM_16BF16_B S("v_smfmac_f32_16x16x64_bf16 v[24:27], v[56:59], v[64:71], v20")
#define SM_32BF16   S("v_smfmac_f32_32x32x32_bf16 v[32:47], v[56:59], v[64:71], v20")
#define SM_32BF16_B S("v_smfmac_f32_32x32x32_bf16 v[80:95], v[56:59], v[64:71], v20")
#define SM_16FP8    S("v_smfmac_f32_16x16x128_fp8_fp8 v[8:11], v[56:59], v[64:71], v20")
#define SM_16FP8_B  S("v_smfmac_f32_16x16x128_fp8_fp8 v[24:27], v[56:59], v[64:71], v20")
#define SM_32FP8    S("v_smfmac_f32_32x32x64_fp8_fp8 v[32:47], v[56:59], v[64:71], v20")
#define SM_32FP8_B  S("v_smfmac_f32_32x32x64_fp8_fp8 v[80:95], v[56:59], v[64:71], v20")
// Legacy sparse XDL (both targets).
#define SM_LEG      S("v_smfmac_f32_16x16x32_f16 v[8:11], v[0:1], v[56:59], v20")
#define SM_LEG_B    S("v_smfmac_f32_16x16x32_f16 v[24:27], v[0:1], v[56:59], v20")

#define PK_FMA      S("v_pk_fma_f32 v[12:13], v[4:5], v[6:7], v[14:15]")
#define PK_MUL      S("v_pk_mul_f32 v[16:17], v[4:5], v[6:7]")
#define PK_ADD      S("v_pk_add_f32 v[18:19], v[4:5], v[6:7]")
#define PK_MOV      S("v_pk_mov_b32 v[20:21], v[4:5], v[6:7] op_sel:[0,1]")
#define FMA_F32     S("v_fma_f32 v22, v4, v6, v7")
#define FMA_F32_B   S("v_fma_f32 v23, v5, v6, v7")
#define FILL        S("v_mov_b32 v22, v5")
#define VNOP        S("v_nop")                          // touches no VGPR
#define ADD_U32     S("v_add_u32 v22, v4, v5")          // 4-byte VOP2
#define SALU        S("s_mov_b32 s0, 0")
#define MOV_FAR     S("v_mov_b32 v100, s0")             // far from every MFMA operand
// Packed 16-bit and other never-coissue VALU (SIInstrInfo::isNeverCoissue).
#define PK_ADD_F16  S("v_pk_add_f16 v22, v4, v5")
#define PK_FMA_F16  S("v_pk_fma_f16 v22, v4, v5, v6")
#define PK_MUL_F16  S("v_pk_mul_f16 v22, v4, v5")
#define PK_FMA_F16_B S("v_pk_fma_f16 v23, v4, v5, v6")
#define PK_MUL_F16_C S("v_pk_mul_f16 v100, v4, v5")
#define PK16X3      PK_ADD_F16 PK_FMA_F16_B PK_MUL_F16_C
#define PK_ADD_U16  S("v_pk_add_u16 v22, v4, v5")
#define PK_MUL_LO   S("v_pk_mul_lo_u16 v22, v4, v5")
#define DOT2_F16    S("v_dot2_f32_f16 v22, v4, v5, v6")
#define EXP_F32     S("v_exp_f32 v22, v4")
#define CVT_PK_FP8  S("v_cvt_pk_fp8_f32 v22, v4, v5")
#define DOT2C_F16   S("v_dot2c_f32_f16 v22, v4, v5")
#define DOT2_I16    S("v_dot2_i32_i16 v22, v4, v5, v6")
#define DOT4_I8     S("v_dot4_i32_i8 v22, v4, v5, v6")
// VOP2 DOTnC: IsDOT in LLVM but, unlike the VOP3P DOTs, not IsPacked.
#define DOT4C_I8    S("v_dot4c_i32_i8 v22, v4, v5")
#define DOT2C_BF16  S("v_dot2c_f32_bf16 v22, v4, v5")   // gfx950 only

#define REP(n, body) S(".rept " #n) body S(".endr")

// Fill v0..v95 with 0x3c3c3c3c. Read as any MFMA input type this is a normal
// number (bf16 ~0.011, f16 ~1.06, fp8 0.75 FNUZ / 1.5 OCP, f32 ~0.011), never
// NaN or denormal. Byte 0x80 must be avoided: it is NaN in gfx942 FNUZ fp8.
// The exception is X2_SCALE, whose E8M0 scales (0x3c = 2^-67) make its products
// f32-denormal. Its kernels only measure timing.
#define INIT                                                                  \
  S("s_mov_b32 s0, 0x3c3c3c3c")                                               \
  S(".irp r, 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39") \
  S("v_mov_b32 v\\r, s0") S(".endr")                                          \
  S(".irp r, 40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79") \
  S("v_mov_b32 v\\r, s0") S(".endr")                                          \
  S(".irp r, 80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95")                \
  S("v_mov_b32 v\\r, s0") S(".endr")                                          \
  S("v_accvgpr_write_b32 a0, v0") S("v_accvgpr_write_b32 a1, v0")           \
  S("v_accvgpr_write_b32 a2, v0") S("v_accvgpr_write_b32 a3, v0")           \
  DRAIN

#define CLOBBERS                                                              \
  "s0", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",    \
      "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20",   \
      "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30",   \
      "v31", "v32", "v33", "v34", "v35", "v36", "v37", "v38", "v39", "v40",   \
      "v41", "v42", "v43", "v44", "v45", "v46", "v47", "v48", "v49", "v50",   \
      "v51", "v52", "v53", "v54", "v55", "v56", "v57", "v58", "v59", "v60",   \
      "v61", "v62", "v63", "v64", "v65", "v66", "v67", "v68", "v69", "v70",   \
      "v71", "v72", "v73", "v74", "v75", "v76", "v77", "v78", "v79", "v80",   \
      "v81", "v82", "v83", "v84", "v85", "v86", "v87", "v88", "v89", "v90",   \
      "v91", "v92", "v93", "v94", "v95", "v100",                              \
      "a0", "a1", "a2", "a3", "memory"

// ---------------------------------------------------------------------------
// Latency (ATT) kernels: a 32-trip loop of { drain; MFMA; probe; s_nop 0 }.
// A loop rather than .rept keeps the code small, so only the first trip runs
// on a cold instruction cache. The s_nop 0 ends the probe group for
// analyze_att.py before the loop-control SALU ops.
// ---------------------------------------------------------------------------
#define LAT_KERNEL(name, body)                                                \
  __global__ void __launch_bounds__(64) lat_##name(int *out) {                \
    asm volatile(INIT S("s_mov_b32 s1, 32") S("1:") DRAIN body S("s_nop 0")   \
                 S("s_sub_u32 s1, s1, 1") S("s_cmp_lg_u32 s1, 0")             \
                 S("s_cbranch_scc1 1b")::: CLOBBERS, "s1", "scc");            \
    if (threadIdx.x == 0) out[blockIdx.x] = 1;                                \
  }

// Per-MFMA sweep. For MFMA X, _same issues the next identical MFMA (gap = C)
// and _q_{fma,pk,pk16,dot} issue a VALU, PK_F32, packed 16-bit op and DOT.
#define LAT_SAME(n, X, XB) LAT_KERNEL(n##_same, X XB)
#define LAT_QUAD(n, X)                                                        \
  LAT_KERNEL(n##_q_fma, X FMA_F32)                                            \
  LAT_KERNEL(n##_q_pk, X PK_FMA)                                              \
  LAT_KERNEL(n##_q_pk16, X PK_ADD_F16)                                        \
  LAT_KERNEL(n##_q_dot, X DOT2_F16)
#define LAT_ROW(n, X, XB) LAT_SAME(n, X, XB) LAT_QUAD(n, X)
#define LS(n) L(n##_same)
#define LQ(n) L(n##_q_fma), L(n##_q_pk), L(n##_q_pk16), L(n##_q_dot)
#define LR(n) LS(n), LQ(n)
#define LAT_DOTC(n, X)                                                        \
  LAT_KERNEL(n##_dot2c, X DOT2C_F16) LAT_KERNEL(n##_dot4c, X DOT4C_I8)
#define LDC(n) L(n##_dot2c), L(n##_dot4c)

LAT_ROW(xdl16, MFMA_16, MFMA_16_B)
LAT_QUAD(xdl16acc, MFMA_16_ACC)
LAT_QUAD(xdl4, MFMA_4X4)
LAT_QUAD(xdl32, MFMA_32)
LAT_QUAD(f16, MFMA_F16)
LAT_QUAD(i8, MFMA_I8)
LAT_QUAD(fp8, MFMA_FP8)
LAT_ROW(smleg, SM_LEG, SM_LEG_B)
LAT_QUAD(sgemm, MFMA_F32)
LAT_QUAD(dgemm16, MFMA_F64)
LAT_ROW(dgemm4, MFMA_F64_4X4, MFMA_F64_4X4_B)
#ifndef GFX950
LAT_QUAD(xf32, MFMA_XF32)
#endif

// Legacy XDL: a VALU between two MFMAs, other probe types, other PK opcodes.
LAT_KERNEL(mfma16_fma_mfma16, MFMA_16 FMA_F32 MFMA_16_B)
LAT_KERNEL(mfma16_vnop, MFMA_16 VNOP)
LAT_KERNEL(mfma16_movfar, MFMA_16 MOV_FAR)
LAT_KERNEL(mfma16_addu32, MFMA_16 ADD_U32)
LAT_KERNEL(mfma16_salu, MFMA_16 SALU)
LAT_KERNEL(mfma16_pkmul, MFMA_16 PK_MUL)
LAT_KERNEL(mfma16_pkadd, MFMA_16 PK_ADD)
LAT_KERNEL(mfma16_pkmov, MFMA_16 PK_MOV)
// Ladder: N independent VALU fillers between MFMA and PK.
LAT_KERNEL(mfma16_fill1_pkfma, MFMA_16 REP(1, FILL) PK_FMA)
LAT_KERNEL(mfma16_fill2_pkfma, MFMA_16 REP(2, FILL) PK_FMA)
LAT_KERNEL(mfma16_fill3_pkfma, MFMA_16 REP(3, FILL) PK_FMA)
LAT_KERNEL(mfma16_fill4_pkfma, MFMA_16 REP(4, FILL) PK_FMA)
LAT_KERNEL(mfma16_fill5_pkfma, MFMA_16 REP(5, FILL) PK_FMA)
LAT_KERNEL(mfma16_fill6_pkfma, MFMA_16 REP(6, FILL) PK_FMA)
// Does a pending PK hold back the next MFMA?
LAT_KERNEL(mfma16_pkfma_mfma16, MFMA_16 PK_FMA MFMA_16_B)
LAT_KERNEL(mfma16_dot_mfma16, MFMA_16 DOT2_F16 MFMA_16_B)
// Other packed 16-bit and never-coissue VALU after XDL.
LAT_KERNEL(mfma16_pkf16x3_mfma16, MFMA_16 PK16X3 MFMA_16_B)
LAT_KERNEL(mfma16_pkfmaf16, MFMA_16 PK_FMA_F16)
LAT_KERNEL(mfma16_pkmulf16, MFMA_16 PK_MUL_F16)
LAT_KERNEL(mfma16_pkaddu16, MFMA_16 PK_ADD_U16)
LAT_KERNEL(mfma16_pkmullo, MFMA_16 PK_MUL_LO)
LAT_KERNEL(mfma16_exp, MFMA_16 EXP_F32)
LAT_KERNEL(mfma16_cvtpkfp8, MFMA_16 CVT_PK_FP8)
LAT_KERNEL(mfma16_dot2c, MFMA_16 DOT2C_F16)
LAT_KERNEL(mfma16_dot2i16, MFMA_16 DOT2_I16)
LAT_KERNEL(mfma16_dot4i8, MFMA_16 DOT4_I8)
LAT_KERNEL(mfma16_dot4c, MFMA_16 DOT4C_I8)
LAT_DOTC(xdl32, MFMA_32)
LAT_DOTC(sgemm, MFMA_F32)

#ifdef GFX950
LAT_ROW(x2_16, X2_16, X2_16_B)
LAT_QUAD(x2_16acc, X2_16_ACC)
LAT_ROW(x2f16, X2_F16, X2_F16_B)
LAT_ROW(x2i8, X2_I8, X2_I8_B)
LAT_ROW(x2_32, X2_32, X2_32_B)
LAT_ROW(x232f16, X2_32F16, X2_32F16_B)
LAT_ROW(x232i8, X2_32I8, X2_32I8_B)
LAT_ROW(x2fp4, X2_FP4, X2_FP4_B)
LAT_ROW(x2fp6, X2_FP6, X2_FP6_B)
LAT_ROW(x2fp8, X2_FP8, X2_FP8_B)
LAT_ROW(x2f8f4, X2_F8F4, X2_F8F4_B)
LAT_ROW(x232fp4, X2_32FP4, X2_32FP4_B)
LAT_ROW(x232fp8, X2_32FP8, X2_32FP8_B)
LAT_ROW(x2scale, X2_SCALE, X2_SCALE_B)
LAT_ROW(x2scalefp6, X2_SCALE_FP6, X2_SCALE_FP6_B)
LAT_ROW(x2scalefp4, X2_SCALE_FP4, X2_SCALE_FP4_B)
LAT_SAME(x2scale32fp8, X2_SCALE32, X2_SCALE32_B)
LAT_SAME(x2scale32fp4, X2_SCALE32_FP4, X2_SCALE32_FP4_B)
LAT_ROW(sm16bf16, SM_16BF16, SM_16BF16_B)
LAT_ROW(sm32bf16, SM_32BF16, SM_32BF16_B)
LAT_ROW(sm16fp8, SM_16FP8, SM_16FP8_B)
LAT_ROW(sm32fp8, SM_32FP8, SM_32FP8_B)

// XDL2x: switching families, other probe types, other PK opcodes.
LAT_KERNEL(mfma16_x2_16, MFMA_16 X2_16_B)
LAT_KERNEL(x2_16_mfma16, X2_16 MFMA_16_B)
// Does a scaled MFMA's LD_SCALE slot overlap the previous MFMA?
LAT_KERNEL(x2fp8_x2scale, X2_FP8 X2_SCALE_B)
LAT_KERNEL(x2scale_x2fp8, X2_SCALE X2_FP8_B)
// A probe after an MFMA X that issues right behind MFMA P rather than into an
// idle pipe. The probe's delay is counted from X (pos 2 - pos 1): does a
// scaled X still add a cycle to V and P?
#define LAT_FOLLOW(n, P, X)                                                   \
  LAT_KERNEL(n##_fma, P X FMA_F32)                                            \
  LAT_KERNEL(n##_pk, P X PK_FMA)                                              \
  LAT_KERNEL(n##_dot, P X DOT2_F16)
#define LF(n) L(n##_fma), L(n##_pk), L(n##_dot)
LAT_FOLLOW(x2fp8_x2fp8, X2_FP8, X2_FP8_B)
LAT_FOLLOW(x2fp8_x2scale, X2_FP8, X2_SCALE_B)
LAT_FOLLOW(x2scale_x2scale, X2_SCALE, X2_SCALE_B)
LAT_FOLLOW(x2fp4_x2fp4, X2_FP4, X2_FP4_B)
LAT_FOLLOW(x2fp4_x2scalefp4, X2_FP4, X2_SCALE_FP4_B)
LAT_FOLLOW(x2scalefp4_x2scalefp4, X2_SCALE_FP4, X2_SCALE_FP4_B)
LAT_DOTC(x2fp8_x2scale, X2_FP8 X2_SCALE_B)
LAT_KERNEL(x2fp4_x2scalefp4_pk16, X2_FP4 X2_SCALE_FP4_B PK_ADD_F16)
LAT_KERNEL(x2fp8_x2scale_pk16, X2_FP8 X2_SCALE_B PK_ADD_F16)
LAT_KERNEL(x2scale_x2scale_pk16, X2_SCALE X2_SCALE_B PK_ADD_F16)
LAT_KERNEL(x2scalefp4_x2scalefp4_pk16, X2_SCALE_FP4 X2_SCALE_FP4_B PK_ADD_F16)
LAT_DOTC(x2scale_x2scale, X2_SCALE X2_SCALE_B)
LAT_DOTC(x2fp4_x2scalefp4, X2_FP4 X2_SCALE_FP4_B)
LAT_DOTC(x2scalefp4_x2scalefp4, X2_SCALE_FP4 X2_SCALE_FP4_B)
// Ladder: N VALU fillers between an MFMA and a scaled MFMA. Where does the
// pipe go idle enough for the scaled MFMA to pay its extra cycle again?
#define LAT_SCALE_LADDER(n, P, N)                                             \
  LAT_KERNEL(n##_fill##N##_x2scale_pk, P REP(N, FILL) X2_SCALE_B PK_FMA)
#define LSL(n, N) L(n##_fill##N##_x2scale_pk)
LAT_SCALE_LADDER(x2fp8, X2_FP8, 1)
LAT_SCALE_LADDER(x2fp8, X2_FP8, 2)
LAT_SCALE_LADDER(x2fp8, X2_FP8, 3)
LAT_SCALE_LADDER(x2fp8, X2_FP8, 4)
LAT_SCALE_LADDER(x2fp8, X2_FP8, 5)
LAT_SCALE_LADDER(x2fp8, X2_FP8, 6)
LAT_SCALE_LADDER(x2fp8, X2_FP8, 7)
LAT_SCALE_LADDER(x2fp4, X2_FP4, 1)
LAT_SCALE_LADDER(x2fp4, X2_FP4, 2)
LAT_SCALE_LADDER(x2fp4, X2_FP4, 3)
LAT_SCALE_LADDER(x2fp4, X2_FP4, 4)
// Previous MFMA from another family or pipe.
LAT_KERNEL(mfma16_x2scale_pk, MFMA_16 X2_SCALE_B PK_FMA)
LAT_KERNEL(x2_16_x2scale_pk, X2_16 X2_SCALE_B PK_FMA)
LAT_KERNEL(sgemm_x2scale_pk, MFMA_F32 X2_SCALE_B PK_FMA)
LAT_KERNEL(dgemm16_x2scale_pk, MFMA_F64 X2_SCALE_B PK_FMA)
LAT_KERNEL(x2_16_vnop, X2_16 VNOP)
LAT_KERNEL(x2_16_salu, X2_16 SALU)
LAT_KERNEL(x2_16_fma_x2_16, X2_16 FMA_F32 X2_16_B)
LAT_KERNEL(x2_16_pkmul, X2_16 PK_MUL)
LAT_KERNEL(x2_16_pkadd, X2_16 PK_ADD)
LAT_KERNEL(x2_16_pkmov, X2_16 PK_MOV)
// Ladder after XDL2x.
LAT_KERNEL(x2_16_fill1_pkfma, X2_16 REP(1, FILL) PK_FMA)
LAT_KERNEL(x2_16_fill2_pkfma, X2_16 REP(2, FILL) PK_FMA)
LAT_KERNEL(x2_16_fill3_pkfma, X2_16 REP(3, FILL) PK_FMA)
LAT_KERNEL(x2_16_fill4_pkfma, X2_16 REP(4, FILL) PK_FMA)
LAT_KERNEL(x2_16_pkfma_x2_16, X2_16 PK_FMA X2_16_B)
LAT_KERNEL(x2_16_dot_x2_16, X2_16 DOT2_F16 X2_16_B)
// DOTnC after XDL2x and scaled MFMAs.
LAT_DOTC(x2_16, X2_16)
LAT_DOTC(x2fp8, X2_FP8)
LAT_DOTC(x2scale, X2_SCALE)
LAT_KERNEL(mfma16_dot2cbf16, MFMA_16 DOT2C_BF16)
LAT_KERNEL(x2_16_dot2cbf16, X2_16 DOT2C_BF16)
#endif

// s_memtime into %[out] unless %[timed] is 0. A taken branch keeps the ATT
// decoder in sync; n is a unique numeric label.
#define TSTAMP(n, out)                                                        \
  S("s_cmp_eq_u32 %[timed], 0") S("s_cbranch_scc1 " #n "f")                   \
  S("s_memtime %[" #out "]") S("s_waitcnt lgkmcnt(0)") S(#n ":")

// ---------------------------------------------------------------------------
// Throughput kernels: 256 iterations of a pattern, timed with s_memtime.
// Four rotating accumulators keep MFMAs independent.
// ---------------------------------------------------------------------------
#define TPUT_KERNEL(name, body)                                               \
  __global__ void __launch_bounds__(64)                                      \
      tput_##name(unsigned long long *out, int timed) {                       \
    unsigned long long t0, t1;                                                \
    asm volatile(INIT TSTAMP(1, t0) REP(64, body) TSTAMP(2, t1)               \
                 : [t0] "=&s"(t0), [t1] "=&s"(t1)                             \
                 : [timed] "s"(timed)                                         \
                 : CLOBBERS, "scc");                                          \
    if (threadIdx.x == 0) out[blockIdx.x] = t1 - t0;                          \
  }
#define X4(a, b, c, d) a b c d

// SIPreEmitPeephole output before/after the unpack-window fix: U(lo, hi) is
// an unpacked v_pk_fma_f32, PKF(lo, hi) a packed one. Like the lit test they
// mirror, each instruction reads a single source register three times.
#define U(lo, hi) S("v_fma_f32 v" #lo ", v4, v4, v4") S("v_fma_f32 v" #hi ", v5, v5, v5")
#define PKF(lo, hi) S("v_pk_fma_f32 v[" #lo ":" #hi "], v[4:5], v[4:5], v[4:5]")
#define SEQ4_BEFORE U(12, 13) PKF(14, 15) PKF(16, 17) PKF(18, 19)
#define SEQ4_AFTER U(12, 13) U(14, 15) PKF(16, 17) PKF(18, 19)
#define SEQ4_PACKED PKF(12, 13) PKF(14, 15) PKF(16, 17) PKF(18, 19)
#define SEQ4_UNPACKED U(12, 13) U(14, 15) U(16, 17) U(18, 19)
#define MFMA_F64_B S("v_mfma_f64_16x16x4_f64 v[40:47], v[0:1], v[2:3], v[40:47]")
#define MFMA_F64_C S("v_mfma_f64_16x16x4_f64 v[80:87], v[0:1], v[2:3], v[80:87]")
#define MFMA_F64_D S("v_mfma_f64_16x16x4_f64 v[88:95], v[0:1], v[2:3], v[88:95]")
#define STREAM4(M1, M2, M3, M4, SEQ) X4(M1 SEQ, M2 SEQ, M3 SEQ, M4 SEQ)
TPUT_KERNEL(win_xdl_before, STREAM4(MFMA_16, MFMA_16_B, MFMA_16_C, MFMA_16_D, SEQ4_BEFORE))
TPUT_KERNEL(win_xdl_after, STREAM4(MFMA_16, MFMA_16_B, MFMA_16_C, MFMA_16_D, SEQ4_AFTER))
TPUT_KERNEL(win_xdl_unpacked, STREAM4(MFMA_16, MFMA_16_B, MFMA_16_C, MFMA_16_D, SEQ4_UNPACKED))
// The PK_F32s replaced with packed 16-bit ops and with DOTs, which the pass
// never unpacks.
#define SEQ4_PK16 S("v_pk_add_f16 v12, v4, v5") S("v_pk_add_f16 v13, v4, v5") \
                  S("v_pk_add_f16 v14, v4, v5") S("v_pk_add_f16 v15, v4, v5")
#define SEQ4_DOT S("v_dot2_f32_f16 v12, v4, v5, v6") S("v_dot2_f32_f16 v13, v4, v5, v6") \
                 S("v_dot2_f32_f16 v14, v4, v5, v6") S("v_dot2_f32_f16 v15, v4, v5, v6")
TPUT_KERNEL(win_xdl_pk16, STREAM4(MFMA_16, MFMA_16_B, MFMA_16_C, MFMA_16_D, SEQ4_PK16))
TPUT_KERNEL(win_xdl_dot, STREAM4(MFMA_16, MFMA_16_B, MFMA_16_C, MFMA_16_D, SEQ4_DOT))
// Before the fix the pass unpacked 3 PKs after a DGEMM on gfx942 (8 passes)
// and 4 on gfx950 (16 passes).
#ifdef GFX950
TPUT_KERNEL(win_dgemm_before, STREAM4(MFMA_F64, MFMA_F64_B, MFMA_F64_C, MFMA_F64_D, SEQ4_UNPACKED))
#else
TPUT_KERNEL(win_dgemm_before, STREAM4(MFMA_F64, MFMA_F64_B, MFMA_F64_C, MFMA_F64_D, U(12, 13) U(14, 15) U(16, 17) PKF(18, 19)))
#endif
TPUT_KERNEL(win_dgemm_after, STREAM4(MFMA_F64, MFMA_F64_B, MFMA_F64_C, MFMA_F64_D, SEQ4_PACKED))
// Before the fix the pass unpacked 3 PKs after an SGEMM (8 passes) on both.
#define MFMA_F32_B S("v_mfma_f32_16x16x4_f32 v[24:27], v0, v1, v[24:27]")
#define MFMA_F32_C S("v_mfma_f32_16x16x4_f32 v[28:31], v0, v1, v[28:31]")
#define MFMA_F32_D S("v_mfma_f32_16x16x4_f32 v[48:51], v0, v1, v[48:51]")
TPUT_KERNEL(win_sgemm_before, STREAM4(MFMA_F32, MFMA_F32_B, MFMA_F32_C, MFMA_F32_D, U(12, 13) U(14, 15) U(16, 17) PKF(18, 19)))
TPUT_KERNEL(win_sgemm_after, STREAM4(MFMA_F32, MFMA_F32_B, MFMA_F32_C, MFMA_F32_D, SEQ4_PACKED))
// One PK more than the fixed pass unpacks: _before/_after are the pass output
// before/after the fix, _more also unpacks the last PK.
#define SEQ3_BEFORE U(12, 13) PKF(14, 15) PKF(16, 17)
#define SEQ3_AFTER U(12, 13) U(14, 15) PKF(16, 17)
#define SEQ3_MORE U(12, 13) U(14, 15) U(16, 17)
#define SEQ5_BEFORE U(12, 13) U(14, 15) U(16, 17) PKF(18, 19) PKF(20, 21)
#define SEQ5_AFTER U(12, 13) U(14, 15) U(16, 17) U(18, 19) PKF(20, 21)
#define SEQ5_MORE U(12, 13) U(14, 15) U(16, 17) U(18, 19) U(20, 21)
#define MFMA_32_B S("v_mfma_f32_32x32x8_bf16 v[80:95], v[0:1], v[2:3], v[80:95]")
TPUT_KERNEL(win3_xdl_before, STREAM4(MFMA_16, MFMA_16_B, MFMA_16_C, MFMA_16_D, SEQ3_BEFORE))
TPUT_KERNEL(win3_xdl_after, STREAM4(MFMA_16, MFMA_16_B, MFMA_16_C, MFMA_16_D, SEQ3_AFTER))
TPUT_KERNEL(win3_xdl_more, STREAM4(MFMA_16, MFMA_16_B, MFMA_16_C, MFMA_16_D, SEQ3_MORE))
TPUT_KERNEL(win5_xdl32_before, STREAM4(MFMA_32, MFMA_32_B, MFMA_32, MFMA_32_B, SEQ5_BEFORE))
TPUT_KERNEL(win5_xdl32_after, STREAM4(MFMA_32, MFMA_32_B, MFMA_32, MFMA_32_B, SEQ5_AFTER))
TPUT_KERNEL(win5_xdl32_more, STREAM4(MFMA_32, MFMA_32_B, MFMA_32, MFMA_32_B, SEQ5_MORE))

TPUT_KERNEL(mfma16, X4(MFMA_16, MFMA_16_B, MFMA_16_C, MFMA_16_D))
TPUT_KERNEL(fma, X4(FMA_F32, FMA_F32, FMA_F32, FMA_F32))
TPUT_KERNEL(pkfma, X4(PK_FMA, PK_FMA, PK_FMA, PK_FMA))
TPUT_KERNEL(mfma16_fma, X4(MFMA_16 FMA_F32, MFMA_16_B FMA_F32,
                           MFMA_16_C FMA_F32, MFMA_16_D FMA_F32))
TPUT_KERNEL(mfma16_fma2, X4(MFMA_16 FMA_F32 FMA_F32_B, MFMA_16_B FMA_F32 FMA_F32_B,
                            MFMA_16_C FMA_F32 FMA_F32_B, MFMA_16_D FMA_F32 FMA_F32_B))
TPUT_KERNEL(mfma16_pkfma, X4(MFMA_16 PK_FMA, MFMA_16_B PK_FMA,
                             MFMA_16_C PK_FMA, MFMA_16_D PK_FMA))
TPUT_KERNEL(mfma16_pkmov, X4(MFMA_16 PK_MOV, MFMA_16_B PK_MOV,
                             MFMA_16_C PK_MOV, MFMA_16_D PK_MOV))
TPUT_KERNEL(pkf16x3, X4(PK16X3, PK16X3, PK16X3, PK16X3))
TPUT_KERNEL(mfma16_pkf16, X4(MFMA_16 PK_ADD_F16, MFMA_16_B PK_ADD_F16,
                             MFMA_16_C PK_ADD_F16, MFMA_16_D PK_ADD_F16))
TPUT_KERNEL(mfma16_pkf16x3, X4(MFMA_16 PK16X3, MFMA_16_B PK16X3,
                               MFMA_16_C PK16X3, MFMA_16_D PK16X3))
#ifdef GFX950
TPUT_KERNEL(x2_16, X4(X2_16, X2_16_B, X2_16_C, X2_16_D))
TPUT_KERNEL(x2_16_fma, X4(X2_16 FMA_F32, X2_16_B FMA_F32,
                          X2_16_C FMA_F32, X2_16_D FMA_F32))
TPUT_KERNEL(x2_16_pkfma, X4(X2_16 PK_FMA, X2_16_B PK_FMA,
                            X2_16_C PK_FMA, X2_16_D PK_FMA))
// Alternate families every MFMA: legacy, XDL2x, legacy, XDL2x.
TPUT_KERNEL(alt_mfma16_x2_16, X4(MFMA_16, X2_16_B, MFMA_16_C, X2_16_D))
#define X2_FP8_C S("v_mfma_f32_16x16x128_f8f6f4 v[28:31], v[64:71], v[72:79], v[28:31]")
#define X2_FP8_D S("v_mfma_f32_16x16x128_f8f6f4 v[48:51], v[64:71], v[72:79], v[48:51]")
#define SEQ6_BEFORE U(12, 13) U(14, 15) U(16, 17) PKF(18, 19) PKF(20, 21) PKF(22, 23)
#define SEQ6_AFTER U(12, 13) U(14, 15) U(16, 17) U(18, 19) PKF(20, 21) PKF(22, 23)
TPUT_KERNEL(win_x2_before, STREAM4(X2_16, X2_16_B, X2_16_C, X2_16_D, SEQ4_BEFORE))
TPUT_KERNEL(win_x2_after, STREAM4(X2_16, X2_16_B, X2_16_C, X2_16_D, SEQ4_AFTER))
TPUT_KERNEL(win_f8_before, STREAM4(X2_FP8, X2_FP8_B, X2_FP8_C, X2_FP8_D, SEQ6_BEFORE))
TPUT_KERNEL(win_f8_after, STREAM4(X2_FP8, X2_FP8_B, X2_FP8_C, X2_FP8_D, SEQ6_AFTER))
TPUT_KERNEL(win3_x2_before, STREAM4(X2_16, X2_16_B, X2_16_C, X2_16_D, SEQ3_BEFORE))
TPUT_KERNEL(win3_x2_after, STREAM4(X2_16, X2_16_B, X2_16_C, X2_16_D, SEQ3_AFTER))
TPUT_KERNEL(win3_x2_more, STREAM4(X2_16, X2_16_B, X2_16_C, X2_16_D, SEQ3_MORE))
TPUT_KERNEL(win5_x2_32_before, STREAM4(X2_32, X2_32_B, X2_32, X2_32_B, SEQ5_BEFORE))
TPUT_KERNEL(win5_x2_32_after, STREAM4(X2_32, X2_32_B, X2_32, X2_32_B, SEQ5_AFTER))
TPUT_KERNEL(win5_x2_32_more, STREAM4(X2_32, X2_32_B, X2_32, X2_32_B, SEQ5_MORE))
TPUT_KERNEL(win5_f8_before, STREAM4(X2_FP8, X2_FP8_B, X2_FP8_C, X2_FP8_D, SEQ5_BEFORE))
TPUT_KERNEL(win5_f8_after, STREAM4(X2_FP8, X2_FP8_B, X2_FP8_C, X2_FP8_D, SEQ5_AFTER))
TPUT_KERNEL(win5_f8_more, STREAM4(X2_FP8, X2_FP8_B, X2_FP8_C, X2_FP8_D, SEQ5_MORE))
// F8F6F4 streams, unscaled and scaled: does the LD_SCALE slot cost throughput?
#define X2_FP6_C S("v_mfma_f32_16x16x128_f8f6f4 v[28:31], v[64:69], v[72:77], v[28:31] cbsz:2 blgp:2")
#define X2_FP6_D S("v_mfma_f32_16x16x128_f8f6f4 v[48:51], v[64:69], v[72:77], v[48:51] cbsz:2 blgp:2")
#define X2_FP4_C S("v_mfma_f32_16x16x128_f8f6f4 v[28:31], v[64:67], v[72:75], v[28:31] cbsz:4 blgp:4")
#define X2_FP4_D S("v_mfma_f32_16x16x128_f8f6f4 v[48:51], v[64:67], v[72:75], v[48:51] cbsz:4 blgp:4")
TPUT_KERNEL(x2fp8, X4(X2_FP8, X2_FP8_B, X2_FP8_C, X2_FP8_D))
TPUT_KERNEL(x2scale, X4(X2_SCALE, X2_SCALE_B, X2_SCALE_C, X2_SCALE_D))
TPUT_KERNEL(x2fp6, X4(X2_FP6, X2_FP6_B, X2_FP6_C, X2_FP6_D))
TPUT_KERNEL(x2scalefp6, X4(X2_SCALE_FP6, X2_SCALE_FP6_B, X2_SCALE_FP6_C, X2_SCALE_FP6_D))
TPUT_KERNEL(x2fp4, X4(X2_FP4, X2_FP4_B, X2_FP4_C, X2_FP4_D))
TPUT_KERNEL(x2scalefp4, X4(X2_SCALE_FP4, X2_SCALE_FP4_B, X2_SCALE_FP4_C, X2_SCALE_FP4_D))
TPUT_KERNEL(alt_x2fp8_x2scale, X4(X2_FP8, X2_SCALE_B, X2_FP8_C, X2_SCALE_D))
#endif

// ---------------------------------------------------------------------------
// Cross-wave kernels: 8 waves per workgroup and one workgroup per CU, so each
// SIMD hosts wave w (MFMA stream) and wave w+4 (probe stream). Each wave
// records s_memtime at start/end and its SIMD id. The barrier makes both
// streams start together; without it the probe wave's INIT can be stuck
// behind the MFMA stream.
// ---------------------------------------------------------------------------
#define XWAVE_BODY(body)                                                      \
  asm volatile(INIT S("s_barrier") TSTAMP(1, t0) body TSTAMP(2, t1)           \
               S("s_getreg_b32 %[hw], hwreg(HW_REG_HW_ID)")                   \
               : [t0] "=&s"(t0), [t1] "=&s"(t1), [hw] "=&s"(hw)              \
               : [timed] "s"(timed)                                           \
               : CLOBBERS, "scc")

#define XWAVE_KERNEL(name, mfma_body, probe_body)                             \
  __global__ void __launch_bounds__(512)                                     \
      x_##name(unsigned long long *out, int timed) {                          \
    unsigned long long t0, t1;                                                \
    unsigned hw;                                                              \
    int w = __builtin_amdgcn_readfirstlane(threadIdx.x / 64);                 \
    if (w < 4) XWAVE_BODY(mfma_body); else XWAVE_BODY(probe_body);           \
    if (threadIdx.x % 64 == 0) {                                              \
      unsigned long long *o = out + (blockIdx.x * 8 + w) * 3;                 \
      o[0] = t0; o[1] = t1; o[2] = (hw >> 4) & 3;                             \
    }                                                                         \
  }

static const int kXMfma = 256, kXProbe = 512;
#define MFMA_STREAM REP(64, X4(MFMA_16, MFMA_16_B, MFMA_16_C, MFMA_16_D))
#define X2_STREAM REP(64, X4(X2_16, X2_16_B, X2_16_C, X2_16_D))
#define PROBES(p) REP(512, p)

XWAVE_KERNEL(mfma_none, MFMA_STREAM, "")       // baselines
XWAVE_KERNEL(none_fma, "", PROBES(FMA_F32))
XWAVE_KERNEL(none_pkfma, "", PROBES(PK_FMA))
XWAVE_KERNEL(mfma_fma, MFMA_STREAM, PROBES(FMA_F32))
XWAVE_KERNEL(mfma_pkfma, MFMA_STREAM, PROBES(PK_FMA))
XWAVE_KERNEL(mfma_pkmov, MFMA_STREAM, PROBES(PK_MOV))
XWAVE_KERNEL(none_pkaddf16, "", PROBES(PK_ADD_F16))
XWAVE_KERNEL(mfma_pkaddf16, MFMA_STREAM, PROBES(PK_ADD_F16))
XWAVE_KERNEL(none_dot, "", PROBES(DOT2_F16))
XWAVE_KERNEL(mfma_dot, MFMA_STREAM, PROBES(DOT2_F16))
#ifdef GFX950
XWAVE_KERNEL(x2_fma, X2_STREAM, PROBES(FMA_F32))
XWAVE_KERNEL(x2_pkfma, X2_STREAM, PROBES(PK_FMA))
XWAVE_KERNEL(x2_dot, X2_STREAM, PROBES(DOT2_F16))
// Both waves stream F8F6F4 MFMAs: can one wave's LD_SCALE issue under the
// other wave's MFMA? Here the "probe" wave is the second MFMA stream.
#define FP8_STREAM REP(64, X4(X2_FP8, X2_FP8_B, X2_FP8_C, X2_FP8_D))
#define SCALE_STREAM REP(64, X4(X2_SCALE, X2_SCALE_B, X2_SCALE_C, X2_SCALE_D))
#define FP4_STREAM REP(64, X4(X2_FP4, X2_FP4_B, X2_FP4_C, X2_FP4_D))
#define SCALE_FP4_STREAM                                                      \
  REP(64, X4(X2_SCALE_FP4, X2_SCALE_FP4_B, X2_SCALE_FP4_C, X2_SCALE_FP4_D))
XWAVE_KERNEL(x2fp8_x2fp8, FP8_STREAM, FP8_STREAM)
XWAVE_KERNEL(x2scale_x2scale, SCALE_STREAM, SCALE_STREAM)
XWAVE_KERNEL(x2fp4_x2fp4, FP4_STREAM, FP4_STREAM)
XWAVE_KERNEL(x2scalefp4_x2scalefp4, SCALE_FP4_STREAM, SCALE_FP4_STREAM)
#endif

// ---------------------------------------------------------------------------
// Host.
// ---------------------------------------------------------------------------
struct LatK { const char *name; void (*fn)(int *); };
struct TputK { const char *name; void (*fn)(unsigned long long *, int); };

#define L(n) {"lat_" #n, lat_##n}
static const LatK kLat[] = {
    LR(xdl16),    LQ(xdl16acc), LQ(xdl4),     LQ(xdl32),    LQ(f16),
    LQ(i8),       LQ(fp8),      LR(smleg),    LQ(sgemm),    LQ(dgemm16),
    LR(dgemm4),
#ifndef GFX950
    LQ(xf32),
#endif
    L(mfma16_fma_mfma16),  L(mfma16_vnop),        L(mfma16_movfar),
    L(mfma16_addu32),      L(mfma16_salu),        L(mfma16_pkmul),
    L(mfma16_pkadd),       L(mfma16_pkmov),
    L(mfma16_fill1_pkfma), L(mfma16_fill2_pkfma), L(mfma16_fill3_pkfma),
    L(mfma16_fill4_pkfma), L(mfma16_fill5_pkfma), L(mfma16_fill6_pkfma),
    L(mfma16_pkfma_mfma16), L(mfma16_dot_mfma16),
    L(mfma16_pkf16x3_mfma16), L(mfma16_pkfmaf16), L(mfma16_pkmulf16),
    L(mfma16_pkaddu16),    L(mfma16_pkmullo),     L(mfma16_exp),
    L(mfma16_cvtpkfp8),    L(mfma16_dot2c),       L(mfma16_dot2i16),
    L(mfma16_dot4i8),      L(mfma16_dot4c),       LDC(xdl32),
    LDC(sgemm),
#ifdef GFX950
    LR(x2_16),    LQ(x2_16acc), LR(x2f16),    LR(x2i8),     LR(x2_32),
    LR(x232f16),  LR(x232i8),   LR(x2fp4),    LR(x2fp6),    LR(x2fp8),
    LR(x2f8f4),   LR(x232fp4),  LR(x232fp8),  LR(x2scale),  LR(x2scalefp6),
    LR(x2scalefp4), LS(x2scale32fp8), LS(x2scale32fp4),     LR(sm16bf16),
    LR(sm32bf16), LR(sm16fp8),  LR(sm32fp8),
    L(mfma16_x2_16),       L(x2_16_mfma16),       L(x2fp8_x2scale),
    L(x2scale_x2fp8),
    LF(x2fp8_x2fp8),       LF(x2fp8_x2scale),     LF(x2scale_x2scale),
    LF(x2fp4_x2fp4),       LF(x2fp4_x2scalefp4),  LF(x2scalefp4_x2scalefp4),
    LDC(x2fp8_x2scale),    L(x2fp4_x2scalefp4_pk16), L(x2fp8_x2scale_pk16),
    L(x2scale_x2scale_pk16), L(x2scalefp4_x2scalefp4_pk16),
    LDC(x2scale_x2scale),  LDC(x2fp4_x2scalefp4), LDC(x2scalefp4_x2scalefp4),
    LSL(x2fp8, 1), LSL(x2fp8, 2), LSL(x2fp8, 3), LSL(x2fp8, 4), LSL(x2fp8, 5),
    LSL(x2fp8, 6), LSL(x2fp8, 7),
    LSL(x2fp4, 1), LSL(x2fp4, 2), LSL(x2fp4, 3), LSL(x2fp4, 4),
    L(mfma16_x2scale_pk),  L(x2_16_x2scale_pk),   L(sgemm_x2scale_pk),
    L(dgemm16_x2scale_pk),
    L(x2_16_vnop),
    L(x2_16_salu),         L(x2_16_fma_x2_16),    L(x2_16_pkmul),
    L(x2_16_pkadd),        L(x2_16_pkmov),
    L(x2_16_fill1_pkfma),  L(x2_16_fill2_pkfma),  L(x2_16_fill3_pkfma),
    L(x2_16_fill4_pkfma),  L(x2_16_pkfma_x2_16),  L(x2_16_dot_x2_16),
    LDC(x2_16),   LDC(x2fp8),   LDC(x2scale),
    L(mfma16_dot2cbf16),   L(x2_16_dot2cbf16),
#endif
};
#define T(n) {"tput_" #n, tput_##n}
static const TputK kTput[] = {
    T(mfma16), T(fma), T(pkfma), T(mfma16_fma), T(mfma16_fma2), T(mfma16_pkfma),
    T(mfma16_pkmov), T(pkf16x3), T(mfma16_pkf16), T(mfma16_pkf16x3),
    T(win_xdl_before), T(win_xdl_after), T(win_xdl_unpacked),
    T(win_xdl_pk16), T(win_xdl_dot),
    T(win_dgemm_before), T(win_dgemm_after),
    T(win_sgemm_before), T(win_sgemm_after),
    T(win3_xdl_before), T(win3_xdl_after), T(win3_xdl_more),
    T(win5_xdl32_before), T(win5_xdl32_after), T(win5_xdl32_more),
#ifdef GFX950
    T(x2_16), T(x2_16_fma), T(x2_16_pkfma), T(alt_mfma16_x2_16),
    T(win_x2_before), T(win_x2_after), T(win_f8_before), T(win_f8_after),
    T(win3_x2_before), T(win3_x2_after), T(win3_x2_more),
    T(win5_x2_32_before), T(win5_x2_32_after), T(win5_x2_32_more),
    T(win5_f8_before), T(win5_f8_after), T(win5_f8_more),
    T(x2fp8), T(x2scale), T(x2fp6), T(x2scalefp6), T(x2fp4), T(x2scalefp4),
    T(alt_x2fp8_x2scale),
#endif
};
static const int kIters = 64 * 4;  // REP(64, X4(...))
#define XK(n, nm, np) {"x_" #n, x_##n, nm, np}
struct XK { const char *name; void (*fn)(unsigned long long *, int); int nm, np; };
static const XK kX[] = {
    XK(mfma_none, kXMfma, 0),       XK(none_fma, 0, kXProbe),
    XK(none_pkfma, 0, kXProbe),     XK(mfma_fma, kXMfma, kXProbe),
    XK(mfma_pkfma, kXMfma, kXProbe), XK(mfma_pkmov, kXMfma, kXProbe),
    XK(none_pkaddf16, 0, kXProbe),  XK(mfma_pkaddf16, kXMfma, kXProbe),
    XK(none_dot, 0, kXProbe),       XK(mfma_dot, kXMfma, kXProbe),
#ifdef GFX950
    XK(x2_fma, kXMfma, kXProbe),    XK(x2_pkfma, kXMfma, kXProbe),
    XK(x2_dot, kXMfma, kXProbe),
    XK(x2fp8_x2fp8, kXMfma, kXMfma), XK(x2scale_x2scale, kXMfma, kXMfma),
    XK(x2fp4_x2fp4, kXMfma, kXMfma), XK(x2scalefp4_x2scalefp4, kXMfma, kXMfma),
#endif
};

// "" runs everything, "lat"/"tput"/"x" runs one group, anything else is an
// exact kernel name (so a profiler sees exactly one dispatch).
static bool selected(const char *name, const char *filter) {
  if (!*filter || !strcmp(name, filter)) return true;
  size_t n = strlen(filter);
  return (!strcmp(filter, "lat") || !strcmp(filter, "tput") ||
          !strcmp(filter, "x")) &&
         !strncmp(name, filter, n) && name[n] == '_';
}

// Requests the whole CU's LDS so at most one workgroup lives on a CU.
template <typename... Args>
static void launch(void (*fn)(Args...), int grid, int block, size_t lds,
                   Args... args) {
  CHECK(hipFuncSetAttribute(reinterpret_cast<const void *>(fn),
                            hipFuncAttributeMaxDynamicSharedMemorySize, lds));
  hipLaunchKernelGGL(fn, dim3(grid), dim3(block), lds, 0, args...);
  CHECK(hipGetLastError());
  CHECK(hipDeviceSynchronize());
}

int main(int argc, char **argv) {
  const char *filter = "";
  int timed = 1;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--att")) timed = 0;
    else filter = argv[i];
  }
  const int launches = timed ? 2 : 1;  // first timed launch is warm-up
  hipDeviceProp_t prop;
  CHECK(hipGetDeviceProperties(&prop, 0));
  const size_t lds = prop.maxSharedMemoryPerMultiProcessor;
  const int grid = prop.multiProcessorCount * 4;
  printf("# %s (%s), %d CUs, grid=%d, lds=%zu\n", prop.name, prop.gcnArchName,
         prop.multiProcessorCount, grid, lds);

  int *dout;
  unsigned long long *tout;
  CHECK(hipMalloc(&dout, grid * sizeof(int)));
  CHECK(hipMalloc(&tout, grid * sizeof(unsigned long long)));
  std::vector<unsigned long long> h(grid);

  for (const auto &k : kLat) {
    if (!selected(k.name, filter)) continue;
    launch(k.fn, grid, 64, lds, dout);
    printf("ran %s\n", k.name);
  }

  bool any = false;
  for (const auto &k : kTput) any |= selected(k.name, filter);
  if (any && timed)
    printf("\n%-22s %12s %12s\n", "throughput kernel", "cyc/iter", "min cyc/iter");
  for (const auto &k : kTput) {
    if (!selected(k.name, filter)) continue;
    for (int i = 0; i < launches; ++i) launch(k.fn, grid, 64, lds, tout, timed);
    if (!timed) continue;
    CHECK(hipMemcpy(h.data(), tout, grid * sizeof(h[0]), hipMemcpyDeviceToHost));
    double sum = 0;
    unsigned long long mn = ~0ull;
    for (auto v : h) { sum += v; mn = v < mn ? v : mn; }
    printf("%-22s %12.2f %12.2f\n", k.name, sum / grid / kIters,
           double(mn) / kIters);
  }

  // Cross-wave: per SIMD, the MFMA wave and the probe wave run concurrently.
  const int xgrid = prop.multiProcessorCount * 2;
  unsigned long long *xout;
  CHECK(hipMalloc(&xout, xgrid * 8 * 3 * sizeof(unsigned long long)));
  std::vector<unsigned long long> x(xgrid * 8 * 3);
  any = false;
  for (const auto &k : kX) any |= selected(k.name, filter);
  if (any && timed)
    printf("\n%-16s %14s %15s %22s %9s\n", "cross-wave kernel",
           "MFMA cyc/inst", "probe cyc/inst", "probe end - MFMA end", "bad pair");
  for (const auto &k : kX) {
    if (!selected(k.name, filter)) continue;
    for (int i = 0; i < launches; ++i) launch(k.fn, xgrid, 512, lds, xout, timed);
    if (!timed) continue;
    CHECK(hipMemcpy(x.data(), xout, x.size() * sizeof(x[0]), hipMemcpyDeviceToHost));
    double m = 0, p = 0, d = 0;
    int bad = 0, n = 0;
    for (int b = 0; b < xgrid; ++b)
      for (int w = 0; w < 4; ++w, ++n) {
        const unsigned long long *mw = &x[(b * 8 + w) * 3];
        const unsigned long long *pw = &x[(b * 8 + w + 4) * 3];
        bad += mw[2] != pw[2];
        m += k.nm ? double(mw[1] - mw[0]) / k.nm : 0;
        p += k.np ? double(pw[1] - pw[0]) / k.np : 0;
        d += double(pw[1]) - double(mw[1]);
      }
    printf("%-16s %14.2f %15.2f %22.0f %9d\n", k.name, m / n, p / n, d / n, bad);
  }
  CHECK(hipFree(xout));
  CHECK(hipFree(dout));
  CHECK(hipFree(tout));
  return 0;
}
