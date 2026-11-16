# A Unified Dual-Mode FP4 MAC Accelerator for LLM Inference on RISC-V Processors

> A unified MAC datapath supporting both MXFP4 (OCP Microscaling) and NVFP4 (NVIDIA FP4) in a single datapath, with RISC-V co-processor integration.

---

## Overview

Large Language Models demand extreme weight compression for edge deployment. Two competing 4-bit floating-point standards — **MXFP4** and **NVFP4** — each offer 7–8× compression over FP32, but use different block-scaling mechanisms. Neither dominates: they are the two endpoints of a narrow accuracy–footprint frontier, and a device does not choose the format of the checkpoint it is handed.

This project presents a **unified dual-mode MAC unit** that handles both formats in a single datapath by exploiting their shared E2M1 weight representation and FP32 arithmetic back-end. A one-bit `FORMAT_MODE` signal selects the active scale path at runtime, with no reconfiguration, pipeline flush, or state reset.

### Key results (45 nm GPDK045, Cadence Genus, 400 MHz)

| Result | Measurement |
|---|---|
| Cost of adding MXFP4 support to an NVFP4-capable MAC | **0.88%** of cell area (41 cells) — below the flow's own 0.90% run-to-run variation |
| Unified vs. standalone NVFP4-only | +1.78% cell area |
| E8M0 scale decoder | 3 standard cells (identical across three independent syntheses) |
| Scale-multiplier reduction, MXFP4 vs NVFP4 mode | **14.9×** |
| Area asymmetry, dedicated MXFP4 vs NVFP4 MAC | 2.6× |
| Per-mode power on identical silicon (SAIF-annotated) | NVFP4 costs **9.6% more** (15.845 mW vs 14.459 mW) |
| Speedup over soft-float FP4 on the same PicoRV32 | **40.2×** (76.0 vs 3,055.6 cycles/element-pair) |
| Kernel code size | 3,540 → **162 bytes** (21.9×) |
| Adapter agreement, PCPI vs APU | 0.16% |

**The mechanism.** MXFP4's E8M0 block scale is a power of two, so its FP32 mantissa is identically zero, and the 16-entry E2M1 LUT emits only two mantissa patterns. Synthesis therefore eliminates nearly the entire 24×24 partial-product array in the scale multipliers. This single effect — *mantissa collapse* — explains the area asymmetry, the per-mode power difference, and the per-instance toggle counts (7.03 M vs 8.20 M, +16.6%).

---

## Accuracy

WikiText-2 perplexity on TinyLlama-1.1B, all 2-D weight tensors quantized:

| Configuration | Perplexity | Compression |
|---|---|---|
| FP32 baseline | 10.43 | 1.0× |
| All-NVFP4 | 11.03 | 7.11× |
| NVFP4 attention + `down_proj` | 11.23 | 7.35× |
| NVFP4 `down_proj` only | **11.32** | **7.43×** |
| NVFP4 attention / MXFP4 FFN | 11.33 | 7.40× |
| All-MXFP4 | 11.54 | 7.53× |

Both formats are usable. The full span costs 4.6% accuracy across a 5.9% footprint range — which is precisely why selecting at runtime is worth silicon.

**Layer-wise sensitivity.** Each of the 154 linear projections was switched NVFP4→MXFP4 in isolation. No single layer is a cliff: the largest degradation is +0.066 perplexity, the mean is +0.0027, against a measured resolution limit of ±0.015. Sensitivity concentrates in `down_proj`, not in attention — which is why keeping NVFP4 for `down_proj` alone beats the conventional attention/FFN split on *both* axes.

**Weight reconstruction RMSE.** NVFP4's 16-element FP8 block scale yields 1.24× lower error than MXFP4's 32-element power-of-two scale. The ratio holds to within 1% (1.238–1.248×) across six architectures from 1.1 B to 7.6 B parameters, while the absolute error varies more than tenfold — indicating the gap follows from the scaling scheme itself, not from any model's weight distribution.

---

## ASIC synthesis

Cadence Genus 21.10-p002_1, GPDK045 45 nm (`gsclib045`), `fast_vdd1v2` corner, 400 MHz (2.5 ns), hierarchy preserved (`auto_ungroup none`). All five configurations close timing with zero failing paths.

| Design | Cell area | MAC core | Adapter | Timing |
|---|---:|---:|---:|:--:|
| MXFP4-only | 9,342 | — | — | met |
| NVFP4-only | 24,585 | — | — | met |
| **Unified** | **25,022** | — | — | met |
| Unified + PCPI | 21,450 | 19,970 | 1,480 | met |
| Unified + APU | 21,748 | 20,270 | 1,478 | met |

The MAC core is smaller when instantiated than when synthesized standalone because a synthesis top must drive default external loads on every port and cannot prune logic reachable from its primary inputs, whereas the instantiated core drives known adapter registers with several control inputs tied by the decode logic. Elaboration confirms identical logic in both cases. The Unified vs. NVFP4-only comparison is unaffected: both are standalone builds.

> **Note on `Genus Reports/`.** The committed reports are from an earlier auto-ungrouped flow and do not reproduce the table above, which comes from a hierarchy-preserved run. Those reports are being added.

### Per-mode power (SAIF-annotated, 10,000 vectors)

| Component | MXFP4 mode | NVFP4 mode |
|---|---|---|
| Leakage | 15.328 µW | 15.222 µW |
| Internal | 6.372 mW | 6.914 mW |
| Switching | 8.072 mW | 8.916 mW |
| **Total** | **14.459 mW** | **15.845 mW** |

The difference is localized entirely to the multipliers; the scale decoders register identical toggle counts in both modes. Reduced mantissa activity in the *shared* multipliers, not gating of an idle scale path, is what lowers MXFP4 power.

---

## Architecture

```
            packed_byte                 scale_code_a   scale_code_b  tensor_scale
                 │                            │             │             │
          ┌──────▼──────┐              ┌──────▼─────┐  ┌────▼─────┐      │
          │   u_nib     │              │  u_e8m0    │  │  u_fp8   │      │
          │  (nibble)   │              │  E8M0 →    │  │ FP8 E4M3 │      │
          └──┬───────┬──┘              │ {0,code,0} │  │  decoder │      │
             │       │                 │  0 gates   │  └────┬─────┘      │
        ┌────▼──┐ ┌──▼────┐            └──────┬─────┘       │            │
        │u_dec0 │ │u_dec1 │                   │        ┌────▼────────────▼──┐
        │ E2M1  │ │ E2M1  │                   │        │      u_eff         │
        └────┬──┘ └──┬────┘                   │        │  FP32 multiply     │
             │       │                        │        └─────────┬──────────┘
             │       │                        └──────┐  ┌────────┘
             │       │                            ┌──▼──▼──┐
             │       │                            │  MUX   │◀── FORMAT_MODE
             │       │                            └───┬────┘
             │       │                      eff_scale │
          ┌──▼───────▼──┐                             │
          │  u_smul0/1  │◀────────────────────────────┘
          └──┬───────┬──┘
             │       │        b_elem_0 / b_elem_1
          ┌──▼───────▼──┐
          │  u_dmul0/1  │
          └──┬───────┬──┘
          ┌──▼───────▼──┐    ┌──────────┐    ┌─────────────┐
          │   u_add01   │───▶│ u_accadd │───▶│ accum_out   │
          └─────────────┘    └────▲─────┘    └──────┬──────┘
                                  └─────────────────┘
```

Of the 14 blocks in the datapath, 12 are shared. Only two are format-specific: the E8M0 decoder (zero cells — pure wire routing) and the FP8 E4M3 decoder feeding `u_eff`. The datapath is combinational; `accum_out` is the only register.

Integrated into two RISC-V cores:
- **PicoRV32** — via the Pico Co-Processor Interface (PCPI), custom-0 opcode `0x0B`
- **CV32E40P** — via the Auxiliary Processing Unit (APU), OP-FP opcode `0x53`

Both use a two-instruction protocol: `QMAC_CFG` (packs format mode, E2M1 byte and scale codes; resets the accumulator) and `QMAC_EXEC` (passes two FP32 activations, triggers the MAC, returns the running sum).

---

## Repository structure

```
.
├── rtl/
│   ├── common/          nibble_extractor, e2m1_decoder, fp32_mul, fp32_add
│   ├── mxfp4/           e8m0_decoder, mxfp4_mac_top
│   ├── nvfp4/           fp8_e4m3_decoder, nvfp4_mac_top
│   ├── unified/         unified_mac_top.v          ★ the unified dual-mode MAC
│   ├── pcpi/            PicoRV32 integration + SoC testbench + firmware
│   ├── apu/             CV32E40P integration + SoC testbench + firmware
│   ├── tb/              unit testbenches and Python vector generators
│   ├── synth/           Genus scripts and SDC constraints
│   └── wrapper/         top-level wrappers
├── Phase-1/             Python FP4 quantization library (mxfp4.py, nvfp4.py)
├── Phase-2/             C GEMM benchmarks
├── Phase-3/             LLM weight analysis and RMSE
├── Phase-4/             WikiText-2 perplexity
├── Phase-5/             hardware-accurate Python emulation of the RTL datapath
├── Genus Reports/       synthesis reports (see note above)
└── Results/             RMSE charts across six models
```

---

## Format comparison

| Property | MXFP4 (OCP MX) | NVFP4 (NVIDIA) |
|---|---|---|
| Weight encoding | E2M1 (4-bit) | E2M1 (4-bit) |
| Block scale | E8M0 (8-bit, exponent-only) | FP8 E4M3 (8-bit) |
| Block size | 32 elements | 16 elements |
| Tensor scale | None | FP32, per tensor |
| Scale decode logic | Zero gates (wire routing) | FP8 decoder + FP32 multiplier |
| WikiText-2 PPL (TinyLlama-1.1B) | 11.54 | 11.03 |
| Compression | 7.53× | 7.11× |

---

## Getting started

**Prerequisites.** Xilinx Vivado 2022.2+ for behavioral simulation; Cadence Genus 21.10 with GPDK045 for synthesis; Python 3.8+ with `torch`, `transformers`, `datasets` for the quantization and perplexity work.

### RTL simulation

```bash
cd rtl/tb/
python gen_unified_vectors.py
xvlog --sv unified_mac_tb.sv ../unified/unified_mac_top.v ../common/*.v ../mxfp4/*.v ../nvfp4/*.v
xelab unified_mac_tb -debug all
xsim unified_mac_tb -runall
```

SoC-level, PicoRV32 + PCPI:

```bash
cd rtl/pcpi/
xvlog --sv soc_pcpi_tb.sv soc_pcpi_top.v mac_pcpi_coprocessor.v ../unified/*.v ../common/*.v ../mxfp4/*.v ../nvfp4/*.v
xelab soc_pcpi_tb -debug all
xsim soc_pcpi_tb -runall
```

SoC-level, CV32E40P + APU:

```bash
cd rtl/apu/
xvlog --sv soc_apu_tb.sv soc_apu_top.sv mac_apu_coprocessor.sv ../unified/*.v ../common/*.v ../mxfp4/*.v ../nvfp4/*.v
xelab soc_apu_tb -debug all
xsim soc_apu_tb -runall
```

### ASIC synthesis

```bash
cd rtl/synth/
bash run_all_genus.sh          # all five configurations
genus -files genus_unified.tcl # or individually
```

### Perplexity evaluation

```bash
cd Phase-5/
python phase5_hw_emulated_perplexity.py
```

Perplexity is measured through a Python emulation of the RTL datapath, since 16 KB of on-chip SRAM cannot hold TinyLlama-1.1B's weights. The emulated datapath was validated against RTL simulation over 100 randomized vectors spanning both formats, with bit-exact agreement in the FP32 accumulator output.

---

## Acknowledgments

Conducted as part of a Summer Research Internship Program (SRIP). The PicoRV32 core is by [Claire Wolf / YosysHQ](https://github.com/YosysHQ/picorv32); the CV32E40P core is by the [OpenHW Group](https://github.com/openhwgroup/cv32e40p).
