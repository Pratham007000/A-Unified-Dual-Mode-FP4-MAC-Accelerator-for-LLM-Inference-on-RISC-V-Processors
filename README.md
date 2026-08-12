# A Unified Dual-Mode FP4 MAC Accelerator for LLM Inference on RISC-V Processors

> A unified MAC datapath supporting both MXFP4 (OCP Microscaling) and NVFP4 (NVIDIA FP4) formats with RISC-V co-processor integration.

---

## Overview

Large Language Models (LLMs) demand extreme weight compression for edge deployment. Two competing 4-bit floating-point standards — **MXFP4** and **NVFP4** — offer 7–8× compression over FP32 but use fundamentally different block-scaling mechanisms. Supporting both naively requires two separate MAC datapaths, roughly doubling area and power.

This project presents a **unified dual-mode MAC unit** that handles both formats in a single datapath by exploiting their shared E2M1 weight representation and FP32 arithmetic back-end. A one-bit `FORMAT_MODE` signal switches between formats at runtime with zero reconfiguration overhead.

### Key Results (45 nm, FreePDK, 500 MHz)

| Metric | Unified MAC | Two Separate MACs | Savings |
|--------|-------------|-------------------|---------|
| Cell Area | 30,430 | 44,319 | **31%** |
| Total Power | 19.8 mW | 44.9 mW | **56%** |
| vs. NVFP4-only | +2% area | — | Near-zero cost for MXFP4 support |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Unified MAC Datapath                      │
│                                                             │
│  ┌──────────┐    ┌────────────┐                             │
│  │  Nibble   │───▶│  E2M1 LUT  │──┐                         │
│  │ Extractor │    │  Decoder×2 │  │                         │
│  └──────────┘    └────────────┘  │                         │
│       (Shared)       (Shared)     │                         │
│                                   ▼                         │
│  ┌──────────────────────────────────────┐                   │
│  │       Parallel Scale Decoding        │                   │
│  │  ┌─────────┐    ┌──────────────┐     │                   │
│  │  │  E8M0   │    │ FP8 E4M3    │     │                   │
│  │  │ Decoder │    │ Decoder +   │     │                   │
│  │  │(0 gates)│    │ FP32 × Tscl │     │                   │
│  │  └────┬────┘    └──────┬──────┘     │                   │
│  │       │    ┌──────┐    │            │                   │
│  │       └───▶│ MUX  │◀───┘            │                   │
│  │            │(MODE)│                 │                   │
│  │            └──┬───┘                 │                   │
│  └───────────────┼─────────────────────┘                   │
│                  ▼                                          │
│  ┌──────────────────────────────┐                           │
│  │  FP32 Scale × Dot Product   │  (4 multipliers, shared)  │
│  │  FP32 Accumulator + Adder   │                           │
│  └──────────────────────────────┘                           │
└─────────────────────────────────────────────────────────────┘
```

The design is integrated into two RISC-V cores via co-processor interfaces:
- **PicoRV32** — via Pico Co-Processor Interface (PCPI)
- **CV32E40P** — via Auxiliary Processing Unit (APU)

---

## Repository Structure

```
.
├── rtl/                        # Verilog/SystemVerilog RTL source
│   ├── common/                 # Shared arithmetic blocks
│   │   ├── nibble_extractor.v
│   │   ├── e2m1_decoder.v      # 16-entry LUT: 4-bit → FP32
│   │   ├── fp32_mul.v          # IEEE-754 FP32 multiplier
│   │   └── fp32_add.v          # IEEE-754 FP32 adder
│   ├── mxfp4/                  # MXFP4-specific modules
│   │   ├── e8m0_decoder.v      # E8M0 scale decoder (zero gates)
│   │   └── mxfp4_mac_top.v     # Standalone MXFP4 MAC
│   ├── nvfp4/                  # NVFP4-specific modules
│   │   ├── fp8_e4m3_decoder.v  # FP8 E4M3 block-scale decoder
│   │   └── nvfp4_mac_top.v     # Standalone NVFP4 MAC
│   ├── unified/
│   │   └── unified_mac_top.v   # ★ Unified dual-mode MAC
│   ├── pcpi/                   # PicoRV32 PCPI integration
│   │   ├── mac_pcpi_coprocessor.v
│   │   ├── soc_pcpi_top.v      # PicoRV32 SoC wrapper
│   │   ├── soc_pcpi_tb.sv      # SoC testbench
│   │   ├── firmware_pcpi.c     # Test firmware (C source)
│   │   └── firmware_pcpi.hex   # Compiled firmware
│   ├── apu/                    # CV32E40P APU integration
│   │   ├── mac_apu_coprocessor.sv
│   │   ├── soc_apu_top.sv      # CV32E40P SoC wrapper
│   │   ├── soc_apu_tb.sv       # SoC testbench
│   │   └── firmware_apu.hex    # Compiled firmware
│   ├── tb/                     # Unit-level testbenches
│   │   ├── mxfp4_mac_tb.sv
│   │   ├── nvfp4_mac_tb.sv
│   │   ├── unified_mac_tb.sv
│   │   ├── accuracy_tb.sv      # Comprehensive accuracy verification
│   │   ├── gen_test_vectors.py  # Python vector generators
│   │   ├── gen_unified_vectors.py
│   │   └── gen_accuracy_vectors.py
│   ├── synth/                  # ASIC synthesis scripts
│   │   ├── genus_mxfp4.tcl
│   │   ├── genus_nvfp4.tcl
│   │   ├── genus_unified.tcl
│   │   ├── genus_pcpi.tcl
│   │   ├── genus_apu.tcl
│   │   ├── mac_unit.sdc        # Timing constraints (500 MHz)
│   │   └── run_all_genus.sh    # Batch synthesis runner
│   └── wrapper/                # Top-level wrappers
│
├── Phase-1/                    # Python FP4 quantization library
│   ├── fp_utils.py             # IEEE-754 FP32 utilities
│   ├── mxfp4.py                # MXFP4 quantizer (E2M1 + E8M0)
│   ├── nvfp4.py                # NVFP4 quantizer (E2M1 + FP8×tensor)
│   └── test_quantizers.py      # Unit tests
│
├── Phase-2/                    # C GEMM benchmarks
│   ├── bench_gemm.c            # Matrix multiply benchmark
│   ├── gemm_mxfp4.h            # MXFP4 GEMM kernel
│   ├── gemm_nvfp4.h            # NVFP4 GEMM kernel
│   └── quant_types.h           # Quantization type definitions
│
├── Phase-3/                    # LLM weight analysis
│   ├── step1_inspect_model.py  # Model weight distribution analysis
│   └── step2_quantize_weights.py  # Weight quantization + RMSE
│
├── Phase-4/                    # Perplexity evaluation
│   └── phase4_perplexity.py    # WikiText-2 perplexity measurement
│
├── Phase-5/                    # Hardware-accurate emulation
│   ├── phase5_hw_emulated_perplexity.py   # Bit-exact HW emulator
│   ├── phase5_unified_perplexity.py       # Unified MAC emulation
│   └── phase5_ri5cy_apu_perplexity.py     # RI5CY APU emulation
│
├── Genus Reports/              # ASIC synthesis reports
│   ├── MXFP4_Output/
│   ├── NVFP4_Output/
│   ├── Unified_Output/
│   ├── PCPI_OUTPUT/
│   ├── Apu_Output/
│   └── Wrapper_Output/
│
├── Results/                    # RMSE bar charts (6 LLMs)
│
├── paper/                      # ASP-DAC 2027 manuscript
│   ├── aspdac_paper.tex
│   └── refs.bib
│
└── README.md
```

---

## FP4 Format Comparison

| Property | MXFP4 (OCP MX) | NVFP4 (NVIDIA) |
|----------|-----------------|----------------|
| Weight encoding | E2M1 (4-bit) | E2M1 (4-bit) |
| Block scale format | E8M0 (8-bit) | FP8 E4M3 (8-bit) |
| Block size | 32 elements | 16 elements |
| Tensor scale | None | FP32 |
| Scale decode logic | Zero gates (wire routing) | FP8 decoder + FP32 multiplier |
| WikiText-2 PPL (TinyLlama) | 100.6 ❌ | 11.07 ✅ |

**Mixed-precision strategy:** NVFP4 for attention layers + MXFP4 for FFN layers → **PPL 15.36** at 7.4× compression.

---

## Getting Started

### Prerequisites

- **RTL Simulation:** [Xilinx Vivado](https://www.xilinx.com/products/design-tools/vivado.html) 2022.2+ (behavioral simulation)
- **ASIC Synthesis:** [Cadence Genus](https://www.cadence.com/) 21.14 + FreePDK 45 nm library
- **Python Emulation:** Python 3.8+ with `torch`, `transformers`, `datasets`

### Running RTL Simulations

**Unit-level MAC testbench:**
```bash
# In Vivado Tcl console or command line
cd rtl/tb/

# Generate test vectors
python gen_unified_vectors.py

# Run simulation (Vivado)
xvlog --sv unified_mac_tb.sv ../unified/unified_mac_top.v ../common/*.v ../mxfp4/*.v ../nvfp4/*.v
xelab unified_mac_tb -debug all
xsim unified_mac_tb -runall
```

**SoC-level simulation (PicoRV32 + PCPI):**
```bash
cd rtl/pcpi/
# Simulate in Vivado — loads firmware_pcpi.hex automatically
xvlog --sv soc_pcpi_tb.sv soc_pcpi_top.v mac_pcpi_coprocessor.v ../unified/*.v ../common/*.v ../mxfp4/*.v ../nvfp4/*.v
xelab soc_pcpi_tb -debug all
xsim soc_pcpi_tb -runall
```

**SoC-level simulation (CV32E40P + APU):**
```bash
cd rtl/apu/
xvlog --sv soc_apu_tb.sv soc_apu_top.sv mac_apu_coprocessor.sv ../unified/*.v ../common/*.v ../mxfp4/*.v ../nvfp4/*.v
xelab soc_apu_tb -debug all
xsim soc_apu_tb -runall
```

### Running ASIC Synthesis

```bash
cd rtl/synth/

# Synthesize all configurations
bash run_all_genus.sh

# Or run individually
genus -files genus_unified.tcl    # Unified MAC
genus -files genus_pcpi.tcl       # Unified + PicoRV32 PCPI
genus -files genus_apu.tcl        # Unified + CV32E40P APU
```

### Running Perplexity Evaluation

```bash
# Phase 5: Hardware-accurate perplexity (bit-exact RTL emulation)
cd Phase-5/
python phase5_hw_emulated_perplexity.py
```

This emulates the exact RTL datapath in Python, quantizing weights through the same decode pipeline as the Verilog modules, and measures WikiText-2 perplexity on TinyLlama-1.1B.

---

## ASIC Synthesis Results

**Technology:** FreePDK 45 nm | **Target Clock:** 500 MHz (2.0 ns period)

| Configuration | Cell Area | Total Power (mW) | Timing Met |
|--------------|-----------|-------------------|------------|
| MXFP4 MAC (standalone) | 13,278 | 7.3 | ✅ |
| NVFP4 MAC (standalone) | 31,041 | 37.6 | ✅ |
| **Unified MAC** | **30,430** | **19.8** | ✅ |
| Unified + PCPI adapter | 33,488 | 21.8 | ✅ |
| Unified + APU adapter | 33,610 | 22.1 | ✅ |

---

## Perplexity Results (TinyLlama-1.1B, WikiText-2)

| Quantization Strategy | Perplexity | vs. FP32 Baseline |
|----------------------|------------|-------------------|
| FP32 (baseline) | 10.43 | — |
| All-NVFP4 | 11.07 | +6.1% |
| Mixed (NVFP4 attn + MXFP4 FFN) | 15.36 | +47.3% |
| All-MXFP4 | 100.60 | ❌ Unusable |

---

## RISC-V Integration

The unified MAC connects to RISC-V cores through a **two-instruction protocol**:

| Instruction | PCPI (PicoRV32) | APU (CV32E40P) | Function |
|-------------|-----------------|----------------|----------|
| **LOAD** | `custom0` funct3=0 | `fadd.s` (OP-FP) | Load weights, scales, activations, mode |
| **EXEC** | `custom0` funct3=1 | `fmul.s` (OP-FP) | Trigger MAC, return accumulated result |

- **PicoRV32:** 975 ns test completion (single-issue, serialized)
- **CV32E40P:** 415 ns test completion (4-stage pipeline, 2.3× faster)

---

## Project Phases

| Phase | Description | Key Output |
|-------|-------------|------------|
| **Phase 1** | Python FP4 quantization library | `mxfp4.py`, `nvfp4.py` — bit-exact quantizers |
| **Phase 2** | C GEMM benchmarks | Performance comparison of FP4 matrix multiply |
| **Phase 3** | LLM weight analysis | RMSE measurements across 6 models (1.1B–7.6B params) |
| **Phase 4** | Perplexity evaluation | WikiText-2 perplexity with format-aware quantization |
| **Phase 5** | Hardware-accurate emulation | Bit-exact RTL datapath emulation in Python |

---
<!--
## Citation

If you use this work, please cite:

```bibtex
@inproceedings{fp4mac2027,
  title={A Unified Dual-Mode {FP4} {MAC} Accelerator for {LLM} Inference on {RISC-V} Processors},
  booktitle={Proceedings of the 32nd Asia and South Pacific Design Automation Conference (ASP-DAC)},
  year={2027}
}
```

---

## License

This project is released for academic and research use. See individual source files for details.

---
-->

## Acknowledgments

This work was conducted as part of a Summer Research Internship Program (SRIP). The PicoRV32 core is by [Claire Wolf / YosysHQ](https://github.com/YosysHQ/picorv32). The CV32E40P core is by [OpenHW Group](https://github.com/openhwgroup/cv32e40p).
