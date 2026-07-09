#!/bin/bash
# ============================================================
# run_all_genus.sh - Synthesize all 3 MAC designs
# Run from: Satyam_SRIP/genus/
# Usage: bash run_all_genus.sh
# ============================================================

SCRIPT_DIR=/home/redhatacademy19/Documents/Satyam_SRIP/scripts

echo "============================================"
echo "  MAC Unit Synthesis - FreePDK 45nm"
echo "============================================"
echo ""

# Step 1: MXFP4
echo "[1/3] Synthesizing MXFP4 MAC..."
genus -f ${SCRIPT_DIR}/genus_mxfp4.tcl -log genus_mxfp4.log
echo "  Done."
echo ""

# Step 2: NVFP4
echo "[2/3] Synthesizing NVFP4 MAC..."
genus -f ${SCRIPT_DIR}/genus_nvfp4.tcl -log genus_nvfp4.log
echo "  Done."
echo ""

# Step 3: Unified
echo "[3/3] Synthesizing Unified MAC..."
genus -f ${SCRIPT_DIR}/genus_unified.tcl -log genus_unified.log
echo "  Done."
echo ""

echo "============================================"
echo "  All synthesis complete!"
echo "  Check OUTPUT/ directory for results"
echo "============================================"
