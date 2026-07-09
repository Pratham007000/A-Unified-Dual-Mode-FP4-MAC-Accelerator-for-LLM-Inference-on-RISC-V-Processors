# ============================================================
# genus_mxfp4.tcl - Genus synthesis for MXFP4 MAC Unit
# Target: FreePDK 45nm (gsclib045)
# Run from genus/ folder: genus -f ../scripts/genus_mxfp4.tcl
# ============================================================

# ---- System info ----
if {[file exists /proc/cpuinfo]} {
  sh grep "model name" /proc/cpuinfo
  sh grep "cpu MHz"    /proc/cpuinfo
}
puts "Hostname : [info hostname]"

# ---- Design name ----
set DESIGN mxfp4_mac_top
set SYN_EFF medium
set MAP_EFF medium
set OPT_EFF medium

# ---- Output directories ----
set RELEASE [lindex [get_db program_version] end]
set _OUTPUTS_PATH OUTPUT/outputs_${RELEASE}
set _REPORTS_PATH OUTPUT/reports_${RELEASE}

if {![file exists ${_OUTPUTS_PATH}]} {
  file mkdir ${_OUTPUTS_PATH}
  puts "Creating directory ${_OUTPUTS_PATH}"
}
if {![file exists ${_REPORTS_PATH}]} {
  file mkdir ${_REPORTS_PATH}
  puts "Creating directory ${_REPORTS_PATH}"
}

# ============================================================
# PATHS - All relative to Satyam_SRIP base directory
# ============================================================
set BASE_DIR /home/redhatacademy19/Documents/Satyam_SRIP
set PDK_DIR  /home/redhatacademy19/Documents/Satyam_SRIP

set rtlDir    ${BASE_DIR}/rtl
set scriptDir ${BASE_DIR}/scripts
set genusDir  ${BASE_DIR}/genus

set libDir [list \
  ${PDK_DIR}/gsclib045_all_v4.8/gsclib045/timing      \
  ${PDK_DIR}/gsclib045_all_v4.8/gsclib045_hvt/timing   \
  ${PDK_DIR}/gsclib045_all_v4.8/gsclib045_lvt/timing   \
]

set libList [list \
  fast_vdd1v2_basicCells.lib          \
  fast_vdd1v2_multibitsDFF.lib        \
  fast_vdd1v2_basicCells_hvt.lib      \
  fast_vdd1v2_basicCells_lvt.lib      \
]
# ============================================================

# ---- Tool settings ----
set_db init_lib_search_path $libDir
set_db script_search_path [list $scriptDir]
set_db init_hdl_search_path [list $rtlDir]
set_db max_cpus_per_server 8
set_db information_level 9
set_db tns_opto true

# ---- Read libraries ----
puts "Reading libraries..."
read_libs $libList

read_physical -lefs [list \
  ${PDK_DIR}/gsclib045_all_v4.8/gsclib045/lef/gsclib045_tech.lef          \
  ${PDK_DIR}/gsclib045_all_v4.8/gsclib045/lef/gsclib045_macro.lef         \
  ${PDK_DIR}/gsclib045_all_v4.8/gsclib045/lef/gsclib045_multibitsDFF.lef  \
  ${PDK_DIR}/gsclib045_all_v4.8/gsclib045_hvt/lef/gsclib045_hvt_macro.lef \
  ${PDK_DIR}/gsclib045_all_v4.8/gsclib045_lvt/lef/gsclib045_lvt_macro.lef \
]

set_db qrc_tech_file ${PDK_DIR}/gsclib045_all_v4.8/gsclib045_tech/qrc/qx/gpdk045.tch

# ---- Read RTL ----
puts "Reading RTL..."
set rtlList [list \
  ${rtlDir}/common/nibble_extractor.v \
  ${rtlDir}/common/e2m1_decoder.v     \
  ${rtlDir}/common/fp32_mul.v         \
  ${rtlDir}/common/fp32_add.v         \
  ${rtlDir}/mxfp4/e8m0_decoder.v      \
  ${rtlDir}/mxfp4/mxfp4_mac_top.v     \
]

read_hdl $rtlList

# ---- Elaborate ----
elaborate $DESIGN
set_top_module $DESIGN

# ---- Read constraints ----
read_sdc ${genusDir}/mac_unit.sdc

check_design -unresolved

# ============================================================
# Synthesize
# ============================================================

syn_generic
write_snapshot -directory $_OUTPUTS_PATH/$DESIGN/generic -tag generic
report_summary -directory $_REPORTS_PATH/$DESIGN/generic
report_power > $_REPORTS_PATH/$DESIGN/generic/power.rpt
puts "Runtime & Memory after 'syn_generic'"
time_info GENERIC

syn_map
write_snapshot -directory $_OUTPUTS_PATH/$DESIGN/mapped -tag mapped
report_summary -directory $_REPORTS_PATH/$DESIGN/mapped
report_power > $_REPORTS_PATH/$DESIGN/mapped/power.rpt
puts "Runtime & Memory after 'syn_map'"
time_info MAPPED

syn_opt
write_snapshot -innovus -directory $_OUTPUTS_PATH/$DESIGN/opt -tag opt
report_summary -directory $_REPORTS_PATH/$DESIGN/opt
report_power > $_REPORTS_PATH/$DESIGN/opt/power.rpt
puts "Runtime & Memory after 'syn_opt'"
time_info OPT

# ---- Write final database ----
write_db -to_file ${DESIGN}.db

puts "Final Runtime & Memory."
time_info FINAL
puts "============================"
puts "MXFP4 MAC Synthesis Finished"
puts "============================"
