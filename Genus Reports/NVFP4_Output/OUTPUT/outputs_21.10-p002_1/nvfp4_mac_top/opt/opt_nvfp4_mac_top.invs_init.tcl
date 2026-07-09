################################################################################
#
# Init setup file
# Created by Genus(TM) Synthesis Solution on 06/17/2026 00:16:25
#
################################################################################
if { ![is_common_ui_mode] } { error "ERROR: This script requires common_ui to be active."}

read_mmmc OUTPUT/outputs_21.10-p002_1/nvfp4_mac_top/opt/opt_nvfp4_mac_top.mmmc.tcl

read_physical -lef {/home/redhatacademy19/Documents/Satyam_SRIP/gsclib045_all_v4.8/gsclib045/lef/gsclib045_tech.lef /home/redhatacademy19/Documents/Satyam_SRIP/gsclib045_all_v4.8/gsclib045/lef/gsclib045_macro.lef /home/redhatacademy19/Documents/Satyam_SRIP/gsclib045_all_v4.8/gsclib045/lef/gsclib045_multibitsDFF.lef /home/redhatacademy19/Documents/Satyam_SRIP/gsclib045_all_v4.8/gsclib045_hvt/lef/gsclib045_hvt_macro.lef /home/redhatacademy19/Documents/Satyam_SRIP/gsclib045_all_v4.8/gsclib045_lvt/lef/gsclib045_lvt_macro.lef}

read_netlist OUTPUT/outputs_21.10-p002_1/nvfp4_mac_top/opt/opt_nvfp4_mac_top.v.gz

init_design
