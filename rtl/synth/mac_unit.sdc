# mac_unit.sdc - Timing constraints for MAC units
# Used by all 3 designs (MXFP4, NVFP4, Unified)

# Clock: 500 MHz target (2ns period)
create_clock -name clk -period 2.0 [get_ports clk]

# Input/output delays
set_input_delay  0.2 -clock clk [remove_from_collection [all_inputs] [get_ports clk]]
set_output_delay 0.2 -clock clk [all_outputs]

# Reset is async but constrain it
set_false_path -from [get_ports rst]

# Drive and load
set_driving_cell -lib_cell INVX2 [all_inputs]
set_load 0.01 [all_outputs]
