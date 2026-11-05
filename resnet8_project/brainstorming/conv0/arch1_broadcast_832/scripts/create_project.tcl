# Vivado 2024.2. Run: vivado -mode batch -source scripts/create_project.tcl
# Optional: -tclargs <installed KV260 board_part> <fresh build directory>
set root [file normalize [file join [file dirname [info script]] ..]]
set build [file join $root build]
if {$argc>1} {set build [file normalize [lindex $argv 1]]}
if {[file exists [file join $build conv0_kv260.xpr]]} {
    error "Project already exists. Open build/conv0_kv260.xpr, or choose a fresh build directory."
}
create_project conv0_kv260 $build -part xck26-sfvc784-2LV-c
set boards [get_board_parts -quiet *kv260*]
if {$argc>0} {set board [lindex $argv 0]} elseif {[llength $boards]>0} {set board [lindex [lsort -dictionary $boards] end]} else {
    error "Install AMD KV260 board files in Vivado, then rerun. PS DDR/MIO configuration must use the board preset."
}
set_property board_part $board [current_project]
puts "Using board preset: $board"
add_files [glob [file join $root src *.sv]]
add_files [file join $root src conv0_axi_bd.v]
set_property file_type SystemVerilog [get_files *.sv]
update_compile_order -fileset sources_1
create_bd_design system
create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e:* ps
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e -config {apply_board_preset "1"} [get_bd_cells ps]
set_property -dict [list CONFIG.PSU__USE__M_AXI_GP0 {1} CONFIG.PSU__MAXIGP0__DATA_WIDTH {32} \
    CONFIG.PSU__USE__M_AXI_GP1 {0} CONFIG.PSU__USE__M_AXI_GP2 {0} \
    CONFIG.PSU__USE__S_AXI_GP2 {1} CONFIG.PSU__SAXIGP2__DATA_WIDTH {128} \
    CONFIG.PSU__FPGA_PL0_ENABLE {1} CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ {100} \
    CONFIG.PSU__FPGA_PL1_ENABLE {1} CONFIG.PSU__CRL_APB__PL1_REF_CTRL__FREQMHZ {200}] [get_bd_cells ps]
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:* dma
set_property -dict [list CONFIG.c_include_sg {0} CONFIG.c_include_mm2s {1} CONFIG.c_include_s2mm {1} \
    CONFIG.c_addr_width {64} CONFIG.c_sg_length_width {23} \
    CONFIG.c_m_axi_mm2s_data_width {256} CONFIG.c_m_axi_s2mm_data_width {256} \
    CONFIG.c_m_axis_mm2s_tdata_width {256} CONFIG.c_s_axis_s2mm_tdata_width {256} \
    CONFIG.c_mm2s_burst_size {16} CONFIG.c_s2mm_burst_size {16} \
    CONFIG.c_include_mm2s_dre {0} CONFIG.c_include_s2mm_dre {0}] [get_bd_cells dma]
create_bd_cell -type module -reference conv0_axi_bd accelerator
create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:* ctrl
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {2}] [get_bd_cells ctrl]
create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:* memory
set_property -dict [list CONFIG.NUM_SI {2} CONFIG.NUM_MI {1} CONFIG.NUM_CLKS {2}] [get_bd_cells memory]
create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:* reset
set_property CONFIG.C_EXT_RESET_HIGH 0 [get_bd_cells reset]
create_bd_cell -type ip -vlnv xilinx.com:ip:xlconstant:* one
set_property CONFIG.CONST_VAL 1 [get_bd_cells one]
connect_bd_intf_net [get_bd_intf_pins ps/M_AXI_HPM0_FPD] [get_bd_intf_pins ctrl/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins ctrl/M00_AXI] [get_bd_intf_pins accelerator/S_AXI]
connect_bd_intf_net [get_bd_intf_pins ctrl/M01_AXI] [get_bd_intf_pins dma/S_AXI_LITE]
connect_bd_intf_net [get_bd_intf_pins dma/M_AXI_MM2S] [get_bd_intf_pins memory/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins dma/M_AXI_S2MM] [get_bd_intf_pins memory/S01_AXI]
connect_bd_intf_net [get_bd_intf_pins memory/M00_AXI] [get_bd_intf_pins ps/S_AXI_HP0_FPD]
connect_bd_intf_net [get_bd_intf_pins dma/M_AXIS_MM2S] [get_bd_intf_pins accelerator/S_AXIS]
connect_bd_intf_net [get_bd_intf_pins accelerator/M_AXIS] [get_bd_intf_pins dma/S_AXIS_S2MM]
foreach pin {ps/maxihpm0_fpd_aclk dma/s_axi_lite_aclk dma/m_axi_mm2s_aclk dma/m_axi_s2mm_aclk ctrl/aclk memory/aclk accelerator/aclk reset/slowest_sync_clk} {
    connect_bd_net [get_bd_pins ps/pl_clk0] [get_bd_pins $pin]
}
connect_bd_net [get_bd_pins ps/pl_clk1] [get_bd_pins memory/aclk1] [get_bd_pins ps/saxihp0_fpd_aclk]
connect_bd_net [get_bd_pins ps/pl_resetn0] [get_bd_pins reset/ext_reset_in]
connect_bd_net [get_bd_pins one/dout] [get_bd_pins reset/dcm_locked]
foreach pin {dma/axi_resetn ctrl/aresetn memory/aresetn accelerator/aresetn} {
    connect_bd_net [get_bd_pins reset/peripheral_aresetn] [get_bd_pins $pin]
}
assign_bd_address
set accseg [get_bd_addr_segs -of_objects [get_bd_addr_spaces ps/Data] -filter {NAME =~ *accelerator*}]
set dmaseg [get_bd_addr_segs -of_objects [get_bd_addr_spaces ps/Data] -filter {NAME =~ *dma*}]
if {[llength $accseg]!=1 || [llength $dmaseg]!=1} {error "Unexpected control address segments; inspect address editor."}
set_property offset 0xA0000000 $accseg
set_property range 64K $accseg
set_property offset 0xA0010000 $dmaseg
set_property range 64K $dmaseg
validate_bd_design
save_bd_design
generate_target all [get_files system.bd]
set wrapper [make_wrapper -files [get_files system.bd] -top]
add_files -norecurse $wrapper
set_property top system_wrapper [current_fileset]
update_compile_order -fileset sources_1
puts "Project created. Next: source [file join $root scripts build.tcl]"
