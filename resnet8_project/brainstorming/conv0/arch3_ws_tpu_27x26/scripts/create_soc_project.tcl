# Source in Vivado 2024.2. Generates the PS + DMA + accelerator block design and
# HDL wrapper; does NOT synthesize or simulate.
# Mode 1 (default): no project open -> creates a NEW tpu_soc project.
# Mode 2: with an existing KV260 project open, first run
#           set ws_soc_use_open_project 1
#         to add the block design to that project instead.
# Optional before sourcing: set ws_soc_build_dir <fresh_directory>   (mode 1)
# Optional before sourcing: set ws_soc_board_part <installed_KV260_board_part>
set root [file normalize [file join [file dirname [info script]] ..]]
set ws_soc_open_mode [expr {[info exists ws_soc_use_open_project] && $ws_soc_use_open_project}]
if {[current_project -quiet] ne "" && !$ws_soc_open_mode} {
    error "A project is open. Either close it, or run 'set ws_soc_use_open_project 1' to add the block design to it."
}
if {$ws_soc_open_mode && [current_project -quiet] eq ""} {
    error "ws_soc_use_open_project is set but no project is open."
}
if {$ws_soc_open_mode} {
    if {[llength [get_files -quiet system.bd]] > 0} {
        error "This project already has system.bd. Remove it or use a fresh project."
    }
    set ws_soc_part [get_property PART [current_project]]
    if {![string match xck26-sfvc784-2LV-* $ws_soc_part]} {
        error "Open project targets $ws_soc_part, expected xck26-sfvc784-2LV-c (KV260)."
    }
    if {![string match *kv260* [get_property BOARD_PART [current_project]]]} {
        set boards [get_board_parts -quiet *kv260*]
        if {[llength $boards] == 0} {error "Install the KV260 board files first; the PS DDR/MIO preset is required."}
        set_property board_part [lindex [lsort -dictionary $boards] end] [current_project]
    }
    puts "Using open project [current_project], board preset: [get_property BOARD_PART [current_project]]"
    set build [get_property DIRECTORY [current_project]]
    set ws_soc_have {}
    foreach f [get_files -quiet -of_objects [get_filesets sources_1]] {lappend ws_soc_have [file tail $f]}
    foreach f [concat [glob [file join $root rtl *.sv]] [list [file join $root rtl ws_axi_bd.v]]] {
        if {[lsearch -exact $ws_soc_have [file tail $f]] < 0} {
            add_files -fileset sources_1 -norecurse $f
            puts "Added [file tail $f]"
        }
    }
    set_property file_type SystemVerilog [get_files -quiet *.sv]
    update_compile_order -fileset sources_1
} else {
if {[info exists ws_soc_build_dir]} {
    set build [file normalize $ws_soc_build_dir]
} else {
    # main's layout: resnet8_project/brainstorming/conv0/<architecture>.
    set repo [file normalize [file join $root .. .. .. ..]]
    set build [file join $repo .vivado tpu_soc_[clock format [clock seconds] -format %Y%m%d_%H%M%S]]
}
if {[file exists $build]} {
    error "Output directory already exists: $build. Select a fresh ws_soc_build_dir."
}
set boards [get_board_parts -quiet *kv260*]
if {[info exists ws_soc_board_part]} {
    set board $ws_soc_board_part
    if {[lsearch -exact $boards $board] < 0} {error "Requested KV260 board preset not installed: $board"}
} elseif {[llength $boards] > 0} {
    set board [lindex [lsort -dictionary $boards] end]
} else {
    error "Install the KV260 board files first; the PS DDR/MIO preset is required."
}
create_project tpu_soc $build -part xck26-sfvc784-2LV-c
set_property board_part $board [current_project]
puts "Using board preset: $board"
add_files [glob [file join $root rtl *.sv]]
add_files [file join $root rtl ws_axi_bd.v]
set_property file_type SystemVerilog [get_files *.sv]
update_compile_order -fileset sources_1
}
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
# Plain-Verilog, attribute-marked shell around the SystemVerilog struct design.
create_bd_cell -type module -reference ws_axi_bd accelerator
foreach bus {S_AXI S_AXIS M_AXIS} {
    if {[llength [get_bd_intf_pins -quiet accelerator/$bus]] != 1} {
        error "Accelerator interface $bus was not recognized; inspect ws_axi_bd.v."
    }
}
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
puts "TPU_SOC_CREATED: [current_project] in $build (top = system_wrapper)"
puts "Inspect system.bd, clocks/resets and Address Editor. No synthesis or simulation launched."
puts "Next: source [file join $root scripts build_soc.tcl]"
