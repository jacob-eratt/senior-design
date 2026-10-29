# USER-RUN ONLY. Out-of-context synthesis; no simulation, placement or bitstream.
# vivado -mode batch -source scripts/synth_components.tcl -tclargs mac_int8_mixed
# Valid tops: mac_int8_mixed, conv0_requant, conv0_result_fifo, conv0_832_core
# Use a separate Vivado process. Never touches the integrated project.
set root [file normalize [file join [file dirname [info script]] ..]]
if {$argc != 1} {error "Supply exactly one component top"}
set top [lindex $argv 0]
if {$top ni {mac_int8_mixed conv0_requant conv0_result_fifo conv0_832_core}} {error "Unsupported component"}
set dest [file join $root build components $top]
if {[file exists $dest]} {error "Output already exists: $dest. Preserve it; select a fresh copy of this project to rerun."}
file mkdir $dest
create_project -in_memory -part xck26-sfvc784-2LV-c
read_verilog -sv [file join $root src conv0_quant_pkg.sv]
foreach source {mac_int8_mixed conv0_832_core conv0_requant conv0_result_fifo} {
    read_verilog -sv [file join $root src ${source}.sv]
}
set constraints [file join $dest component.xdc]
set handle [open $constraints w]
puts $handle {create_clock -name component_clk -period 10.000 [get_ports clk]}
close $handle
read_xdc $constraints
synth_design -top $top -part xck26-sfvc784-2LV-c -mode out_of_context -flatten_hierarchy rebuilt
report_utilization -hierarchical -file [file join $dest utilization_synth.rpt]
report_timing_summary -report_unconstrained -file [file join $dest timing_synth.rpt]
check_timing -verbose -file [file join $dest check_timing.rpt]
write_checkpoint [file join $dest component.dcp]
puts "Component synthesis only. OOC port timing is incomplete; this does not establish integrated fit or routed timing."
