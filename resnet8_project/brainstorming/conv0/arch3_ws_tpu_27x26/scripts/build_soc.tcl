# Source in Vivado 2024.2 with the TPU SoC project open (after create_soc_project.tcl).
# Synthesizes, implements, checks timing, writes the bitstream and exports the XSA.
# Outputs: arch3_ws_tpu_27x26/build/tpu_kv260.bit, tpu_kv260.xsa, soc_reports/*.rpt
# Optional before sourcing: set ws_soc_jobs <N>   (default 4)
set root [file normalize [file join [file dirname [info script]] ..]]
if {[current_project -quiet] eq ""} {error "Open the TPU SoC project first."}
if {[get_property top [current_fileset]] ne "system_wrapper"} {
    error "Top is [get_property top [current_fileset]], expected system_wrapper. Run create_soc_project.tcl first."
}
set jobs [expr {[info exists ws_soc_jobs] ? $ws_soc_jobs : 4}]
set out [file join $root build]
set rpt [file join $out soc_reports]
file mkdir $rpt
proc ws_run_ok {run} {
    if {[get_property PROGRESS [get_runs $run]] ne "100%"} {
        error "$run failed: [get_property STATUS [get_runs $run]]. Inspect its runme.log."
    }
}
set accel_ips [get_ips -quiet *accelerator*]
if {[llength $accel_ips] > 0} {
    update_module_reference $accel_ips
    generate_target all [get_files system.bd]
}
reset_run synth_1
launch_runs synth_1 -jobs $jobs
wait_on_run synth_1
ws_run_ok synth_1
open_run synth_1
report_utilization -hierarchical -file [file join $rpt utilization_synth_hier.rpt]
close_design
launch_runs impl_1 -to_step route_design -jobs $jobs
wait_on_run impl_1
ws_run_ok impl_1
open_run impl_1
report_utilization -file [file join $rpt utilization_route.rpt]
report_utilization -hierarchical -file [file join $rpt utilization_route_hier.rpt]
report_timing_summary -report_unconstrained -max_paths 20 -file [file join $rpt timing_route.rpt]
report_clocks -file [file join $rpt clocks.rpt]
report_cdc -file [file join $rpt cdc.rpt]
report_drc -file [file join $rpt drc.rpt]
check_timing -verbose -file [file join $rpt check_timing.rpt]
foreach delay {max min} {
    set path [get_timing_paths -delay_type $delay -max_paths 1]
    if {[llength $path] == 0 || [get_property SLACK $path] < 0} {
        error "Timing not met ($delay, slack [get_property SLACK $path] ns). See $rpt/timing_route.rpt. No bitstream written."
    }
    puts "Worst $delay slack: [get_property SLACK $path] ns"
}
close_design
launch_runs impl_1 -to_step write_bitstream -jobs $jobs
wait_on_run impl_1
ws_run_ok impl_1
set bit [file join [get_property DIRECTORY [get_runs impl_1]] system_wrapper.bit]
if {![file exists $bit]} {error "Bitstream missing: $bit"}
file copy -force $bit [file join $out tpu_kv260.bit]
open_run impl_1
write_hw_platform -fixed -include_bit -force -file [file join $out tpu_kv260.xsa]
puts "TPU_SOC_BUILT: [file join $out tpu_kv260.xsa]"
puts "Reports: $rpt. Next: create the Vitis platform from this XSA (see DEPLOY_KV260.md)."
