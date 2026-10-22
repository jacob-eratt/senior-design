# User initiated only: synthesize, implement, check timing, export bitstream/XSA.
set root [file normalize [file join [file dirname [info script]] ..]]
if {[current_project -quiet]==""} {open_project [file join $root build conv0_kv260.xpr]}
set build [get_property DIRECTORY [current_project]]
launch_runs synth_1 -jobs 4
wait_on_run synth_1
if {[get_property PROGRESS [get_runs synth_1]] ne "100%"} {error "Synthesis failed; inspect run log."}
open_run synth_1
file mkdir [file join $build reports]
report_utilization -hierarchical -file [file join $build reports utilization_synth.rpt]
launch_runs impl_1 -to_step route_design -jobs 4
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} {error "Implementation failed; inspect run log."}
open_run impl_1
report_utilization -file [file join $build reports utilization_route.rpt]
report_timing_summary -report_unconstrained -file [file join $build reports timing_route.rpt]
report_drc -file [file join $build reports drc.rpt]
report_clocks -file [file join $build reports clocks.rpt]
report_cdc -file [file join $build reports cdc.rpt]
check_timing -verbose -file [file join $build reports check_timing.rpt]
foreach delay {min max} {
    set path [get_timing_paths -delay_type $delay -max_paths 1]
    if {[llength $path]==0 || [get_property SLACK $path]<0} {error "Timing not met ($delay). No bitstream exported."}
}
# Complete the managed run so write_hw_platform can locate its registered bitstream.
close_design
launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} {error "Bitstream run failed; inspect run log."}
open_run impl_1
set generated_bit [file join [get_property DIRECTORY [get_runs impl_1]] [get_property TOP [current_fileset]].bit]
if {![file exists $generated_bit]} {error "Managed bitstream missing: $generated_bit"}
file copy -force $generated_bit [file join $build conv0_kv260.bit]
write_hw_platform -fixed -include_bit -force -file [file join $build conv0_kv260.xsa]
puts "Exported build/conv0_kv260.bit and .xsa. Review reports before board benchmarking."
