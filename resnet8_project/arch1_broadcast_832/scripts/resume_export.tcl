# Resume only the already-routed project's managed bitstream/export stage.
set root [file normalize [file join [file dirname [info script]] ..]]
set handle [open [file join $root build last_integrated_project.txt] r]
set project [string trim [read $handle]]
close $handle
open_project $project
set build [get_property DIRECTORY [current_project]]
open_run impl_1
foreach delay {min max} {
    set path [get_timing_paths -delay_type $delay -max_paths 1]
    if {[llength $path]==0 || [get_property SLACK $path]<0} {error "Routed timing not met ($delay)"}
}
close_design
launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} {error "Managed bitstream generation failed"}
open_run impl_1
set generated_bit [file join [get_property DIRECTORY [get_runs impl_1]] [get_property TOP [current_fileset]].bit]
if {![file exists $generated_bit]} {error "Managed bitstream missing: $generated_bit"}
file copy -force $generated_bit [file join $build conv0_kv260.bit]
write_hw_platform -fixed -include_bit -force -file [file join $build conv0_kv260.xsa]
puts "EXPORTED: $build/conv0_kv260.bit and $build/conv0_kv260.xsa"
