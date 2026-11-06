# Optional USER run; no DMA/PS IP required. Tests accelerator interfaces directly.
set root [file normalize [file join [file dirname [info script]] ..]]
# Optional batch arguments: <fixture directory> <fresh simulation directory>.
# These allow independent-image and arithmetic-stress runs without replacing golden files.
set fixture [file normalize [file join $root .. fixtures]]
set simdir [file join $root build sim_[clock format [clock seconds] -format %Y%m%d_%H%M%S]]
if {[info exists argc] && $argc>0} {
    if {$argc!=2} {error "Expected fixture directory and fresh simulation directory"}
    set fixture [file normalize [lindex $argv 0]]
    set simdir [file normalize [lindex $argv 1]]
}
if {[file exists $simdir]} {error "Simulation directory already exists; use a fresh directory: $simdir"}
if {![file exists [file join $fixture manifest.json]]} {error "Missing fixture manifest: $fixture"}
create_project conv0_axi_sim $simdir -part xck26-sfvc784-2LV-c
add_files [glob [file join $root src *.sv]]
add_files [file join $root src conv0_axi_bd.v]
add_files -fileset sim_1 [file join $root sim tb_conv0_axi.sv]
set_property top tb_conv0_axi [get_filesets sim_1]
set_property -name xsim.simulate.xsim.more_options -value "-testplusarg {FIXTURE_DIR=$fixture}" -objects [get_filesets sim_1]
set_property xsim.simulate.custom_tcl [file join $root scripts run_no_waves.tcl] [get_filesets sim_1]
update_compile_order -fileset sources_1
update_compile_order -fileset sim_1
launch_simulation
# Close XSim before reading its log; the file is buffered while the snapshot is open.
close_sim
set simlog [file join $simdir conv0_axi_sim.sim sim_1 behav xsim simulate.log]
set log_handle [open $simlog r]
set log_text [read $log_handle]
close $log_handle
if {[string first "PASS tb_conv0_axi:" $log_text]<0 || [regexp -nocase {fatal:|error:} $log_text]} {
    error "Simulation did not pass. Inspect $simlog"
}
puts "Verified testbench PASS in $simlog"
