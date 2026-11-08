# Run one self-checking testbench in Vivado XSim (project mode, KV260 part).
# Batch:   vivado -mode batch -source scripts/sim.tcl -tclargs tb_ws_pe
#          vivado -mode batch -source scripts/sim.tcl -tclargs tb_ws_column ROWS=3
# Console: set argv tb_ws_pe; set argc 1; source scripts/sim.tcl
# Extra NAME=VALUE arguments override testbench parameters.
# Generate vectors first, e.g. python model/gen_pe_vectors.py
set root [file normalize [file join [file dirname [info script]] ..]]
if {![info exists argc] || $argc < 1} {error "Pass the testbench top name, e.g. tb_ws_pe"}
set tb [lindex $argv 0]
set vec [file join $root tb vectors]
set generics [lrange $argv 1 end]
set simdir [file join $root build ${tb}_[clock format [clock seconds] -format %Y%m%d_%H%M%S]]
create_project $tb $simdir -part xck26-sfvc784-2LV-c
add_files [glob [file join $root rtl *.sv]]
add_files -fileset sim_1 [file join $root tb ${tb}.sv]
set_property top $tb [get_filesets sim_1]
set_property -name xsim.simulate.xsim.more_options -value "-testplusarg {VEC_DIR=$vec}" -objects [get_filesets sim_1]
if {[llength $generics] > 0} {
    set gopts {}
    foreach g $generics { append gopts " -generic_top $g" }
    set_property -name xsim.elaborate.xelab.more_options -value $gopts -objects [get_filesets sim_1]
}
set_property xsim.simulate.custom_tcl [file join $root scripts run_all.tcl] [get_filesets sim_1]
update_compile_order -fileset sources_1
update_compile_order -fileset sim_1
launch_simulation
close_sim
set simlog [file join $simdir ${tb}.sim sim_1 behav xsim simulate.log]
set fh [open $simlog r]
set text [read $fh]
close $fh
close_project
if {[string first "PASS ${tb}:" $text] < 0 || [regexp -nocase {fatal:|error:} $text]} {
    error "Simulation did not pass. Inspect $simlog"
}
puts "Verified PASS for $tb ($simlog)"
