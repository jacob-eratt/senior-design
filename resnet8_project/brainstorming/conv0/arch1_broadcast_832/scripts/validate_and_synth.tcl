# Authorized validation run: baseline, all prepared fixtures, then component synthesis.
# Reuses one compiled simulation snapshot. No board programming or bitstream export.
set task_root [file normalize [file join [file dirname [info script]] ..]]
set argc 0
set argv {}
source [file join $task_root scripts simulate.tcl]
set evidence_dir [file join $task_root build validation_[clock format [clock seconds] -format %Y%m%d_%H%M%S]]
file mkdir $evidence_dir
file copy $simlog [file join $evidence_dir baseline.log]
set cases [lsort [glob -types d [file join $task_root build validation_vectors *]]]
if {[llength $cases]!=14} {error "Expected 14 prepared fixtures; run tools/prepare_validation_vectors.py"}
foreach case_dir $cases {
    set case_name [file tail $case_dir]
    puts "VALIDATION CASE: $case_name"
    set_property -name xsim.simulate.xsim.more_options -value "-testplusarg {FIXTURE_DIR=$case_dir}" -objects [get_filesets sim_1]
    launch_simulation -step simulate
    close_sim
    set handle [open $simlog r]
    set contents [read $handle]
    close $handle
    file copy $simlog [file join $evidence_dir ${case_name}.log]
    if {[string first "PASS tb_conv0_axi:" $contents]<0 || [regexp -nocase {fatal:|error:} $contents]} {
        error "Failed fixture $case_name; preserved log in $evidence_dir"
    }
}
puts "PASS VALIDATION SUITE: baseline + 14 cases, each cold/warm. Logs: $evidence_dir"
close_project
foreach component {mac_int8_mixed conv0_requant conv0_result_fifo conv0_832_core} {
    set argc 1
    set argv [list $component]
    source [file join $task_root scripts synth_components.tcl]
    close_project
}
puts "COMPLETED validation and component synthesis. Integrated fit/timing still require implementation."
