# Source in the Vivado PROJECT Tcl console, with any previous simulation closed.
# Adds missing source files, selects the 416-MAC TB, enables the Python fixture,
# and sets a run-all startup script without automatic waveform loading.
# Does not launch a run or change the synthesis top.
# For synthetic-only testing set resnet_416_use_fixture 0 before sourcing.
set resnet_416_dir [file normalize [file dirname [info script]]]
set resnet_416_simset [get_filesets sim_1]
if {[llength $resnet_416_simset] != 1} {
    error "Open a project containing simulation fileset sim_1 first."
}
if {![info exists resnet_416_use_fixture]} {set resnet_416_use_fixture 1}
set resnet_416_artifacts [file normalize [file join $resnet_416_dir ../resnet8/artifacts]]
if {![file exists [file join $resnet_416_artifacts layer_vectors/conv0/expected_acc.mem]]} {
    set resnet_416_artifacts [file normalize [file join $resnet_416_dir ../fixtures]]
}
if {$resnet_416_use_fixture} {
    foreach resnet_416_fixture {
        layer_vectors/conv0/input.mem
        export/conv0_weights.mem
        export/conv0_bias.mem
        layer_vectors/conv0/expected_acc.mem
    } {
        if {![file exists [file join $resnet_416_artifacts $resnet_416_fixture]]} {
            error "Missing $resnet_416_fixture in $resnet_416_artifacts. Generate artifacts, or set resnet_416_use_fixture 0 for synthetic-only tests."
        }
    }
}
foreach {resnet_416_source resnet_416_fileset} {
    conv0_416_dsp.v sources_1
    tb_conv0_416_dsp.sv sim_1
} {
    set resnet_416_path [file join $resnet_416_dir $resnet_416_source]
    if {![file exists $resnet_416_path]} {error "Missing source: $resnet_416_path"}
    set resnet_416_existing [get_files -quiet -of_objects [get_filesets $resnet_416_fileset] *$resnet_416_source]
    if {[llength $resnet_416_existing] == 0} {
        add_files -fileset $resnet_416_fileset -norecurse $resnet_416_path
        set resnet_416_existing [get_files -quiet -of_objects [get_filesets $resnet_416_fileset] *$resnet_416_source]
    }
    if {[llength $resnet_416_existing] != 1} {
        error "Expected one $resnet_416_source in $resnet_416_fileset; remove duplicate source copies."
    }
    puts "Using source: $resnet_416_existing"
    if {$resnet_416_source eq "tb_conv0_416_dsp.sv"} {
        set_property file_type SystemVerilog $resnet_416_existing
    }
}
# Select known defaults instead of carrying parameters from an old testbench.
set resnet_416_generics {OUTPUT_LANES=4}
if {$resnet_416_use_fixture} {
    lappend resnet_416_generics "ARTIFACT_DIR=\"$resnet_416_artifacts\""
} else {
    lappend resnet_416_generics {ARTIFACT_DIR=""}
}
set_property generic $resnet_416_generics $resnet_416_simset
set_property top tb_conv0_416_dsp $resnet_416_simset
set_property xsim.simulate.log_all_signals false $resnet_416_simset
set_property xsim.simulate.custom_tcl [file join $resnet_416_dir xsim_416_no_waves.tcl] $resnet_416_simset
update_compile_order -fileset sim_1
puts "Configured tb_conv0_416_dsp, OUTPUT_LANES=4, Python fixture=$resnet_416_use_fixture"
puts "Now run: launch_simulation -simset sim_1 -mode behavioral"
