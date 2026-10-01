# Source this file in the Vivado PROJECT console with the previous simulation
# closed. Adds missing sources by reference and selects the NEW simulation top.
# Does not launch simulation or change the synthesis top.
# By default require/use the full Python fixture. To run synthetic tests only:
#   set resnet_serial_use_fixture 0
#   source {.../setup_xsim_serial.tcl}

set resnet_serial_dir [file normalize [file dirname [info script]]]
set resnet_serial_simset [get_filesets sim_1]
if {[llength $resnet_serial_simset] != 1} {
    error "Open a Vivado project containing sim_1 before sourcing this script."
}
if {![info exists resnet_serial_use_fixture]} {set resnet_serial_use_fixture 1}
set resnet_serial_artifacts [file normalize [file join $resnet_serial_dir ../resnet8/artifacts]]
if {![file exists [file join $resnet_serial_artifacts layer_vectors/conv0/expected_acc.mem]]} {
    set resnet_serial_artifacts [file normalize [file join $resnet_serial_dir ../fixtures]]
}
if {$resnet_serial_use_fixture} {
    foreach resnet_serial_fixture {
        layer_vectors/conv0/input.mem
        export/conv0_weights.mem
        export/conv0_bias.mem
        layer_vectors/conv0/expected_acc.mem
    } {
        if {![file exists [file join $resnet_serial_artifacts $resnet_serial_fixture]]} {
            error "Missing fixture $resnet_serial_fixture in $resnet_serial_artifacts. Generate artifacts or set resnet_serial_use_fixture 0 for synthetic-only tests."
        }
    }
}
foreach {resnet_serial_source resnet_serial_fileset} {
    conv0_serial_dsp.v sources_1
    tb_conv0_serial_dsp.sv sim_1
} {
    set resnet_serial_path [file join $resnet_serial_dir $resnet_serial_source]
    if {![file exists $resnet_serial_path]} {error "Missing source: $resnet_serial_path"}
    set resnet_serial_existing [get_files -quiet -of_objects [get_filesets $resnet_serial_fileset] *$resnet_serial_source]
    if {[llength $resnet_serial_existing] == 0} {
        add_files -fileset $resnet_serial_fileset -norecurse $resnet_serial_path
        set resnet_serial_existing [get_files -quiet -of_objects [get_filesets $resnet_serial_fileset] *$resnet_serial_source]
    }
    if {[llength $resnet_serial_existing] != 1} {
        error "Expected one $resnet_serial_source in $resnet_serial_fileset; remove duplicate source copies."
    }
    puts "Using source: $resnet_serial_existing"
    if {$resnet_serial_source eq "tb_conv0_serial_dsp.sv"} {
        set_property file_type SystemVerilog $resnet_serial_existing
    }
}

# Keep overrides supported by this testbench, dropping parameters belonging
# to the old parallel testbench when switching simulation tops.
set resnet_serial_generics {}
foreach resnet_serial_generic [get_property generic $resnet_serial_simset] {
    if {[string match "SEED=*" $resnet_serial_generic] || [string match "MAX_FRAME_CYCLES=*" $resnet_serial_generic]} {
        lappend resnet_serial_generics $resnet_serial_generic
    }
}
if {$resnet_serial_use_fixture} {
    lappend resnet_serial_generics "ARTIFACT_DIR=\"$resnet_serial_artifacts\""
} else {
    lappend resnet_serial_generics {ARTIFACT_DIR=""}
}
set_property generic $resnet_serial_generics $resnet_serial_simset
set_property top tb_conv0_serial_dsp $resnet_serial_simset
set_property xsim.simulate.log_all_signals false $resnet_serial_simset
set_property xsim.simulate.custom_tcl [file join $resnet_serial_dir xsim_serial_no_waves.tcl] $resnet_serial_simset
update_compile_order -fileset sim_1
puts "Configured tb_conv0_serial_dsp; Python fixture enabled: $resnet_serial_use_fixture"
puts "Now run: launch_simulation -simset sim_1 -mode behavioral"
