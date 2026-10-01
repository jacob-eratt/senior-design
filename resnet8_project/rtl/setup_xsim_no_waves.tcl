# Source in the Vivado PROJECT Tcl console after closing the stuck simulation.
# This configures the next launch; it does not start/stop a simulation itself.
set resnet_simset [get_filesets sim_1]
if {[llength $resnet_simset] != 1} {
    error "Expected an open project with simulation fileset sim_1."
}
set resnet_sim_script [file normalize [file join [file dirname [info script]] xsim_no_waves.tcl]]
if {![file exists $resnet_sim_script]} {
    error "Missing startup script: $resnet_sim_script"
}
set_property top tb_conv_1024 $resnet_simset
set_property xsim.simulate.log_all_signals false $resnet_simset
set_property xsim.simulate.custom_tcl $resnet_sim_script $resnet_simset
puts "Configured tb_conv_1024 to skip automatic waveform loading."
puts "Now run: launch_simulation -simset sim_1 -mode behavioral"
unset resnet_simset
unset resnet_sim_script
