# Source in the EXISTING Vivado project's Tcl console. No build or simulation.
# Keep the project's ws_axi_top copy if already present; add the other files
# from the canonical repository directory, without duplicate module definitions.
set ws_axi_root [file normalize [file join [file dirname [info script]] ..]]
if {[current_project -quiet] eq ""} {error "Open the TPU Vivado project first."}
set ws_axi_names {
    ws_pkg.sv ws_axi_pkg.sv
    ws_pe.sv ws_column.sv ws_array.sv ws_skew.sv ws_mxu.sv
    ws_axi_regs.sv ws_axis_loader.sv ws_tile_engine.sv ws_axis_output.sv
    ws_axi_top.sv
}
foreach ws_axi_name $ws_axi_names {
    set ws_axi_matches {}
    foreach ws_axi_file [get_files -quiet -of_objects [get_filesets sources_1]] {
        if {[file tail $ws_axi_file] eq $ws_axi_name} {
            lappend ws_axi_matches $ws_axi_file
        }
    }
    if {[llength $ws_axi_matches] > 1} {
        error "Duplicate $ws_axi_name sources; select one canonical copy before continuing."
    }
    if {[llength $ws_axi_matches] == 0} {
        add_files -fileset sources_1 -norecurse [file join $ws_axi_root rtl $ws_axi_name]
    }
}
set_property top ws_axi_top [get_filesets sources_1]
update_compile_order -fileset sources_1
puts "AXI submodules added; sources_1 top is ws_axi_top. No synthesis or simulation launched."
