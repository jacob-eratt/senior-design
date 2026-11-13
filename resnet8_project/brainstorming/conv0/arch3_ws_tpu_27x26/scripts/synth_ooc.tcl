# Out-of-context synthesis of one block on the KV260 part at 200 MHz.
# Batch: vivado -mode batch -source scripts/synth_ooc.tcl -tclargs ws_mxu
#        vivado -mode batch -source scripts/synth_ooc.tcl -tclargs ws_mxu ROWS=8 COLS=5
#        vivado -mode batch -source scripts/synth_ooc.tcl -tclargs ws_mxu impl
# NAME=VALUE arguments override top-level parameters. "impl" also runs
# opt/place/route for a realistic Fmax (slow at 27x26).
set root [file normalize [file join [file dirname [info script]] ..]]
if {![info exists argc] || $argc < 1} {error "Pass the top module name, e.g. ws_mxu"}
set top [lindex $argv 0]
set generics {}
set impl 0
foreach a [lrange $argv 1 end] {
    if {$a eq "impl"} { set impl 1 } else { lappend generics -generic $a }
}
set outdir [file join $root build synth_${top}_[clock format [clock seconds] -format %Y%m%d_%H%M%S]]
file mkdir $outdir
# Packages must precede modules that reference their parameter defaults.
read_verilog -sv [file join $root rtl ws_pkg.sv]
read_verilog -sv [file join $root rtl ws_axi_pkg.sv]
foreach src [lsort [glob [file join $root rtl *.sv]]] {
    if {[file tail $src] ni {ws_pkg.sv ws_axi_pkg.sv}} {read_verilog -sv $src}
}
read_xdc -mode out_of_context [file join $root scripts ooc_200mhz.xdc]
synth_design -top $top -part xck26-sfvc784-2LV-c -mode out_of_context {*}$generics
report_utilization -file [file join $outdir util_synth.rpt]
report_utilization -hierarchical -hierarchical_depth 3 -file [file join $outdir util_hier_synth.rpt]
report_timing_summary -max_paths 10 -file [file join $outdir timing_synth.rpt]
check_timing -verbose -file [file join $outdir check_timing.rpt]
write_checkpoint -force [file join $outdir ${top}_synth.dcp]
# Preserve the exact source used, and inspect mapping instead of assuming
# DSP inference also implements the proposed dedicated cascades/registers.
file mkdir [file join $outdir source]
foreach src [glob [file join $root rtl *.sv]] {file copy $src [file join $outdir source [file tail $src]]}
file copy [info script] [file join $outdir source synth_ooc.tcl]
file copy [file join $root scripts ooc_200mhz.xdc] [file join $outdir source ooc_200mhz.xdc]
set dsps [get_cells -hier -filter {REF_NAME == DSP48E2}]
set df [open [file join $outdir dsp_mapping.csv] w]
puts $df "cell,AREG,BREG,MREG,PREG,ACASCREG,BCASCREG,A_INPUT,B_INPUT,USE_MULT"
foreach cell $dsps {
    set vals [list $cell]
    set props [list_property $cell]
    foreach prop {AREG BREG MREG PREG ACASCREG BCASCREG A_INPUT B_INPUT USE_MULT} {
        if {[lsearch -exact $props $prop] >= 0} {lappend vals [get_property $prop $cell]} else {lappend vals unknown}
    }
    puts $df [join $vals ,]
}
close $df
if {[llength $dsps]} {report_property -all [lindex $dsps 0] -file [file join $outdir dsp_example.rpt]}
set stage synth
if {$impl} {
    opt_design
    place_design
    route_design
    report_utilization -file [file join $outdir util_route.rpt]
    report_timing_summary -max_paths 10 -file [file join $outdir timing_route.rpt]
    set stage route
}
set dsp [llength [get_cells -hier -filter {REF_NAME =~ DSP48E2}]]
set ff  [llength [get_cells -hier -filter {PRIMITIVE_GROUP == REGISTER}]]
# Use the utilization report's combined CLB LUT count, not a nonexistent
# PRIMITIVE_GROUP=LUT category or uncombined LUT primitive count.
set util_file [file join $outdir util_${stage}.rpt]
set uf [open $util_file r]
set util_text [read $uf]
close $uf
if {![regexp {\|\s*CLB LUTs\*?\s*\|\s*([0-9]+)} $util_text -> lut]} {
    error "Cannot extract CLB LUT count from $util_file"
}
set srl [llength [get_cells -hier -filter {REF_NAME =~ SRL*}]]
set wns [get_property SLACK [get_timing_paths -max_paths 1 -setup]]
set fmax [expr {1000.0 / (5.0 - $wns)}]
set fh [open [file join $outdir summary.txt] w]
set msg [format "%s (%s) %s: DSP48E2=%d FF=%d CLB_LUT=%d SRL=%d WNS=%.3f ns estimated_internal_Fmax~%.0f MHz (OOC, not board timing)" \
    $top $generics $stage $dsp $ff $lut $srl $wns $fmax]
puts $fh $msg
close $fh
puts $msg
puts "Reports in $outdir"
