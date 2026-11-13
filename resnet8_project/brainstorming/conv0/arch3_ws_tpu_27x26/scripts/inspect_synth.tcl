# Reopen a completed synthesis checkpoint; no synthesis or implementation.
# vivado -mode batch -source scripts/inspect_synth.tcl -tclargs build/synth_.../ws_mxu_synth.dcp
if {$argc != 1} {error "Pass one synthesis checkpoint path"}
set dcp [file normalize [lindex $argv 0]]
set outdir [file dirname $dcp]
open_checkpoint $dcp
set dsps [get_cells -hier -filter {REF_NAME == DSP48E2}]
set pc_links 0
set bc_links 0
set b_regs 0
set p_regs 0
foreach cell $dsps {
    if {[get_property BREG $cell] != 0} {incr b_regs}
    if {[get_property PREG $cell] != 0} {incr p_regs}
    foreach {input output kind} {PCIN PCOUT pc BCIN BCOUT bc} {
        set pin [get_pins -of_objects $cell -filter "REF_PIN_NAME == ${input}\[0\]"]
        set nets [get_nets -quiet -of_objects $pin]
        set drivers [get_pins -quiet -of_objects $nets -filter {DIRECTION == OUT}]
        foreach driver $drivers {
            if {[get_property REF_PIN_NAME $driver] eq "${output}\[0\]"} {
                if {$kind eq "pc"} {incr pc_links} else {incr bc_links}
            }
        }
    }
}
set uf [open [file join $outdir util_synth.rpt] r]
set util_text [read $uf]
close $uf
if {![regexp {\|\s*CLB LUTs\*?\s*\|\s*([0-9]+)} $util_text -> lut]} {error "Missing CLB LUT count"}
set ff [llength [get_cells -hier -filter {PRIMITIVE_GROUP == REGISTER}]]
set srl [llength [get_cells -hier -filter {REF_NAME =~ SRL*}]]
set wns [get_property SLACK [get_timing_paths -max_paths 1 -setup]]
set f [open [file join $outdir summary_corrected.txt] w]
set msg "ws_mxu synthesis: DSP48E2=[llength $dsps] CLB_LUT=$lut FF=$ff SRL=$srl WNS=$wns ns; OOC only, no routed Fmax established."
puts $f $msg
puts $f "DSPs with BREG>0=$b_regs; PREG>0=$p_regs; direct PCOUT[format %c 91]0[format %c 93] to PCIN links=$pc_links; direct BCOUT to BCIN links=$bc_links."
close $f
puts $msg
puts "MAPPING: BREG=$b_regs PREG=$p_regs PCASCADE=$pc_links BCASCADE=$bc_links"
