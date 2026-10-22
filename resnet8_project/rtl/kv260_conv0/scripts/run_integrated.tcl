# Create an isolated complete system and build it. Never programs a board.
set task_root [file normalize [file join [file dirname [info script]] ..]]
set workspace [file normalize [file join $task_root .. .. ..]]
# Keep generated IP paths shorter on Windows; all outputs remain in the workspace.
set integrated_build [file join $workspace .vivado kv260_[clock format [clock seconds] -format %Y%m%d_%H%M%S]]
set argc 2
set argv [list xilinx.com:kv260_som:part0:1.4 $integrated_build]
source [file join $task_root scripts create_project.tcl]
set handle [open [file join $task_root build last_integrated_project.txt] w]
puts $handle [file join $integrated_build conv0_kv260.xpr]
close $handle
source [file join $task_root scripts build.tcl]
