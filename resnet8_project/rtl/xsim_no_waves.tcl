# XSim startup script. Select using xsim.simulate.custom_tcl in the PROJECT
# console; do not source this script until the simulation snapshot is loaded.
# Deliberately omit add_wave and log_wave, especially recursive/all-signal forms.
# The SystemVerilog testbench compares every output without waveform logging.
puts "Running tb_conv_1024 without automatic waveform loading. Watch for PASS or FAIL."
run all
