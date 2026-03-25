transcript on
if {[file exists work]} {vdel -lib work -all}
vlib work
vmap work work
vlog -sv -work work -f modelsim_fpga_mix12_rtl.f
vlog -sv -work work -f modelsim_fpga_mix12_tb_dump.f
vsim -c -voptargs=+acc work.cnn1d_complex_classifier_deploy_fpga_mix12_tb
run -all
quit -f
