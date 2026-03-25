@echo off
set MSIM=D:\modelsim2020.4\win64
cd /d %~dp0
if exist work rmdir /s /q work
"%MSIM%\vlib.exe" work || exit /b 1
"%MSIM%\vmap.exe" work work || exit /b 1
"%MSIM%\vlog.exe" -sv -work work -f modelsim_fpga_mix12_rtl.f || exit /b 1
"%MSIM%\vlog.exe" -sv -work work -f modelsim_fpga_mix12_tb_dump.f || exit /b 1
"%MSIM%\vsim.exe" -c -voptargs=+acc work.cnn1d_complex_classifier_deploy_fpga_mix12_tb -do "run -all; quit -f"
