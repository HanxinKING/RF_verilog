`timescale 1ns/1ps
`include "cnn1d_defs.vh"

// ============================================================
// cnn1d_complex_weight_rom testbench
// ------------------------------------------------------------
// 验证目标：
// 1. 同一组 weight_addr 能同时读出复权重实部/虚部
// 2. 同一组 bias_addr 能同时读出复偏置实部/虚部
// 3. 未写入的位置默认保持为 0
// ============================================================

module cnn1d_complex_weight_rom_tb;

    reg  [`CNN_WADDR_W-1:0]         weight_addr;
    reg  [`CNN_BADDR_W-1:0]         bias_addr;
    wire signed [`CNN_DATA_W-1:0]   weight_real_data;
    wire signed [`CNN_DATA_W-1:0]   weight_imag_data;
    wire signed [`CNN_ACC_W-1:0]    bias_real_data;
    wire signed [`CNN_ACC_W-1:0]    bias_imag_data;

    cnn1d_complex_weight_rom #(
        .CONV_W_DEPTH(16),
        .CONV_B_DEPTH(8)
    ) dut (
        .weight_addr(weight_addr),
        .bias_addr(bias_addr),
        .weight_real_data(weight_real_data),
        .weight_imag_data(weight_imag_data),
        .bias_real_data(bias_real_data),
        .bias_imag_data(bias_imag_data)
    );

    initial begin
        dut.conv_w_real_mem[3] = 8'sd12;
        dut.conv_w_imag_mem[3] = -8'sd7;
        dut.conv_b_real_mem[2] = 32'sd25;
        dut.conv_b_imag_mem[2] = -32'sd11;

        weight_addr = 0;
        bias_addr   = 0;
        #1;
        if (weight_real_data !== 0 ||
            weight_imag_data !== 0 ||
            bias_real_data   !== 0 ||
            bias_imag_data   !== 0) begin
            $display("TEST FAIL: cnn1d_complex_weight_rom default value mismatch");
            $finish;
        end

        weight_addr = 3;
        bias_addr   = 2;
        #1;
        if (weight_real_data !== 8'sd12 ||
            weight_imag_data !== -8'sd7 ||
            bias_real_data   !== 32'sd25 ||
            bias_imag_data   !== -32'sd11) begin
            $display("TEST FAIL: cnn1d_complex_weight_rom complex readback mismatch");
            $finish;
        end

        $display("TEST PASS: cnn1d_complex_weight_rom");
        $finish;
    end

endmodule
