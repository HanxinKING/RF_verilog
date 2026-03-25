`timescale 1ns/1ps
`include "cnn1d_defs.vh"

// ============================================================
// cnn1d_weight_rom testbench
// ------------------------------------------------------------
// 验证目标：
// 1. layer_sel=0/1/2 时能访问卷积参数区
// 2. layer_sel=3 时能访问 FC 参数区
// ============================================================

module cnn1d_weight_rom_tb;

    reg  [1:0]                      layer_sel;
    reg  [10:0]                     weight_addr;
    reg  [7:0]                      bias_addr;
    wire signed [`CNN_DATA_W-1:0]  weight_data;
    wire signed [`CNN_ACC_W-1:0]   bias_data;

    cnn1d_weight_rom #(
        .NUM_CLASSES(4),
        .CONV_W_DEPTH(16),
        .CONV_B_DEPTH(8),
        .FC_W_DEPTH(16),
        .FC_B_DEPTH(4)
    ) dut (
        .layer_sel(layer_sel),
        .weight_addr(weight_addr),
        .bias_addr(bias_addr),
        .weight_data(weight_data),
        .bias_data(bias_data)
    );

    initial begin
        // 手动写入几组测试值
        dut.conv_w_mem[3] = 8'sd12;
        dut.conv_b_mem[2] = -32'sd5;
        dut.fc_w_mem[7]   = -8'sd9;
        dut.fc_b_mem[1]   = 32'sd22;

        // 读卷积区
        layer_sel   = 2'd0;
        weight_addr = 11'd3;
        bias_addr   = 8'd2;
        #1;
        if (weight_data !== 8'sd12 || bias_data !== -32'sd5) begin
            $display("TEST FAIL: cnn1d_weight_rom conv region mismatch");
            $finish;
        end

        // 读 FC 区
        layer_sel   = 2'd3;
        weight_addr = 11'd7;
        bias_addr   = 8'd1;
        #1;
        if (weight_data !== -8'sd9 || bias_data !== 32'sd22) begin
            $display("TEST FAIL: cnn1d_weight_rom fc region mismatch");
            $finish;
        end

        $display("TEST PASS: cnn1d_weight_rom");
        $finish;
    end

endmodule
