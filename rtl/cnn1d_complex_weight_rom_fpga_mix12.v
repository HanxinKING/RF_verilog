`timescale 1ns/1ps
`include "cnn1d_defs_fpga_mix12.vh"

// ============================================================
// 复值卷积参数 ROM
// ------------------------------------------------------------
// 这个模块用于统一存放“复值卷积层”的参数。
// 为了让后级模块一次就能取到一组复参数，这里把参数拆成 4 份并行存储：
// 1. 卷积权重实部   conv_w_real_mem
// 2. 卷积权重虚部   conv_w_imag_mem
// 3. 卷积偏置实部   conv_b_real_mem
// 4. 卷积偏置虚部   conv_b_imag_mem
//
// 地址组织方式和复卷积核一一对应：
//   weight_addr = base + oc * IN_COMPLEX_CH * KERNEL
//                       + ic * KERNEL
//                       + k
//
// 也就是说：
// - 同一个 weight_addr 同时对应某个复权重的实部/虚部
// - 同一个 bias_addr   同时对应某个输出复通道的偏置实部/虚部
//
// 这样做的好处是：
// - 后级卷积模块不需要分别计算实部地址和虚部地址
// - 实部/虚部天然对齐，不容易在训练参数导出时弄错
// - 很适合后续从 .mem 文件直接加载量化后的复值参数
// ============================================================

module cnn1d_complex_weight_rom_fpga_mix12 #(
    parameter integer CONV_W_DEPTH       = 1,
    parameter integer CONV_B_DEPTH       = 1,
    parameter integer LOAD_CONV_W_REAL   = 0,
    parameter integer LOAD_CONV_W_IMAG   = 0,
    parameter integer LOAD_CONV_B_REAL   = 0,
    parameter integer LOAD_CONV_B_IMAG   = 0,
    parameter CONV_W_REAL_FILE           = "",
    parameter CONV_W_IMAG_FILE           = "",
    parameter CONV_B_REAL_FILE           = "",
    parameter CONV_B_IMAG_FILE           = ""
) (
    input      [`CNN_WADDR_W-1:0]         weight_addr,
    input      [`CNN_BADDR_W-1:0]         bias_addr,
    output reg signed [`CNN_DATA_W-1:0]   weight_real_data,
    output reg signed [`CNN_DATA_W-1:0]   weight_imag_data,
    output reg signed [`CNN_ACC_W-1:0]    bias_real_data,
    output reg signed [`CNN_ACC_W-1:0]    bias_imag_data
);

    // 复权重 / 复偏置分别用实部和虚部两块存储器表示。
    reg signed [`CNN_DATA_W-1:0] conv_w_real_mem [0:CONV_W_DEPTH-1];
    reg signed [`CNN_DATA_W-1:0] conv_w_imag_mem [0:CONV_W_DEPTH-1];
    reg signed [`CNN_ACC_W-1:0]  conv_b_real_mem [0:CONV_B_DEPTH-1];
    reg signed [`CNN_ACC_W-1:0]  conv_b_imag_mem [0:CONV_B_DEPTH-1];

    integer idx;

    initial begin
        // 先整体清零，便于仿真时观察，也避免未初始化导致 X 传播。
        for (idx = 0; idx < CONV_W_DEPTH; idx = idx + 1) begin
            conv_w_real_mem[idx] = {`CNN_DATA_W{1'b0}};
            conv_w_imag_mem[idx] = {`CNN_DATA_W{1'b0}};
        end

        for (idx = 0; idx < CONV_B_DEPTH; idx = idx + 1) begin
            conv_b_real_mem[idx] = {`CNN_ACC_W{1'b0}};
            conv_b_imag_mem[idx] = {`CNN_ACC_W{1'b0}};
        end

        // 如果打开加载开关，则从外部文件把量化后的复参数读入 ROM。
        if (LOAD_CONV_W_REAL != 0) begin
            $display("INFO: loading complex CONV_W_REAL from %s", CONV_W_REAL_FILE);
            $readmemh(CONV_W_REAL_FILE, conv_w_real_mem);
        end

        if (LOAD_CONV_W_IMAG != 0) begin
            $display("INFO: loading complex CONV_W_IMAG from %s", CONV_W_IMAG_FILE);
            $readmemh(CONV_W_IMAG_FILE, conv_w_imag_mem);
        end

        if (LOAD_CONV_B_REAL != 0) begin
            $display("INFO: loading complex CONV_B_REAL from %s", CONV_B_REAL_FILE);
            $readmemh(CONV_B_REAL_FILE, conv_b_real_mem);
        end

        if (LOAD_CONV_B_IMAG != 0) begin
            $display("INFO: loading complex CONV_B_IMAG from %s", CONV_B_IMAG_FILE);
            $readmemh(CONV_B_IMAG_FILE, conv_b_imag_mem);
        end
    end

    // 组合读：地址一变化，当前复权重/复偏置立刻可见。
    always @(*) begin
        weight_real_data = conv_w_real_mem[weight_addr];
        weight_imag_data = conv_w_imag_mem[weight_addr];
        bias_real_data   = conv_b_real_mem[bias_addr];
        bias_imag_data   = conv_b_imag_mem[bias_addr];
    end

endmodule
