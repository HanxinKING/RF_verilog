`timescale 1ns/1ps
`include "cnn1d_defs.vh"

// ============================================================
// cnn1d_complex_conv_same_layer testbench
// ------------------------------------------------------------
// 验证目标：
// 1. 复值 SAME 卷积公式正确
// 2. 边界点会按 zero-padding 处理
// 3. 实部/虚部输出地址按相邻通道展开
//
// 测试配置：
// - IN_LEN         = 4
// - IN_COMPLEX_CH  = 1
// - OUT_COMPLEX_CH = 2
// - KERNEL         = 3
// - ACT_SHIFT      = 0
//
// 输入复序列：
//   x = [(1+j2), (-1+j3), (4+j0), (2-j1)]
//
// 输出通道 0：
//   仅保留中心 tap，权重 = 1 + j0，bias = 0
//   输出应与输入一致
//
// 输出通道 1：
//   仅保留左侧 tap，权重 = 1 + j0，bias = 1 - j1
//   same padding 下：
//   y = [(1-j1), (2+j1), (0+j2), (5-j1)]
// ============================================================

module cnn1d_complex_conv_same_layer_tb;

    localparam integer TB_IN_LEN         = 4;
    localparam integer TB_IN_COMPLEX_CH  = 1;
    localparam integer TB_OUT_COMPLEX_CH = 2;
    localparam integer TB_KERNEL         = 3;
    localparam integer TB_ACT_SHIFT      = 0;
    localparam integer TB_OUT_FLAT_CH    = TB_OUT_COMPLEX_CH << 1;

    reg                              clk;
    reg                              rst_n;
    reg                              start;
    wire [`CNN_ADDR_W-1:0]           feat_rd_addr;
    wire signed [`CNN_DATA_W-1:0]    feat_rd_data;
    wire                             feat_wr_en;
    wire [`CNN_ADDR_W-1:0]           feat_wr_addr;
    wire signed [`CNN_DATA_W-1:0]    feat_wr_data;
    wire [`CNN_WADDR_W-1:0]          weight_addr;
    wire [`CNN_BADDR_W-1:0]          bias_addr;
    wire signed [`CNN_DATA_W-1:0]    weight_real_data;
    wire signed [`CNN_DATA_W-1:0]    weight_imag_data;
    wire signed [`CNN_ACC_W-1:0]     bias_real_data;
    wire signed [`CNN_ACC_W-1:0]     bias_imag_data;
    wire                             busy;
    wire                             done;

    reg signed [`CNN_DATA_W-1:0] in_mem          [0:(TB_IN_COMPLEX_CH<<1)*TB_IN_LEN-1];
    reg signed [`CNN_DATA_W-1:0] weight_real_mem [0:TB_OUT_COMPLEX_CH*TB_IN_COMPLEX_CH*TB_KERNEL-1];
    reg signed [`CNN_DATA_W-1:0] weight_imag_mem [0:TB_OUT_COMPLEX_CH*TB_IN_COMPLEX_CH*TB_KERNEL-1];
    reg signed [`CNN_ACC_W-1:0]  bias_real_mem   [0:TB_OUT_COMPLEX_CH-1];
    reg signed [`CNN_ACC_W-1:0]  bias_imag_mem   [0:TB_OUT_COMPLEX_CH-1];
    reg signed [`CNN_DATA_W-1:0] out_mem         [0:TB_OUT_FLAT_CH*TB_IN_LEN-1];

    integer i;

    assign feat_rd_data      = in_mem[feat_rd_addr];
    assign weight_real_data  = weight_real_mem[weight_addr];
    assign weight_imag_data  = weight_imag_mem[weight_addr];
    assign bias_real_data    = bias_real_mem[bias_addr];
    assign bias_imag_data    = bias_imag_mem[bias_addr];

    cnn1d_complex_conv_same_layer #(
        .IN_LEN(TB_IN_LEN),
        .IN_COMPLEX_CH(TB_IN_COMPLEX_CH),
        .OUT_COMPLEX_CH(TB_OUT_COMPLEX_CH),
        .KERNEL(TB_KERNEL),
        .ACT_SHIFT(TB_ACT_SHIFT),
        .W_BASE(0),
        .B_BASE(0)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .in_base({`CNN_ADDR_W{1'b0}}),
        .out_base({`CNN_ADDR_W{1'b0}}),
        .feat_rd_addr(feat_rd_addr),
        .feat_rd_data(feat_rd_data),
        .feat_wr_en(feat_wr_en),
        .feat_wr_addr(feat_wr_addr),
        .feat_wr_data(feat_wr_data),
        .weight_addr(weight_addr),
        .bias_addr(bias_addr),
        .weight_real_data(weight_real_data),
        .weight_imag_data(weight_imag_data),
        .bias_real_data(bias_real_data),
        .bias_imag_data(bias_imag_data),
        .busy(busy),
        .done(done)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    always @(posedge clk) begin
        if (feat_wr_en) begin
            out_mem[feat_wr_addr] <= feat_wr_data;
        end
    end

    initial begin
        rst_n = 1'b0;
        start = 1'b0;

        // real(x)
        in_mem[0] =  8'sd1;
        in_mem[1] = -8'sd1;
        in_mem[2] =  8'sd4;
        in_mem[3] =  8'sd2;

        // imag(x)
        in_mem[4] =  8'sd2;
        in_mem[5] =  8'sd3;
        in_mem[6] =  8'sd0;
        in_mem[7] = -8'sd1;

        for (i = 0; i < TB_OUT_COMPLEX_CH*TB_IN_COMPLEX_CH*TB_KERNEL; i = i + 1) begin
            weight_real_mem[i] = 0;
            weight_imag_mem[i] = 0;
        end

        for (i = 0; i < TB_OUT_COMPLEX_CH; i = i + 1) begin
            bias_real_mem[i] = 0;
            bias_imag_mem[i] = 0;
        end

        for (i = 0; i < TB_OUT_FLAT_CH*TB_IN_LEN; i = i + 1) begin
            out_mem[i] = 0;
        end

        // oc0: center tap = 1 + j0
        weight_real_mem[1] = 8'sd1;

        // oc1: left tap = 1 + j0, bias = 1 - j1
        weight_real_mem[3] = 8'sd1;
        bias_real_mem[1]   = 32'sd1;
        bias_imag_mem[1]   = -32'sd1;

        repeat (4) @(posedge clk);
        rst_n = 1'b1;

        @(posedge clk);
        start = 1'b1;
        @(posedge clk);
        start = 1'b0;

        repeat (300) begin
            @(posedge clk);
            if (done) begin
                if (out_mem[0]  !==  8'sd1 ||
                    out_mem[1]  !== -8'sd1 ||
                    out_mem[2]  !==  8'sd4 ||
                    out_mem[3]  !==  8'sd2 ||
                    out_mem[4]  !==  8'sd2 ||
                    out_mem[5]  !==  8'sd3 ||
                    out_mem[6]  !==  8'sd0 ||
                    out_mem[7]  !== -8'sd1 ||
                    out_mem[8]  !==  8'sd1 ||
                    out_mem[9]  !==  8'sd2 ||
                    out_mem[10] !==  8'sd0 ||
                    out_mem[11] !==  8'sd5 ||
                    out_mem[12] !== -8'sd1 ||
                    out_mem[13] !==  8'sd1 ||
                    out_mem[14] !==  8'sd2 ||
                    out_mem[15] !== -8'sd1) begin
                    $display("TEST FAIL: cnn1d_complex_conv_same_layer output mismatch");
                    $finish;
                end

                $display("TEST PASS: cnn1d_complex_conv_same_layer");
                $finish;
            end
        end

        $display("TEST FAIL: cnn1d_complex_conv_same_layer timeout");
        $finish;
    end

endmodule
