`timescale 1ns/1ps
`include "cnn1d_defs.vh"

// ============================================================
// cnn1d_conv_layer testbench
// ------------------------------------------------------------
// 验证目标：
// 1. 卷积地址生成正确
// 2. 单输出通道和多输出通道的顺序计算正确
// 3. bias、ReLU、输出写回地址都正确
//
// 测试配置：
// - IN_LEN  = 5
// - IN_CH   = 1
// - OUT_CH  = 2
// - KERNEL  = 3
// - ACT_SHIFT = 0
//
// 输入：
//   x = [1, 2, 3, 4, 5]
//
// 卷积核 0：
//   w0 = [1, 1, 1], bias0 = 0
//   输出 = [6, 9, 12]
//
// 卷积核 1：
//   w1 = [1, 0, -1], bias1 = 5
//   原始结果 = [3, 3, 3]
//   ReLU 后仍为 [3, 3, 3]
//
// 最终展平输出：
//   [6, 9, 12, 3, 3, 3]
// ============================================================

module cnn1d_conv_layer_tb;

    localparam integer TB_IN_LEN    = 5;
    localparam integer TB_IN_CH     = 1;
    localparam integer TB_OUT_CH    = 2;
    localparam integer TB_KERNEL    = 3;
    localparam integer TB_ACT_SHIFT = 0;
    localparam integer TB_OUT_LEN   = TB_IN_LEN - TB_KERNEL + 1;

    reg                              clk;
    reg                              rst_n;
    reg                              start;
    wire [`CNN_ADDR_W-1:0]           feat_rd_addr;
    wire signed [`CNN_DATA_W-1:0]    feat_rd_data;
    wire                             feat_wr_en;
    wire [`CNN_ADDR_W-1:0]           feat_wr_addr;
    wire signed [`CNN_DATA_W-1:0]    feat_wr_data;
    wire [10:0]                      weight_addr;
    wire [7:0]                       bias_addr;
    wire signed [`CNN_DATA_W-1:0]    weight_data;
    wire signed [`CNN_ACC_W-1:0]     bias_data;
    wire                             busy;
    wire                             done;

    reg signed [`CNN_DATA_W-1:0] in_mem     [0:TB_IN_LEN*TB_IN_CH-1];
    reg signed [`CNN_DATA_W-1:0] weight_mem [0:TB_OUT_CH*TB_IN_CH*TB_KERNEL-1];
    reg signed [`CNN_ACC_W-1:0]  bias_mem   [0:TB_OUT_CH-1];
    reg signed [`CNN_DATA_W-1:0] out_mem    [0:TB_OUT_CH*TB_OUT_LEN-1];

    integer i;

    assign feat_rd_data = in_mem[feat_rd_addr];
    assign weight_data  = weight_mem[weight_addr];
    assign bias_data    = bias_mem[bias_addr];

    cnn1d_conv_layer #(
        .IN_LEN(TB_IN_LEN),
        .IN_CH(TB_IN_CH),
        .OUT_CH(TB_OUT_CH),
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
        .weight_data(weight_data),
        .bias_data(bias_data),
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

        // 输入 x = [1,2,3,4,5]
        in_mem[0] = 8'sd1;
        in_mem[1] = 8'sd2;
        in_mem[2] = 8'sd3;
        in_mem[3] = 8'sd4;
        in_mem[4] = 8'sd5;

        // 卷积核 0: [1,1,1]
        weight_mem[0] =  8'sd1;
        weight_mem[1] =  8'sd1;
        weight_mem[2] =  8'sd1;

        // 卷积核 1: [1,0,-1]
        weight_mem[3] =  8'sd1;
        weight_mem[4] =  8'sd0;
        weight_mem[5] = -8'sd1;

        // bias
        bias_mem[0] = 32'sd0;
        bias_mem[1] = 32'sd5;

        for (i = 0; i < TB_OUT_CH*TB_OUT_LEN; i = i + 1) begin
            out_mem[i] = 0;
        end

        repeat (4) @(posedge clk);
        rst_n = 1'b1;

        @(posedge clk);
        start = 1'b1;
        @(posedge clk);
        start = 1'b0;

        repeat (200) begin
            @(posedge clk);
            if (done) begin
                if (out_mem[0] !== 8'sd6  ||
                    out_mem[1] !== 8'sd9  ||
                    out_mem[2] !== 8'sd12 ||
                    out_mem[3] !== 8'sd3  ||
                    out_mem[4] !== 8'sd3  ||
                    out_mem[5] !== 8'sd3) begin
                    $display("TEST FAIL: cnn1d_conv_layer output mismatch");
                    $finish;
                end

                $display("TEST PASS: cnn1d_conv_layer");
                $finish;
            end
        end

        $display("TEST FAIL: cnn1d_conv_layer timeout");
    
    end

endmodule
