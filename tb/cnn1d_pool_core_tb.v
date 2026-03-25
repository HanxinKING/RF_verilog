`timescale 1ns/1ps
`include "cnn1d_defs.vh"

// ============================================================
// cnn1d_pool_core testbench
// ------------------------------------------------------------
// 验证目标：
// 1. 池化窗口地址生成正确
// 2. 两通道数据都能正确做 max-pooling
// 3. done 能在全部输出写完后拉高
// ============================================================

module cnn1d_pool_core_tb;

    localparam integer TB_IN_LEN   = 8;
    localparam integer TB_CHANNELS = 2;
    localparam integer TB_OUT_LEN  = TB_IN_LEN >> 1;

    reg                              clk;
    reg                              rst_n;
    reg                              start;
    wire [`CNN_ADDR_W-1:0]           feat_rd_addr;
    wire signed [`CNN_DATA_W-1:0]    feat_rd_data;
    wire                             feat_wr_en;
    wire [`CNN_ADDR_W-1:0]           feat_wr_addr;
    wire signed [`CNN_DATA_W-1:0]    feat_wr_data;
    wire                             busy;
    wire                             done;

    reg signed [`CNN_DATA_W-1:0] in_mem  [0:TB_CHANNELS*TB_IN_LEN-1];
    reg signed [`CNN_DATA_W-1:0] out_mem [0:TB_CHANNELS*TB_OUT_LEN-1];

    integer i;

    // 用组合读模拟输入特征 RAM
    assign feat_rd_data = in_mem[feat_rd_addr];

    cnn1d_pool_core #(
        .IN_LEN(TB_IN_LEN),
        .CHANNELS(TB_CHANNELS)
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
        .busy(busy),
        .done(done)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    // 捕获模块写出的池化结果
    always @(posedge clk) begin
        if (feat_wr_en) begin
            out_mem[feat_wr_addr] <= feat_wr_data;
        end
    end

    initial begin
        rst_n = 1'b0;
        start = 1'b0;

        for (i = 0; i < TB_CHANNELS*TB_OUT_LEN; i = i + 1) begin
            out_mem[i] = 0;
        end

        // 通道 0 输入: [1,4,2,3,8,6,5,7] -> [4,3,8,7]
        in_mem[0] =  8'sd1;
        in_mem[1] =  8'sd4;
        in_mem[2] =  8'sd2;
        in_mem[3] =  8'sd3;
        in_mem[4] =  8'sd8;
        in_mem[5] =  8'sd6;
        in_mem[6] =  8'sd5;
        in_mem[7] =  8'sd7;

        // 通道 1 输入: [9,0,-1,-2,6,6,1,2] -> [9,-1,6,2]
        in_mem[8]  =  8'sd9;
        in_mem[9]  =  8'sd0;
        in_mem[10] = -8'sd1;
        in_mem[11] = -8'sd2;
        in_mem[12] =  8'sd6;
        in_mem[13] =  8'sd6;
        in_mem[14] =  8'sd1;
        in_mem[15] =  8'sd2;

        repeat (4) @(posedge clk);
        rst_n = 1'b1;

        @(posedge clk);
        start = 1'b1;
        @(posedge clk);
        start = 1'b0;

        repeat (200) begin
            @(posedge clk);
            if (done) begin
                if (out_mem[0] !== 8'sd4 ||
                    out_mem[1] !== 8'sd3 ||
                    out_mem[2] !== 8'sd8 ||
                    out_mem[3] !== 8'sd7 ||
                    out_mem[4] !== 8'sd9 ||
                    out_mem[5] !== -8'sd1 ||
                    out_mem[6] !== 8'sd6 ||
                    out_mem[7] !== 8'sd2) begin
                    $display("TEST FAIL: cnn1d_pool_core output mismatch");
                    $finish;
                end

                $display("TEST PASS: cnn1d_pool_core");
                $finish;
            end
        end

        $display("TEST FAIL: cnn1d_pool_core timeout");
        $finish;
    end

endmodule
