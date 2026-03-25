`timescale 1ns/1ps
`include "cnn1d_defs.vh"

// ============================================================
// cnn1d_complex_pool_core testbench
// ------------------------------------------------------------
// 验证目标：
// 1. 复值池化按模平方比较大小
// 2. 被选中的实部/虚部来自同一个采样点
// 3. 模平方相等时，优先保留窗口中的第 1 个点
// ============================================================

module cnn1d_complex_pool_core_tb;

    localparam integer TB_IN_LEN     = 4;
    localparam integer TB_COMPLEX_CH = 2;
    localparam integer TB_OUT_LEN    = TB_IN_LEN >> 1;
    localparam integer TB_FLAT_IN_CH = TB_COMPLEX_CH << 1;
    localparam integer TB_FLAT_OUT_CH = TB_COMPLEX_CH << 1;

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

    reg signed [`CNN_DATA_W-1:0] in_mem  [0:TB_FLAT_IN_CH*TB_IN_LEN-1];
    reg signed [`CNN_DATA_W-1:0] out_mem [0:TB_FLAT_OUT_CH*TB_OUT_LEN-1];

    integer i;

    assign feat_rd_data = in_mem[feat_rd_addr];

    cnn1d_complex_pool_core #(
        .IN_LEN(TB_IN_LEN),
        .COMPLEX_CH(TB_COMPLEX_CH)
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

    always @(posedge clk) begin
        if (feat_wr_en) begin
            out_mem[feat_wr_addr] <= feat_wr_data;
        end
    end

    initial begin
        rst_n = 1'b0;
        start = 1'b0;

        for (i = 0; i < TB_FLAT_OUT_CH*TB_OUT_LEN; i = i + 1) begin
            out_mem[i] = 0;
        end

        // ch0 real
        in_mem[0] = 8'sd1;
        in_mem[1] = 8'sd2;
        in_mem[2] = 8'sd3;
        in_mem[3] = 8'sd0;

        // ch0 imag
        in_mem[4] = 8'sd1;
        in_mem[5] = 8'sd0;
        in_mem[6] = 8'sd4;
        in_mem[7] = 8'sd5;

        // ch1 real
        in_mem[8]  = -8'sd1;
        in_mem[9]  =  8'sd0;
        in_mem[10] =  8'sd4;
        in_mem[11] =  8'sd2;

        // ch1 imag
        in_mem[12] = -8'sd1;
        in_mem[13] = -8'sd3;
        in_mem[14] =  8'sd1;
        in_mem[15] =  8'sd5;

        repeat (4) @(posedge clk);
        rst_n = 1'b1;

        @(posedge clk);
        start = 1'b1;
        @(posedge clk);
        start = 1'b0;

        repeat (200) begin
            @(posedge clk);
            if (done) begin
                if (out_mem[0] !==  8'sd2 ||
                    out_mem[1] !==  8'sd3 ||
                    out_mem[2] !==  8'sd0 ||
                    out_mem[3] !==  8'sd4 ||
                    out_mem[4] !==  8'sd0 ||
                    out_mem[5] !==  8'sd2 ||
                    out_mem[6] !== -8'sd3 ||
                    out_mem[7] !==  8'sd5) begin
                    $display("TEST FAIL: cnn1d_complex_pool_core output mismatch");
                    $finish;
                end

                $display("TEST PASS: cnn1d_complex_pool_core");
                $finish;
            end
        end

        $display("TEST FAIL: cnn1d_complex_pool_core timeout");
        $finish;
    end

endmodule
