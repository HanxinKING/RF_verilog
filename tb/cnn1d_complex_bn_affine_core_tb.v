`timescale 1ns/1ps
`include "cnn1d_defs.vh"

// ============================================================
// cnn1d_complex_bn_affine_core testbench
// ------------------------------------------------------------
// 验证目标：
// 1. 复值 2x2 affine 计算正确
// 2. 输入输出地址布局正确
// 3. 与门控无关的 BN-only 路径能单独验证
// ============================================================

module cnn1d_complex_bn_affine_core_tb;

    localparam integer TB_FEAT_LEN   = 3;
    localparam integer TB_COMPLEX_CH = 2;
    localparam integer TB_FLAT_CH    = TB_COMPLEX_CH << 1;

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

    reg signed [`CNN_DATA_W-1:0] in_mem  [0:TB_FLAT_CH*TB_FEAT_LEN-1];
    reg signed [`CNN_DATA_W-1:0] out_mem [0:TB_FLAT_CH*TB_FEAT_LEN-1];

    integer i;

    assign feat_rd_data = in_mem[feat_rd_addr];

    cnn1d_complex_bn_affine_core #(
        .FEAT_LEN(TB_FEAT_LEN),
        .COMPLEX_CH(TB_COMPLEX_CH),
        .AFFINE_SHIFT(0)
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

        // ch0 real
        in_mem[0] =  8'sd1;
        in_mem[1] =  8'sd2;
        in_mem[2] = -8'sd1;

        // ch0 imag
        in_mem[3] =  8'sd3;
        in_mem[4] = -8'sd2;
        in_mem[5] =  8'sd0;

        // ch1 real
        in_mem[6] =  8'sd2;
        in_mem[7] = -8'sd3;
        in_mem[8] =  8'sd1;

        // ch1 imag
        in_mem[9]  =  8'sd1;
        in_mem[10] =  8'sd1;
        in_mem[11] = -8'sd2;

        for (i = 0; i < TB_FLAT_CH*TB_FEAT_LEN; i = i + 1) begin
            out_mem[i] = 0;
        end

        // ch0:
        // y_r = 2*x_r - 1*x_i + 1
        // y_i = 1*x_r + 1*x_i - 2
        dut.a_rr_mem[0] = 32'sd2;
        dut.a_ri_mem[0] = -32'sd1;
        dut.a_ir_mem[0] = 32'sd1;
        dut.a_ii_mem[0] = 32'sd1;
        dut.bias_real_mem[0] = 32'sd1;
        dut.bias_imag_mem[0] = -32'sd2;

        // ch1:
        // identity + bias
        dut.a_rr_mem[1] = 32'sd1;
        dut.a_ri_mem[1] = 32'sd0;
        dut.a_ir_mem[1] = 32'sd0;
        dut.a_ii_mem[1] = 32'sd1;
        dut.bias_real_mem[1] = -32'sd1;
        dut.bias_imag_mem[1] = 32'sd2;

        repeat (4) @(posedge clk);
        rst_n = 1'b1;

        @(posedge clk);
        start = 1'b1;
        @(posedge clk);
        start = 1'b0;

        repeat (300) begin
            @(posedge clk);
            if (done) begin
                if (out_mem[0]  !== 8'sd0  ||
                    out_mem[1]  !== 8'sd7  ||
                    out_mem[2]  !== -8'sd1 ||
                    out_mem[3]  !== 8'sd2  ||
                    out_mem[4]  !== -8'sd2 ||
                    out_mem[5]  !== -8'sd3 ||
                    out_mem[6]  !== 8'sd1  ||
                    out_mem[7]  !== -8'sd4 ||
                    out_mem[8]  !== 8'sd0  ||
                    out_mem[9]  !== 8'sd3  ||
                    out_mem[10] !== 8'sd3  ||
                    out_mem[11] !== 8'sd0) begin
                    $display("TEST FAIL: cnn1d_complex_bn_affine_core output mismatch");
                    $finish;
                end

                $display("TEST PASS: cnn1d_complex_bn_affine_core");
                $finish;
            end
        end

        $display("TEST FAIL: cnn1d_complex_bn_affine_core timeout");
        $finish;
    end

endmodule
