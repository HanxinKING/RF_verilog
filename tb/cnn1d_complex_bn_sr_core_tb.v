`timescale 1ns/1ps
`include "cnn1d_defs.vh"

// ============================================================
// cnn1d_complex_bn_sr_core testbench
// ------------------------------------------------------------
// 验证目标：
// 1. 复值 BN 仿射计算正确
// 2. SR 门控会对同一复通道的实部/虚部同步生效
// 3. |gate| <= SR_THRESH 时，整对输出清零
// ============================================================

module cnn1d_complex_bn_sr_core_tb;

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

    cnn1d_complex_bn_sr_core #(
        .FEAT_LEN(TB_FEAT_LEN),
        .COMPLEX_CH(TB_COMPLEX_CH),
        .SCALE_SHIFT(0),
        .GATE_SHIFT(1),
        .SR_THRESH(1)
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

        // ch0: active, output should equal affine result
        dut.scale_real_mem[0] =  8'sd2;
        dut.scale_imag_mem[0] = -8'sd1;
        dut.bias_real_mem[0]  = 32'sd1;
        dut.bias_imag_mem[0]  = 32'sd2;
        dut.gate_mem[0]       = 8'sd2;

        // ch1: gate_abs == SR_THRESH, whole complex channel should be suppressed
        dut.scale_real_mem[1] =  8'sd1;
        dut.scale_imag_mem[1] =  8'sd1;
        dut.bias_real_mem[1]  = 32'sd3;
        dut.bias_imag_mem[1]  = -32'sd4;
        dut.gate_mem[1]       = 8'sd1;

        repeat (4) @(posedge clk);
        rst_n = 1'b1;

        @(posedge clk);
        start = 1'b1;
        @(posedge clk);
        start = 1'b0;

        repeat (300) begin
            @(posedge clk);
            if (done) begin
                if (out_mem[0]  !==  8'sd3 ||
                    out_mem[1]  !==  8'sd5 ||
                    out_mem[2]  !== -8'sd1 ||
                    out_mem[3]  !== -8'sd1 ||
                    out_mem[4]  !==  8'sd4 ||
                    out_mem[5]  !==  8'sd2 ||
                    out_mem[6]  !==  8'sd0 ||
                    out_mem[7]  !==  8'sd0 ||
                    out_mem[8]  !==  8'sd0 ||
                    out_mem[9]  !==  8'sd0 ||
                    out_mem[10] !==  8'sd0 ||
                    out_mem[11] !==  8'sd0) begin
                    $display("TEST FAIL: cnn1d_complex_bn_sr_core output mismatch");
                    $finish;
                end

                $display("TEST PASS: cnn1d_complex_bn_sr_core");
                $finish;
            end
        end

        $display("TEST FAIL: cnn1d_complex_bn_sr_core timeout");
        $finish;
    end

endmodule
