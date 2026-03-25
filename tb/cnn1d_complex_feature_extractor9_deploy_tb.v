`timescale 1ns/1ps
`include "cnn1d_defs.vh"

// ============================================================
// 未剪枝 64 通道部署链路 testbench
// ------------------------------------------------------------
// 这份 testbench 用于验证：
// 1. Python 导出的未剪枝 64 通道参数能够被 Verilog 正确加载
// 2. 量化后的输入 .mem 能够按 FPGA 地址布局正确写入
// 3. 9 层复值特征提取部署顶层的输出，和 Python 侧 fixed-point
//    golden 结果逐点一致
//
// 运行前需要先执行：
//   python tools/slimsei64_flow.py make-demo ...
//
// 默认读取目录：
//   mem/slimsei64_demo/
// ============================================================

module cnn1d_complex_feature_extractor9_deploy_tb;

    localparam integer TB_INPUT_LEN        = 1000;
    localparam integer TB_INPUT_COMPLEX_CH = 1;
    localparam integer TB_STAGE_COMPLEX_CH = 64;
    localparam integer TB_KERNEL           = 3;
    localparam integer TB_INPUT_FLAT_LEN   = TB_INPUT_LEN * 2;
    localparam integer TB_OUTPUT_FLAT_LEN  = TB_STAGE_COMPLEX_CH * 2;

    localparam integer TB_INPUT_SHIFT      = 6;
    localparam integer TB_ACT_SHIFT        = 6;
    localparam integer TB_BN_SHIFT         = 10;
    localparam integer TB_SR_SHIFT         = 0;
    localparam integer TB_SR_THRESH        = 0;

    reg                              clk;
    reg                              rst_n;
    reg                              start;
    reg                              load_en;
    reg  [`CNN_ADDR_W-1:0]           load_addr;
    reg  signed [`CNN_DATA_W-1:0]    load_data;
    reg  [`CNN_ADDR_W-1:0]           feat_out_addr;
    wire signed [`CNN_DATA_W-1:0]    feat_out_data;
    wire                             busy;
    wire                             done;

    reg signed [`CNN_DATA_W-1:0] input_mem   [0:TB_INPUT_FLAT_LEN-1];
    reg signed [`CNN_DATA_W-1:0] expect_mem  [0:TB_OUTPUT_FLAT_LEN-1];

    integer i;
    reg got_done;

    cnn1d_complex_feature_extractor9_deploy_top #(
        .INPUT_LEN(TB_INPUT_LEN),
        .INPUT_COMPLEX_CH(TB_INPUT_COMPLEX_CH),
        .STAGE_COMPLEX_CH(TB_STAGE_COMPLEX_CH),
        .KERNEL(TB_KERNEL),
        .ACT_SHIFT(TB_ACT_SHIFT),
        .BN_SHIFT(TB_BN_SHIFT),
        .SR_SHIFT(TB_SR_SHIFT),
        .SR_THRESH(TB_SR_THRESH),
        .RAM_DEPTH(140000),
        .LOAD_CONV_W_REAL(1),
        .LOAD_CONV_W_IMAG(1),
        .LOAD_CONV_B_REAL(1),
        .LOAD_CONV_B_IMAG(1),
        .CONV_W_REAL_FILE("mem/slimsei64_demo/conv_w_real.mem"),
        .CONV_W_IMAG_FILE("mem/slimsei64_demo/conv_w_imag.mem"),
        .CONV_B_REAL_FILE("mem/slimsei64_demo/conv_b_real.mem"),
        .CONV_B_IMAG_FILE("mem/slimsei64_demo/conv_b_imag.mem")
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .load_en(load_en),
        .load_addr(load_addr),
        .load_data(load_data),
        .feat_out_addr(feat_out_addr),
        .feat_out_data(feat_out_data),
        .busy(busy),
        .done(done)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    task load_input_mem;
        begin
            load_en = 1'b1;
            for (i = 0; i < TB_INPUT_FLAT_LEN; i = i + 1) begin
                @(posedge clk);
                load_addr = i[`CNN_ADDR_W-1:0];
                load_data = input_mem[i];
            end
            @(posedge clk);
            load_en = 1'b0;
            load_addr = {`CNN_ADDR_W{1'b0}};
            load_data = {`CNN_DATA_W{1'b0}};
        end
    endtask

    task start_once;
        begin
            @(posedge clk);
            start = 1'b1;
            @(posedge clk);
            start = 1'b0;
        end
    endtask

    task check_output;
        begin
            for (i = 0; i < TB_OUTPUT_FLAT_LEN; i = i + 1) begin
                feat_out_addr = i[`CNN_ADDR_W-1:0];
                @(posedge clk);
                if (feat_out_data !== expect_mem[i]) begin
                    $display("TEST FAIL: output mismatch at addr=%0d got=%0d expect=%0d", i, feat_out_data, expect_mem[i]);
                    $finish;
                end
            end
        end
    endtask

    initial begin
        rst_n = 1'b0;
        start = 1'b0;
        load_en = 1'b0;
        load_addr = {`CNN_ADDR_W{1'b0}};
        load_data = {`CNN_DATA_W{1'b0}};
        feat_out_addr = {`CNN_ADDR_W{1'b0}};
        got_done = 1'b0;

        $readmemh("mem/slimsei64_demo/input_iq.mem", input_mem);
        $readmemh("mem/slimsei64_demo/expected_feature.mem", expect_mem);

        // 复 BN 折叠参数与 SR gate 直接写入各层内部存储器。
        $readmemh("mem/slimsei64_demo/bn1_a_rr.mem", dut.u_bn_sr1.a_rr_mem);
        $readmemh("mem/slimsei64_demo/bn1_a_ri.mem", dut.u_bn_sr1.a_ri_mem);
        $readmemh("mem/slimsei64_demo/bn1_a_ir.mem", dut.u_bn_sr1.a_ir_mem);
        $readmemh("mem/slimsei64_demo/bn1_a_ii.mem", dut.u_bn_sr1.a_ii_mem);
        $readmemh("mem/slimsei64_demo/bn1_b_real.mem", dut.u_bn_sr1.bias_real_mem);
        $readmemh("mem/slimsei64_demo/bn1_b_imag.mem", dut.u_bn_sr1.bias_imag_mem);
        $readmemh("mem/slimsei64_demo/sr1_gate.mem", dut.u_bn_sr1.gate_mem);

        $readmemh("mem/slimsei64_demo/bn2_a_rr.mem", dut.u_bn_sr2.a_rr_mem);
        $readmemh("mem/slimsei64_demo/bn2_a_ri.mem", dut.u_bn_sr2.a_ri_mem);
        $readmemh("mem/slimsei64_demo/bn2_a_ir.mem", dut.u_bn_sr2.a_ir_mem);
        $readmemh("mem/slimsei64_demo/bn2_a_ii.mem", dut.u_bn_sr2.a_ii_mem);
        $readmemh("mem/slimsei64_demo/bn2_b_real.mem", dut.u_bn_sr2.bias_real_mem);
        $readmemh("mem/slimsei64_demo/bn2_b_imag.mem", dut.u_bn_sr2.bias_imag_mem);
        $readmemh("mem/slimsei64_demo/sr2_gate.mem", dut.u_bn_sr2.gate_mem);

        $readmemh("mem/slimsei64_demo/bn3_a_rr.mem", dut.u_bn_sr3.a_rr_mem);
        $readmemh("mem/slimsei64_demo/bn3_a_ri.mem", dut.u_bn_sr3.a_ri_mem);
        $readmemh("mem/slimsei64_demo/bn3_a_ir.mem", dut.u_bn_sr3.a_ir_mem);
        $readmemh("mem/slimsei64_demo/bn3_a_ii.mem", dut.u_bn_sr3.a_ii_mem);
        $readmemh("mem/slimsei64_demo/bn3_b_real.mem", dut.u_bn_sr3.bias_real_mem);
        $readmemh("mem/slimsei64_demo/bn3_b_imag.mem", dut.u_bn_sr3.bias_imag_mem);
        $readmemh("mem/slimsei64_demo/sr3_gate.mem", dut.u_bn_sr3.gate_mem);

        $readmemh("mem/slimsei64_demo/bn4_a_rr.mem", dut.u_bn_sr4.a_rr_mem);
        $readmemh("mem/slimsei64_demo/bn4_a_ri.mem", dut.u_bn_sr4.a_ri_mem);
        $readmemh("mem/slimsei64_demo/bn4_a_ir.mem", dut.u_bn_sr4.a_ir_mem);
        $readmemh("mem/slimsei64_demo/bn4_a_ii.mem", dut.u_bn_sr4.a_ii_mem);
        $readmemh("mem/slimsei64_demo/bn4_b_real.mem", dut.u_bn_sr4.bias_real_mem);
        $readmemh("mem/slimsei64_demo/bn4_b_imag.mem", dut.u_bn_sr4.bias_imag_mem);
        $readmemh("mem/slimsei64_demo/sr4_gate.mem", dut.u_bn_sr4.gate_mem);

        $readmemh("mem/slimsei64_demo/bn5_a_rr.mem", dut.u_bn_sr5.a_rr_mem);
        $readmemh("mem/slimsei64_demo/bn5_a_ri.mem", dut.u_bn_sr5.a_ri_mem);
        $readmemh("mem/slimsei64_demo/bn5_a_ir.mem", dut.u_bn_sr5.a_ir_mem);
        $readmemh("mem/slimsei64_demo/bn5_a_ii.mem", dut.u_bn_sr5.a_ii_mem);
        $readmemh("mem/slimsei64_demo/bn5_b_real.mem", dut.u_bn_sr5.bias_real_mem);
        $readmemh("mem/slimsei64_demo/bn5_b_imag.mem", dut.u_bn_sr5.bias_imag_mem);
        $readmemh("mem/slimsei64_demo/sr5_gate.mem", dut.u_bn_sr5.gate_mem);

        $readmemh("mem/slimsei64_demo/bn6_a_rr.mem", dut.u_bn_sr6.a_rr_mem);
        $readmemh("mem/slimsei64_demo/bn6_a_ri.mem", dut.u_bn_sr6.a_ri_mem);
        $readmemh("mem/slimsei64_demo/bn6_a_ir.mem", dut.u_bn_sr6.a_ir_mem);
        $readmemh("mem/slimsei64_demo/bn6_a_ii.mem", dut.u_bn_sr6.a_ii_mem);
        $readmemh("mem/slimsei64_demo/bn6_b_real.mem", dut.u_bn_sr6.bias_real_mem);
        $readmemh("mem/slimsei64_demo/bn6_b_imag.mem", dut.u_bn_sr6.bias_imag_mem);
        $readmemh("mem/slimsei64_demo/sr6_gate.mem", dut.u_bn_sr6.gate_mem);

        $readmemh("mem/slimsei64_demo/bn7_a_rr.mem", dut.u_bn_sr7.a_rr_mem);
        $readmemh("mem/slimsei64_demo/bn7_a_ri.mem", dut.u_bn_sr7.a_ri_mem);
        $readmemh("mem/slimsei64_demo/bn7_a_ir.mem", dut.u_bn_sr7.a_ir_mem);
        $readmemh("mem/slimsei64_demo/bn7_a_ii.mem", dut.u_bn_sr7.a_ii_mem);
        $readmemh("mem/slimsei64_demo/bn7_b_real.mem", dut.u_bn_sr7.bias_real_mem);
        $readmemh("mem/slimsei64_demo/bn7_b_imag.mem", dut.u_bn_sr7.bias_imag_mem);
        $readmemh("mem/slimsei64_demo/sr7_gate.mem", dut.u_bn_sr7.gate_mem);

        $readmemh("mem/slimsei64_demo/bn8_a_rr.mem", dut.u_bn_sr8.a_rr_mem);
        $readmemh("mem/slimsei64_demo/bn8_a_ri.mem", dut.u_bn_sr8.a_ri_mem);
        $readmemh("mem/slimsei64_demo/bn8_a_ir.mem", dut.u_bn_sr8.a_ir_mem);
        $readmemh("mem/slimsei64_demo/bn8_a_ii.mem", dut.u_bn_sr8.a_ii_mem);
        $readmemh("mem/slimsei64_demo/bn8_b_real.mem", dut.u_bn_sr8.bias_real_mem);
        $readmemh("mem/slimsei64_demo/bn8_b_imag.mem", dut.u_bn_sr8.bias_imag_mem);
        $readmemh("mem/slimsei64_demo/sr8_gate.mem", dut.u_bn_sr8.gate_mem);

        $readmemh("mem/slimsei64_demo/bn9_a_rr.mem", dut.u_bn_sr9.a_rr_mem);
        $readmemh("mem/slimsei64_demo/bn9_a_ri.mem", dut.u_bn_sr9.a_ri_mem);
        $readmemh("mem/slimsei64_demo/bn9_a_ir.mem", dut.u_bn_sr9.a_ir_mem);
        $readmemh("mem/slimsei64_demo/bn9_a_ii.mem", dut.u_bn_sr9.a_ii_mem);
        $readmemh("mem/slimsei64_demo/bn9_b_real.mem", dut.u_bn_sr9.bias_real_mem);
        $readmemh("mem/slimsei64_demo/bn9_b_imag.mem", dut.u_bn_sr9.bias_imag_mem);
        $readmemh("mem/slimsei64_demo/sr9_gate.mem", dut.u_bn_sr9.gate_mem);

        repeat (5) @(posedge clk);
        rst_n = 1'b1;

        load_input_mem();
        start_once();

        // 这条部署链路仍然是单 MAC 顺序卷积，9 层完整跑完需要较多时钟周期。
        repeat (80000000) begin
            @(posedge clk);
            if (!got_done && done) begin
                got_done = 1'b1;
                check_output();
                $display("TEST PASS: cnn1d_complex_feature_extractor9_deploy_top");
                $finish;
            end
        end

        $display("TEST FAIL: cnn1d_complex_feature_extractor9_deploy_top timeout");
        $finish;
    end

endmodule
