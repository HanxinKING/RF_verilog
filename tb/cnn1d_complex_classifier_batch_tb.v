`timescale 1ns/1ps
`include "cnn1d_defs.vh"
`include "slimsei64_batch_cfg.vh"

// ============================================================
// 批量 IQ 样本分类部署 testbench
// ------------------------------------------------------------
// 这份 testbench 会顺序执行多条样本：
//   load -> start -> wait done -> record result
//
// 输入与期望输出都来自 Python 侧生成的 mem 文件。
// 最终 RTL 结果会写到 rtl_results.csv，供 Python 汇总统计。
// ============================================================

module cnn1d_complex_classifier_batch_tb;

    localparam integer TB_NUM_SAMPLES      = `SLIMSEI64_BATCH_SAMPLES;
    localparam integer TB_INPUT_LEN        = 1000;
    localparam integer TB_INPUT_COMPLEX_CH = 1;
    localparam integer TB_STAGE_COMPLEX_CH = 64;
    localparam integer TB_KERNEL           = 3;
    localparam integer TB_DENSE1_DIM       = 1024;
    localparam integer TB_NUM_CLASSES      = 7;

    localparam integer TB_ACT_SHIFT1       = `SLIMSEI64_BATCH_ACT_SHIFT1;
    localparam integer TB_ACT_SHIFT2       = `SLIMSEI64_BATCH_ACT_SHIFT2;
    localparam integer TB_ACT_SHIFT3       = `SLIMSEI64_BATCH_ACT_SHIFT3;
    localparam integer TB_ACT_SHIFT4       = `SLIMSEI64_BATCH_ACT_SHIFT4;
    localparam integer TB_ACT_SHIFT5       = `SLIMSEI64_BATCH_ACT_SHIFT5;
    localparam integer TB_ACT_SHIFT6       = `SLIMSEI64_BATCH_ACT_SHIFT6;
    localparam integer TB_ACT_SHIFT7       = `SLIMSEI64_BATCH_ACT_SHIFT7;
    localparam integer TB_ACT_SHIFT8       = `SLIMSEI64_BATCH_ACT_SHIFT8;
    localparam integer TB_ACT_SHIFT9       = `SLIMSEI64_BATCH_ACT_SHIFT9;
    localparam integer TB_BN_SHIFT         = `SLIMSEI64_BATCH_BN_SHIFT;
    localparam integer TB_SR_SHIFT         = `SLIMSEI64_BATCH_SR_SHIFT;
    localparam integer TB_SR_THRESH        = `SLIMSEI64_BATCH_SR_THRESH;
    localparam integer TB_DENSE1_SHIFT     = `SLIMSEI64_BATCH_DENSE1_SHIFT;
    localparam integer TB_DENSE2_SHIFT     = `SLIMSEI64_BATCH_DENSE2_SHIFT;

    localparam integer TB_INPUT_FLAT_LEN   = TB_INPUT_LEN * 2;
    localparam integer TB_MAX_WAIT_CYCLES  = 50000000;

    reg                              clk;
    reg                              rst_n;
    reg                              start;
    reg                              load_en;
    reg  [`CNN_ADDR_W-1:0]           load_addr;
    reg  signed [`CNN_DATA_W-1:0]    load_data;
    wire                             busy;
    wire                             done;
    wire                             out_valid;
    wire [`CNN_CLASS_W-1:0]          out_class;
    wire signed [`CNN_ACC_W-1:0]     out_score;

    reg signed [`CNN_DATA_W-1:0] input_mem [0:TB_NUM_SAMPLES*TB_INPUT_FLAT_LEN-1];
    reg [7:0] expect_class_mem [0:TB_NUM_SAMPLES-1];
    reg signed [`CNN_ACC_W-1:0] expect_score_mem [0:TB_NUM_SAMPLES-1];

    integer i;
    integer sample_idx;
    integer wait_cnt;
    integer result_fd;
    integer logits_fd;
    integer class_match_count;
    integer score_match_count;
    reg got_done;
    reg class_ok;
    reg score_ok;

    cnn1d_complex_classifier_deploy_top #(
        .INPUT_LEN(TB_INPUT_LEN),
        .INPUT_COMPLEX_CH(TB_INPUT_COMPLEX_CH),
        .STAGE_COMPLEX_CH(TB_STAGE_COMPLEX_CH),
        .KERNEL(TB_KERNEL),
        .ACT_SHIFT1(TB_ACT_SHIFT1),
        .ACT_SHIFT2(TB_ACT_SHIFT2),
        .ACT_SHIFT3(TB_ACT_SHIFT3),
        .ACT_SHIFT4(TB_ACT_SHIFT4),
        .ACT_SHIFT5(TB_ACT_SHIFT5),
        .ACT_SHIFT6(TB_ACT_SHIFT6),
        .ACT_SHIFT7(TB_ACT_SHIFT7),
        .ACT_SHIFT8(TB_ACT_SHIFT8),
        .ACT_SHIFT9(TB_ACT_SHIFT9),
        .BN_SHIFT(TB_BN_SHIFT),
        .SR_SHIFT(TB_SR_SHIFT),
        .SR_THRESH(TB_SR_THRESH),
        .DENSE1_DIM(TB_DENSE1_DIM),
        .NUM_CLASSES(TB_NUM_CLASSES),
        .DENSE1_SHIFT(TB_DENSE1_SHIFT),
        .DENSE2_SHIFT(TB_DENSE2_SHIFT),
        .RAM_DEPTH(140000),
        .LOAD_CONV_W_REAL(1),
        .LOAD_CONV_W_IMAG(1),
        .LOAD_CONV_B_REAL(1),
        .LOAD_CONV_B_IMAG(1),
        .LOAD_DENSE1_W(1),
        .LOAD_DENSE1_B(1),
        .LOAD_SR10_GATE(1),
        .LOAD_DENSE2_W(1),
        .LOAD_DENSE2_B(1),
        .CONV_W_REAL_FILE(`SLIMSEI64_BATCH_CONV_W_REAL_FILE),
        .CONV_W_IMAG_FILE(`SLIMSEI64_BATCH_CONV_W_IMAG_FILE),
        .CONV_B_REAL_FILE(`SLIMSEI64_BATCH_CONV_B_REAL_FILE),
        .CONV_B_IMAG_FILE(`SLIMSEI64_BATCH_CONV_B_IMAG_FILE),
        .DENSE1_W_FILE(`SLIMSEI64_BATCH_DENSE1_W_FILE),
        .DENSE1_B_FILE(`SLIMSEI64_BATCH_DENSE1_B_FILE),
        .SR10_GATE_FILE(`SLIMSEI64_BATCH_SR10_GATE_FILE),
        .DENSE2_W_FILE(`SLIMSEI64_BATCH_DENSE2_W_FILE),
        .DENSE2_B_FILE(`SLIMSEI64_BATCH_DENSE2_B_FILE)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .load_en(load_en),
        .load_addr(load_addr),
        .load_data(load_data),
        .busy(busy),
        .done(done),
        .out_valid(out_valid),
        .out_class(out_class),
        .out_score(out_score)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    always @(posedge clk) begin
        if (dut.u_dense2.score_valid) begin
            $fwrite(logits_fd, "%0d,%0d,%0d\n",
                sample_idx,
                dut.u_dense2.score_class_idx,
                dut.u_dense2.score_class_value
            );
        end
    end

    task load_sample;
        input integer sample_id;
        begin
            load_en = 1'b1;
            for (i = 0; i < TB_INPUT_FLAT_LEN; i = i + 1) begin
                @(posedge clk);
                load_addr = i[`CNN_ADDR_W-1:0];
                load_data = input_mem[sample_id*TB_INPUT_FLAT_LEN + i];
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

    initial begin
        rst_n = 1'b0;
        start = 1'b0;
        load_en = 1'b0;
        load_addr = {`CNN_ADDR_W{1'b0}};
        load_data = {`CNN_DATA_W{1'b0}};
        class_match_count = 0;
        score_match_count = 0;
        got_done = 1'b0;
        class_ok = 1'b0;
        score_ok = 1'b0;

        $readmemh(`SLIMSEI64_BATCH_INPUT_FILE, input_mem);
        $readmemh(`SLIMSEI64_BATCH_EXPECT_CLASS_FILE, expect_class_mem);
        $readmemh(`SLIMSEI64_BATCH_EXPECT_SCORE_FILE, expect_score_mem);

        $readmemh(`SLIMSEI64_BATCH_BN1_A_RR_FILE, dut.u_feat9.u_bn_sr1.a_rr_mem);
        $readmemh(`SLIMSEI64_BATCH_BN1_A_RI_FILE, dut.u_feat9.u_bn_sr1.a_ri_mem);
        $readmemh(`SLIMSEI64_BATCH_BN1_A_IR_FILE, dut.u_feat9.u_bn_sr1.a_ir_mem);
        $readmemh(`SLIMSEI64_BATCH_BN1_A_II_FILE, dut.u_feat9.u_bn_sr1.a_ii_mem);
        $readmemh(`SLIMSEI64_BATCH_BN1_B_REAL_FILE, dut.u_feat9.u_bn_sr1.bias_real_mem);
        $readmemh(`SLIMSEI64_BATCH_BN1_B_IMAG_FILE, dut.u_feat9.u_bn_sr1.bias_imag_mem);
        $readmemh(`SLIMSEI64_BATCH_SR1_GATE_FILE, dut.u_feat9.u_bn_sr1.gate_mem);

        $readmemh(`SLIMSEI64_BATCH_BN2_A_RR_FILE, dut.u_feat9.u_bn_sr2.a_rr_mem);
        $readmemh(`SLIMSEI64_BATCH_BN2_A_RI_FILE, dut.u_feat9.u_bn_sr2.a_ri_mem);
        $readmemh(`SLIMSEI64_BATCH_BN2_A_IR_FILE, dut.u_feat9.u_bn_sr2.a_ir_mem);
        $readmemh(`SLIMSEI64_BATCH_BN2_A_II_FILE, dut.u_feat9.u_bn_sr2.a_ii_mem);
        $readmemh(`SLIMSEI64_BATCH_BN2_B_REAL_FILE, dut.u_feat9.u_bn_sr2.bias_real_mem);
        $readmemh(`SLIMSEI64_BATCH_BN2_B_IMAG_FILE, dut.u_feat9.u_bn_sr2.bias_imag_mem);
        $readmemh(`SLIMSEI64_BATCH_SR2_GATE_FILE, dut.u_feat9.u_bn_sr2.gate_mem);

        $readmemh(`SLIMSEI64_BATCH_BN3_A_RR_FILE, dut.u_feat9.u_bn_sr3.a_rr_mem);
        $readmemh(`SLIMSEI64_BATCH_BN3_A_RI_FILE, dut.u_feat9.u_bn_sr3.a_ri_mem);
        $readmemh(`SLIMSEI64_BATCH_BN3_A_IR_FILE, dut.u_feat9.u_bn_sr3.a_ir_mem);
        $readmemh(`SLIMSEI64_BATCH_BN3_A_II_FILE, dut.u_feat9.u_bn_sr3.a_ii_mem);
        $readmemh(`SLIMSEI64_BATCH_BN3_B_REAL_FILE, dut.u_feat9.u_bn_sr3.bias_real_mem);
        $readmemh(`SLIMSEI64_BATCH_BN3_B_IMAG_FILE, dut.u_feat9.u_bn_sr3.bias_imag_mem);
        $readmemh(`SLIMSEI64_BATCH_SR3_GATE_FILE, dut.u_feat9.u_bn_sr3.gate_mem);

        $readmemh(`SLIMSEI64_BATCH_BN4_A_RR_FILE, dut.u_feat9.u_bn_sr4.a_rr_mem);
        $readmemh(`SLIMSEI64_BATCH_BN4_A_RI_FILE, dut.u_feat9.u_bn_sr4.a_ri_mem);
        $readmemh(`SLIMSEI64_BATCH_BN4_A_IR_FILE, dut.u_feat9.u_bn_sr4.a_ir_mem);
        $readmemh(`SLIMSEI64_BATCH_BN4_A_II_FILE, dut.u_feat9.u_bn_sr4.a_ii_mem);
        $readmemh(`SLIMSEI64_BATCH_BN4_B_REAL_FILE, dut.u_feat9.u_bn_sr4.bias_real_mem);
        $readmemh(`SLIMSEI64_BATCH_BN4_B_IMAG_FILE, dut.u_feat9.u_bn_sr4.bias_imag_mem);
        $readmemh(`SLIMSEI64_BATCH_SR4_GATE_FILE, dut.u_feat9.u_bn_sr4.gate_mem);

        $readmemh(`SLIMSEI64_BATCH_BN5_A_RR_FILE, dut.u_feat9.u_bn_sr5.a_rr_mem);
        $readmemh(`SLIMSEI64_BATCH_BN5_A_RI_FILE, dut.u_feat9.u_bn_sr5.a_ri_mem);
        $readmemh(`SLIMSEI64_BATCH_BN5_A_IR_FILE, dut.u_feat9.u_bn_sr5.a_ir_mem);
        $readmemh(`SLIMSEI64_BATCH_BN5_A_II_FILE, dut.u_feat9.u_bn_sr5.a_ii_mem);
        $readmemh(`SLIMSEI64_BATCH_BN5_B_REAL_FILE, dut.u_feat9.u_bn_sr5.bias_real_mem);
        $readmemh(`SLIMSEI64_BATCH_BN5_B_IMAG_FILE, dut.u_feat9.u_bn_sr5.bias_imag_mem);
        $readmemh(`SLIMSEI64_BATCH_SR5_GATE_FILE, dut.u_feat9.u_bn_sr5.gate_mem);

        $readmemh(`SLIMSEI64_BATCH_BN6_A_RR_FILE, dut.u_feat9.u_bn_sr6.a_rr_mem);
        $readmemh(`SLIMSEI64_BATCH_BN6_A_RI_FILE, dut.u_feat9.u_bn_sr6.a_ri_mem);
        $readmemh(`SLIMSEI64_BATCH_BN6_A_IR_FILE, dut.u_feat9.u_bn_sr6.a_ir_mem);
        $readmemh(`SLIMSEI64_BATCH_BN6_A_II_FILE, dut.u_feat9.u_bn_sr6.a_ii_mem);
        $readmemh(`SLIMSEI64_BATCH_BN6_B_REAL_FILE, dut.u_feat9.u_bn_sr6.bias_real_mem);
        $readmemh(`SLIMSEI64_BATCH_BN6_B_IMAG_FILE, dut.u_feat9.u_bn_sr6.bias_imag_mem);
        $readmemh(`SLIMSEI64_BATCH_SR6_GATE_FILE, dut.u_feat9.u_bn_sr6.gate_mem);

        $readmemh(`SLIMSEI64_BATCH_BN7_A_RR_FILE, dut.u_feat9.u_bn_sr7.a_rr_mem);
        $readmemh(`SLIMSEI64_BATCH_BN7_A_RI_FILE, dut.u_feat9.u_bn_sr7.a_ri_mem);
        $readmemh(`SLIMSEI64_BATCH_BN7_A_IR_FILE, dut.u_feat9.u_bn_sr7.a_ir_mem);
        $readmemh(`SLIMSEI64_BATCH_BN7_A_II_FILE, dut.u_feat9.u_bn_sr7.a_ii_mem);
        $readmemh(`SLIMSEI64_BATCH_BN7_B_REAL_FILE, dut.u_feat9.u_bn_sr7.bias_real_mem);
        $readmemh(`SLIMSEI64_BATCH_BN7_B_IMAG_FILE, dut.u_feat9.u_bn_sr7.bias_imag_mem);
        $readmemh(`SLIMSEI64_BATCH_SR7_GATE_FILE, dut.u_feat9.u_bn_sr7.gate_mem);

        $readmemh(`SLIMSEI64_BATCH_BN8_A_RR_FILE, dut.u_feat9.u_bn_sr8.a_rr_mem);
        $readmemh(`SLIMSEI64_BATCH_BN8_A_RI_FILE, dut.u_feat9.u_bn_sr8.a_ri_mem);
        $readmemh(`SLIMSEI64_BATCH_BN8_A_IR_FILE, dut.u_feat9.u_bn_sr8.a_ir_mem);
        $readmemh(`SLIMSEI64_BATCH_BN8_A_II_FILE, dut.u_feat9.u_bn_sr8.a_ii_mem);
        $readmemh(`SLIMSEI64_BATCH_BN8_B_REAL_FILE, dut.u_feat9.u_bn_sr8.bias_real_mem);
        $readmemh(`SLIMSEI64_BATCH_BN8_B_IMAG_FILE, dut.u_feat9.u_bn_sr8.bias_imag_mem);
        $readmemh(`SLIMSEI64_BATCH_SR8_GATE_FILE, dut.u_feat9.u_bn_sr8.gate_mem);

        $readmemh(`SLIMSEI64_BATCH_BN9_A_RR_FILE, dut.u_feat9.u_bn_sr9.a_rr_mem);
        $readmemh(`SLIMSEI64_BATCH_BN9_A_RI_FILE, dut.u_feat9.u_bn_sr9.a_ri_mem);
        $readmemh(`SLIMSEI64_BATCH_BN9_A_IR_FILE, dut.u_feat9.u_bn_sr9.a_ir_mem);
        $readmemh(`SLIMSEI64_BATCH_BN9_A_II_FILE, dut.u_feat9.u_bn_sr9.a_ii_mem);
        $readmemh(`SLIMSEI64_BATCH_BN9_B_REAL_FILE, dut.u_feat9.u_bn_sr9.bias_real_mem);
        $readmemh(`SLIMSEI64_BATCH_BN9_B_IMAG_FILE, dut.u_feat9.u_bn_sr9.bias_imag_mem);
        $readmemh(`SLIMSEI64_BATCH_SR9_GATE_FILE, dut.u_feat9.u_bn_sr9.gate_mem);

        result_fd = $fopen(`SLIMSEI64_BATCH_RESULTS_FILE, "w");
        $fwrite(result_fd, "sample_idx,rtl_class,rtl_score,expect_class,expect_score,class_match,score_match\n");
        logits_fd = $fopen(`SLIMSEI64_BATCH_LOGITS_FILE, "w");
        $fwrite(logits_fd, "sample_idx,class_idx,score\n");

        repeat (5) @(posedge clk);
        rst_n = 1'b1;

        for (sample_idx = 0; sample_idx < TB_NUM_SAMPLES; sample_idx = sample_idx + 1) begin
            load_sample(sample_idx);
            start_once();

            got_done = 1'b0;
            class_ok = 1'b0;
            score_ok = 1'b0;

            for (wait_cnt = 0; wait_cnt < TB_MAX_WAIT_CYCLES; wait_cnt = wait_cnt + 1) begin
                @(posedge clk);
                if (!got_done && done) begin
                    got_done = 1'b1;
                    class_ok = (out_class == expect_class_mem[sample_idx]);
                    score_ok = (out_score == expect_score_mem[sample_idx]);
                    if (class_ok) class_match_count = class_match_count + 1;
                    if (score_ok) score_match_count = score_match_count + 1;
                    $fwrite(result_fd, "%0d,%0d,%0d,%0d,%0d,%0d,%0d\n",
                        sample_idx,
                        out_class,
                        out_score,
                        expect_class_mem[sample_idx],
                        expect_score_mem[sample_idx],
                        class_ok,
                        score_ok
                    );
                    @(posedge clk);
                    wait_cnt = TB_MAX_WAIT_CYCLES;
                end
            end

            if (!got_done) begin
                $fwrite(result_fd, "%0d,0,0,%0d,%0d,0,0\n",
                    sample_idx,
                    expect_class_mem[sample_idx],
                    expect_score_mem[sample_idx]
                );
                $display("TEST FAIL: timeout at sample %0d", sample_idx);
                $fclose(result_fd);
                $fclose(logits_fd);
                $finish;
            end
        end

        $fclose(result_fd);
        $fclose(logits_fd);
        $display("BATCH DONE: class_match_count=%0d / %0d", class_match_count, TB_NUM_SAMPLES);
        $display("BATCH DONE: score_match_count=%0d / %0d", score_match_count, TB_NUM_SAMPLES);
        $finish;
    end

endmodule
