`timescale 1ns/1ps
`include "cnn1d_defs_fpga_mix12.vh"

module cnn1d_complex_classifier_deploy_fpga_mix12_fast_tb;

    localparam integer TB_INPUT_LEN        = 1000;
    localparam integer TB_INPUT_COMPLEX_CH = 1;
    localparam integer TB_STAGE_COMPLEX_CH = 64;
    localparam integer TB_KERNEL           = 3;
    localparam integer TB_DENSE1_DIM       = 1024;
    localparam integer TB_NUM_CLASSES      = 7;
    localparam integer TB_INPUT_FLAT_LEN   = TB_INPUT_LEN * 2;
    localparam integer TB_MAX_WAIT_CYCLES  = 150000000;

    localparam integer TB_ACT_SHIFT1       = 12;
    localparam integer TB_ACT_SHIFT2       = 14;
    localparam integer TB_ACT_SHIFT3       = 15;
    localparam integer TB_ACT_SHIFT4       = 13;
    localparam integer TB_ACT_SHIFT5       = 13;
    localparam integer TB_ACT_SHIFT6       = 12;
    localparam integer TB_ACT_SHIFT7       = 13;
    localparam integer TB_ACT_SHIFT8       = 12;
    localparam integer TB_ACT_SHIFT9       = 13;
    localparam integer TB_BN_SHIFT         = 10;
    localparam integer TB_SR_SHIFT         = 0;
    localparam integer TB_SR_THRESH        = 0;
    localparam integer TB_DENSE1_SHIFT     = 9;
    localparam integer TB_DENSE2_SHIFT     = 19;

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

    reg signed [`CNN_DATA_W-1:0] input_mem [0:TB_INPUT_FLAT_LEN-1];
    reg [7:0] expect_class_mem [0:0];
    reg signed [`CNN_ACC_W-1:0] expect_score_mem [0:0];

    integer i;
    integer wait_cnt;

    cnn1d_complex_classifier_deploy_top_fpga_mix12 #(
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
        .CONV_W_REAL_FILE("mem/fpga_mix12_demo/conv_w_real.mem"),
        .CONV_W_IMAG_FILE("mem/fpga_mix12_demo/conv_w_imag.mem"),
        .CONV_B_REAL_FILE("mem/fpga_mix12_demo/conv_b_real.mem"),
        .CONV_B_IMAG_FILE("mem/fpga_mix12_demo/conv_b_imag.mem"),
        .DENSE1_W_FILE("mem/fpga_mix12_demo/dense1_w.mem"),
        .DENSE1_B_FILE("mem/fpga_mix12_demo/dense1_b.mem"),
        .SR10_GATE_FILE("mem/fpga_mix12_demo/sr10_gate.mem"),
        .DENSE2_W_FILE("mem/fpga_mix12_demo/dense2_w.mem"),
        .DENSE2_B_FILE("mem/fpga_mix12_demo/dense2_b.mem")
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

    initial begin
        rst_n = 1'b0;
        start = 1'b0;
        load_en = 1'b0;
        load_addr = {`CNN_ADDR_W{1'b0}};
        load_data = {`CNN_DATA_W{1'b0}};

        $readmemh("mem/fpga_mix12_demo/input_iq.mem", input_mem);
        $readmemh("mem/fpga_mix12_demo/expected_class.mem", expect_class_mem);
        $readmemh("mem/fpga_mix12_demo/expected_score.mem", expect_score_mem);

        $readmemh("mem/fpga_mix12_demo/bn1_a_rr.mem", dut.u_feat9.u_bn_sr1.a_rr_mem);
        $readmemh("mem/fpga_mix12_demo/bn1_a_ri.mem", dut.u_feat9.u_bn_sr1.a_ri_mem);
        $readmemh("mem/fpga_mix12_demo/bn1_a_ir.mem", dut.u_feat9.u_bn_sr1.a_ir_mem);
        $readmemh("mem/fpga_mix12_demo/bn1_a_ii.mem", dut.u_feat9.u_bn_sr1.a_ii_mem);
        $readmemh("mem/fpga_mix12_demo/bn1_b_real.mem", dut.u_feat9.u_bn_sr1.bias_real_mem);
        $readmemh("mem/fpga_mix12_demo/bn1_b_imag.mem", dut.u_feat9.u_bn_sr1.bias_imag_mem);
        $readmemh("mem/fpga_mix12_demo/sr1_gate.mem", dut.u_feat9.u_bn_sr1.gate_mem);
        $readmemh("mem/fpga_mix12_demo/bn2_a_rr.mem", dut.u_feat9.u_bn_sr2.a_rr_mem);
        $readmemh("mem/fpga_mix12_demo/bn2_a_ri.mem", dut.u_feat9.u_bn_sr2.a_ri_mem);
        $readmemh("mem/fpga_mix12_demo/bn2_a_ir.mem", dut.u_feat9.u_bn_sr2.a_ir_mem);
        $readmemh("mem/fpga_mix12_demo/bn2_a_ii.mem", dut.u_feat9.u_bn_sr2.a_ii_mem);
        $readmemh("mem/fpga_mix12_demo/bn2_b_real.mem", dut.u_feat9.u_bn_sr2.bias_real_mem);
        $readmemh("mem/fpga_mix12_demo/bn2_b_imag.mem", dut.u_feat9.u_bn_sr2.bias_imag_mem);
        $readmemh("mem/fpga_mix12_demo/sr2_gate.mem", dut.u_feat9.u_bn_sr2.gate_mem);
        $readmemh("mem/fpga_mix12_demo/bn3_a_rr.mem", dut.u_feat9.u_bn_sr3.a_rr_mem);
        $readmemh("mem/fpga_mix12_demo/bn3_a_ri.mem", dut.u_feat9.u_bn_sr3.a_ri_mem);
        $readmemh("mem/fpga_mix12_demo/bn3_a_ir.mem", dut.u_feat9.u_bn_sr3.a_ir_mem);
        $readmemh("mem/fpga_mix12_demo/bn3_a_ii.mem", dut.u_feat9.u_bn_sr3.a_ii_mem);
        $readmemh("mem/fpga_mix12_demo/bn3_b_real.mem", dut.u_feat9.u_bn_sr3.bias_real_mem);
        $readmemh("mem/fpga_mix12_demo/bn3_b_imag.mem", dut.u_feat9.u_bn_sr3.bias_imag_mem);
        $readmemh("mem/fpga_mix12_demo/sr3_gate.mem", dut.u_feat9.u_bn_sr3.gate_mem);
        $readmemh("mem/fpga_mix12_demo/bn4_a_rr.mem", dut.u_feat9.u_bn_sr4.a_rr_mem);
        $readmemh("mem/fpga_mix12_demo/bn4_a_ri.mem", dut.u_feat9.u_bn_sr4.a_ri_mem);
        $readmemh("mem/fpga_mix12_demo/bn4_a_ir.mem", dut.u_feat9.u_bn_sr4.a_ir_mem);
        $readmemh("mem/fpga_mix12_demo/bn4_a_ii.mem", dut.u_feat9.u_bn_sr4.a_ii_mem);
        $readmemh("mem/fpga_mix12_demo/bn4_b_real.mem", dut.u_feat9.u_bn_sr4.bias_real_mem);
        $readmemh("mem/fpga_mix12_demo/bn4_b_imag.mem", dut.u_feat9.u_bn_sr4.bias_imag_mem);
        $readmemh("mem/fpga_mix12_demo/sr4_gate.mem", dut.u_feat9.u_bn_sr4.gate_mem);
        $readmemh("mem/fpga_mix12_demo/bn5_a_rr.mem", dut.u_feat9.u_bn_sr5.a_rr_mem);
        $readmemh("mem/fpga_mix12_demo/bn5_a_ri.mem", dut.u_feat9.u_bn_sr5.a_ri_mem);
        $readmemh("mem/fpga_mix12_demo/bn5_a_ir.mem", dut.u_feat9.u_bn_sr5.a_ir_mem);
        $readmemh("mem/fpga_mix12_demo/bn5_a_ii.mem", dut.u_feat9.u_bn_sr5.a_ii_mem);
        $readmemh("mem/fpga_mix12_demo/bn5_b_real.mem", dut.u_feat9.u_bn_sr5.bias_real_mem);
        $readmemh("mem/fpga_mix12_demo/bn5_b_imag.mem", dut.u_feat9.u_bn_sr5.bias_imag_mem);
        $readmemh("mem/fpga_mix12_demo/sr5_gate.mem", dut.u_feat9.u_bn_sr5.gate_mem);
        $readmemh("mem/fpga_mix12_demo/bn6_a_rr.mem", dut.u_feat9.u_bn_sr6.a_rr_mem);
        $readmemh("mem/fpga_mix12_demo/bn6_a_ri.mem", dut.u_feat9.u_bn_sr6.a_ri_mem);
        $readmemh("mem/fpga_mix12_demo/bn6_a_ir.mem", dut.u_feat9.u_bn_sr6.a_ir_mem);
        $readmemh("mem/fpga_mix12_demo/bn6_a_ii.mem", dut.u_feat9.u_bn_sr6.a_ii_mem);
        $readmemh("mem/fpga_mix12_demo/bn6_b_real.mem", dut.u_feat9.u_bn_sr6.bias_real_mem);
        $readmemh("mem/fpga_mix12_demo/bn6_b_imag.mem", dut.u_feat9.u_bn_sr6.bias_imag_mem);
        $readmemh("mem/fpga_mix12_demo/sr6_gate.mem", dut.u_feat9.u_bn_sr6.gate_mem);
        $readmemh("mem/fpga_mix12_demo/bn7_a_rr.mem", dut.u_feat9.u_bn_sr7.a_rr_mem);
        $readmemh("mem/fpga_mix12_demo/bn7_a_ri.mem", dut.u_feat9.u_bn_sr7.a_ri_mem);
        $readmemh("mem/fpga_mix12_demo/bn7_a_ir.mem", dut.u_feat9.u_bn_sr7.a_ir_mem);
        $readmemh("mem/fpga_mix12_demo/bn7_a_ii.mem", dut.u_feat9.u_bn_sr7.a_ii_mem);
        $readmemh("mem/fpga_mix12_demo/bn7_b_real.mem", dut.u_feat9.u_bn_sr7.bias_real_mem);
        $readmemh("mem/fpga_mix12_demo/bn7_b_imag.mem", dut.u_feat9.u_bn_sr7.bias_imag_mem);
        $readmemh("mem/fpga_mix12_demo/sr7_gate.mem", dut.u_feat9.u_bn_sr7.gate_mem);
        $readmemh("mem/fpga_mix12_demo/bn8_a_rr.mem", dut.u_feat9.u_bn_sr8.a_rr_mem);
        $readmemh("mem/fpga_mix12_demo/bn8_a_ri.mem", dut.u_feat9.u_bn_sr8.a_ri_mem);
        $readmemh("mem/fpga_mix12_demo/bn8_a_ir.mem", dut.u_feat9.u_bn_sr8.a_ir_mem);
        $readmemh("mem/fpga_mix12_demo/bn8_a_ii.mem", dut.u_feat9.u_bn_sr8.a_ii_mem);
        $readmemh("mem/fpga_mix12_demo/bn8_b_real.mem", dut.u_feat9.u_bn_sr8.bias_real_mem);
        $readmemh("mem/fpga_mix12_demo/bn8_b_imag.mem", dut.u_feat9.u_bn_sr8.bias_imag_mem);
        $readmemh("mem/fpga_mix12_demo/sr8_gate.mem", dut.u_feat9.u_bn_sr8.gate_mem);
        $readmemh("mem/fpga_mix12_demo/bn9_a_rr.mem", dut.u_feat9.u_bn_sr9.a_rr_mem);
        $readmemh("mem/fpga_mix12_demo/bn9_a_ri.mem", dut.u_feat9.u_bn_sr9.a_ri_mem);
        $readmemh("mem/fpga_mix12_demo/bn9_a_ir.mem", dut.u_feat9.u_bn_sr9.a_ir_mem);
        $readmemh("mem/fpga_mix12_demo/bn9_a_ii.mem", dut.u_feat9.u_bn_sr9.a_ii_mem);
        $readmemh("mem/fpga_mix12_demo/bn9_b_real.mem", dut.u_feat9.u_bn_sr9.bias_real_mem);
        $readmemh("mem/fpga_mix12_demo/bn9_b_imag.mem", dut.u_feat9.u_bn_sr9.bias_imag_mem);
        $readmemh("mem/fpga_mix12_demo/sr9_gate.mem", dut.u_feat9.u_bn_sr9.gate_mem);

        repeat (5) @(posedge clk);
        rst_n = 1'b1;

        load_input_mem();
        start_once();

        for (wait_cnt = 0; wait_cnt < TB_MAX_WAIT_CYCLES; wait_cnt = wait_cnt + 1) begin
            @(posedge clk);
            if (done) begin
                $display("RTL class=%0d score=%0d expect_class=%0d expect_score=%0d",
                    out_class, out_score, expect_class_mem[0], expect_score_mem[0]);
                if (out_class == expect_class_mem[0] && out_score == expect_score_mem[0]) begin
                    $display("TEST PASS: cnn1d_complex_classifier_deploy_top_fpga_mix12_fast_tb");
                end else begin
                    $display("TEST FAIL: cnn1d_complex_classifier_deploy_top_fpga_mix12_fast_tb");
                end
                $finish;
            end
        end

        $display("TEST FAIL: timeout");
        $finish;
    end

endmodule
