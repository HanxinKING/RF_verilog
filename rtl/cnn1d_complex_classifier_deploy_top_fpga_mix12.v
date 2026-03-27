`timescale 1ns/1ps
`include "cnn1d_defs_fpga_mix12.vh"

// ============================================================
// 复值 9 层特征提取 + dense1 + sr10 + dense2 分类顶层
// ------------------------------------------------------------
// 这版顶层在 feature_extractor_deploy_top 的基础上，继续接上：
//
// 1. dense1  : 128 -> 1024，带 ReLU
// 2. sr10    : 对 dense1 的 1024 维实值向量逐元素门控
// 3. dense2  : 1024 -> 7，不做 softmax，只输出 argmax
//
// 对应 Python 网络：
//   feature -> Dense(1024, relu) -> SparsityRegularization -> Dense(7)
//
// 其中：
// - Dropout 在部署时忽略
// - softmax 在部署时忽略，只保留 argmax 结果
// ============================================================

(* DONT_TOUCH = "TRUE", KEEP_HIERARCHY = "TRUE" *)
module cnn1d_complex_classifier_deploy_top_fpga_mix12 #(
    parameter integer INPUT_LEN          = 1000,
    parameter integer INPUT_COMPLEX_CH   = 1,
    parameter integer STAGE_COMPLEX_CH   = 64,
    parameter integer KERNEL             = 3,
    parameter integer ACT_SHIFT          = 6,
    parameter integer ACT_SHIFT1         = ACT_SHIFT,
    parameter integer ACT_SHIFT2         = ACT_SHIFT,
    parameter integer ACT_SHIFT3         = ACT_SHIFT,
    parameter integer ACT_SHIFT4         = ACT_SHIFT,
    parameter integer ACT_SHIFT5         = ACT_SHIFT,
    parameter integer ACT_SHIFT6         = ACT_SHIFT,
    parameter integer ACT_SHIFT7         = ACT_SHIFT,
    parameter integer ACT_SHIFT8         = ACT_SHIFT,
    parameter integer ACT_SHIFT9         = ACT_SHIFT,
    parameter integer BN_SHIFT           = 10,
    parameter integer SR_SHIFT           = 0,
    parameter integer SR_THRESH          = 0,
    parameter integer WIDE_STAGES        = 3,
    parameter integer REQUANT_BITS       = `CNN_DATA_W,
    parameter integer DENSE1_DIM         = 1024,
    parameter integer NUM_CLASSES        = 7,
    parameter integer DENSE1_SHIFT       = 10,
    parameter integer DENSE2_SHIFT       = 9,
    parameter integer RAM_DEPTH          = `CNN_FEATURE_RAM_DEPTH,
    parameter integer LOAD_CONV_W_REAL   = 0,
    parameter integer LOAD_CONV_W_IMAG   = 0,
    parameter integer LOAD_CONV_B_REAL   = 0,
    parameter integer LOAD_CONV_B_IMAG   = 0,
    parameter integer LOAD_DENSE1_W      = 0,
    parameter integer LOAD_DENSE1_B      = 0,
    parameter integer LOAD_SR10_GATE     = 0,
    parameter integer LOAD_DENSE2_W      = 0,
    parameter integer LOAD_DENSE2_B      = 0,
    parameter integer LOAD_DENSE2_SHIFT  = 0,
    parameter CONV_W_REAL_FILE           = "",
    parameter CONV_W_IMAG_FILE           = "",
    parameter CONV_B_REAL_FILE           = "",
    parameter CONV_B_IMAG_FILE           = "",
    parameter DENSE1_W_FILE              = "",
    parameter DENSE1_B_FILE              = "",
    parameter SR10_GATE_FILE             = "",
    parameter DENSE2_W_FILE              = "",
    parameter DENSE2_B_FILE              = "",
    parameter DENSE2_SHIFT_FILE          = ""
) (
    input                                 clk,
    input                                 rst_n,
    input                                 start,
    input                                 load_en,
    input      [`CNN_ADDR_W-1:0]          load_addr,
    input signed [`CNN_DATA_W-1:0]        load_data,
    output                                busy,
    output                                done,
    output                                out_valid,
    output reg [`CNN_CLASS_W-1:0]         out_class,
    output reg signed [`CNN_ACC_W-1:0]    out_score
);

    localparam [`CNN_ADDR_W-1:0] ADDR_ZERO = {`CNN_ADDR_W{1'b0}};
    localparam integer FINAL_FEAT_DIM = STAGE_COMPLEX_CH << 1;

    localparam ST_IDLE         = 4'd0;
    localparam ST_FEAT_START   = 4'd1;
    localparam ST_FEAT_WAIT    = 4'd2;
    localparam ST_DENSE1_START = 4'd3;
    localparam ST_DENSE1_WAIT  = 4'd4;
    localparam ST_SR10_START   = 4'd5;
    localparam ST_SR10_WAIT    = 4'd6;
    localparam ST_DENSE2_START = 4'd7;
    localparam ST_DENSE2_WAIT  = 4'd8;
    localparam ST_DONE         = 4'd9;

    reg [3:0] state;

    (* DONT_TOUCH = "TRUE", KEEP = "TRUE" *) wire feat_done;
    (* DONT_TOUCH = "TRUE", KEEP = "TRUE" *) wire dense1_done;
    (* DONT_TOUCH = "TRUE", KEEP = "TRUE" *) wire sr10_done;
    (* DONT_TOUCH = "TRUE", KEEP = "TRUE" *) wire dense2_done;

    wire [`CNN_ADDR_W-1:0] feat_out_addr;
    (* DONT_TOUCH = "TRUE", KEEP = "TRUE" *) wire signed [`CNN_FEAT_W-1:0] feat_out_data;
    wire [`CNN_ADDR_W-1:0] dense1_feat_rd_addr;

    wire dense1_wr_en;
    wire [`CNN_ADDR_W-1:0] dense1_wr_addr;
    wire signed [`CNN_FEAT_W-1:0] dense1_wr_data;
    wire [`CNN_ADDR_W-1:0] dense1_mem_rd_addr;
    (* DONT_TOUCH = "TRUE", KEEP = "TRUE" *) wire signed [`CNN_FEAT_W-1:0] dense1_mem_rd_data;

    wire [`CNN_ADDR_W-1:0] sr10_rd_addr;
    wire sr10_wr_en;
    wire [`CNN_ADDR_W-1:0] sr10_wr_addr;
    wire signed [`CNN_FEAT_W-1:0] sr10_wr_data;
    wire [`CNN_ADDR_W-1:0] sr10_mem_rd_addr;
    (* DONT_TOUCH = "TRUE", KEEP = "TRUE" *) wire signed [`CNN_FEAT_W-1:0] sr10_mem_rd_data;

    wire [`CNN_ADDR_W-1:0] dense2_feat_rd_addr;
    wire dense2_score_valid;
    wire [`CNN_CLASS_W-1:0] dense2_score_class_idx;
    wire signed [`CNN_ACC_W-1:0] dense2_score_class_value;
    (* DONT_TOUCH = "TRUE", KEEP = "TRUE" *) wire [`CNN_CLASS_W-1:0] dense2_class_idx;
    (* DONT_TOUCH = "TRUE", KEEP = "TRUE" *) wire signed [`CNN_ACC_W-1:0] dense2_class_score;

    assign busy = (state != ST_IDLE) && (state != ST_DONE);
    assign done = (state == ST_DONE);
    assign out_valid = (state == ST_DONE);

    assign feat_out_addr = ((state == ST_DENSE1_START) || (state == ST_DENSE1_WAIT)) ? dense1_feat_rd_addr : ADDR_ZERO;
    assign dense1_mem_rd_addr = ((state == ST_SR10_START) || (state == ST_SR10_WAIT)) ? sr10_rd_addr : ADDR_ZERO;
    assign sr10_mem_rd_addr = ((state == ST_DENSE2_START) || (state == ST_DENSE2_WAIT)) ? dense2_feat_rd_addr : ADDR_ZERO;

    (* DONT_TOUCH = "TRUE" *) cnn1d_complex_feature_extractor9_deploy_top_fpga_mix12 #(
        .INPUT_LEN(INPUT_LEN),
        .INPUT_COMPLEX_CH(INPUT_COMPLEX_CH),
        .STAGE_COMPLEX_CH(STAGE_COMPLEX_CH),
        .KERNEL(KERNEL),
        .ACT_SHIFT(ACT_SHIFT),
        .ACT_SHIFT1(ACT_SHIFT1),
        .ACT_SHIFT2(ACT_SHIFT2),
        .ACT_SHIFT3(ACT_SHIFT3),
        .ACT_SHIFT4(ACT_SHIFT4),
        .ACT_SHIFT5(ACT_SHIFT5),
        .ACT_SHIFT6(ACT_SHIFT6),
        .ACT_SHIFT7(ACT_SHIFT7),
        .ACT_SHIFT8(ACT_SHIFT8),
        .ACT_SHIFT9(ACT_SHIFT9),
        .BN_SHIFT(BN_SHIFT),
        .SR_SHIFT(SR_SHIFT),
        .SR_THRESH(SR_THRESH),
        .WIDE_STAGES(WIDE_STAGES),
        .REQUANT_BITS(REQUANT_BITS),
        .RAM_DEPTH(RAM_DEPTH),
        .LOAD_CONV_W_REAL(LOAD_CONV_W_REAL),
        .LOAD_CONV_W_IMAG(LOAD_CONV_W_IMAG),
        .LOAD_CONV_B_REAL(LOAD_CONV_B_REAL),
        .LOAD_CONV_B_IMAG(LOAD_CONV_B_IMAG),
        .CONV_W_REAL_FILE(CONV_W_REAL_FILE),
        .CONV_W_IMAG_FILE(CONV_W_IMAG_FILE),
        .CONV_B_REAL_FILE(CONV_B_REAL_FILE),
        .CONV_B_IMAG_FILE(CONV_B_IMAG_FILE)
    ) u_feat9 (
        .clk(clk),
        .rst_n(rst_n),
        .start(state == ST_FEAT_START),
        .load_en(load_en),
        .load_addr(load_addr),
        .load_data(load_data),
        .feat_out_addr(feat_out_addr),
        .feat_out_data(feat_out_data),
        .busy(),
        .done(feat_done)
    );

    (* DONT_TOUCH = "TRUE" *) cnn1d_feature_ram_fpga_mix12 #(
        .DATA_W(`CNN_FEAT_W),
        .DEPTH(2048)
    ) u_dense1_mem (
        .clk(clk),
        .wr_en(dense1_wr_en),
        .wr_addr(dense1_wr_addr),
        .wr_data(dense1_wr_data),
        .rd_addr(dense1_mem_rd_addr),
        .rd_data(dense1_mem_rd_data)
    );

    (* DONT_TOUCH = "TRUE" *) cnn1d_feature_ram_fpga_mix12 #(
        .DATA_W(`CNN_FEAT_W),
        .DEPTH(2048)
    ) u_sr10_mem (
        .clk(clk),
        .wr_en(sr10_wr_en),
        .wr_addr(sr10_wr_addr),
        .wr_data(sr10_wr_data),
        .rd_addr(sr10_mem_rd_addr),
        .rd_data(sr10_mem_rd_data)
    );

    (* DONT_TOUCH = "TRUE" *) cnn1d_flatten_dense_core_fpga_mix12 #(
        .FEAT_LEN(1),
        .CHANNELS(FINAL_FEAT_DIM),
        .OUT_DIM(DENSE1_DIM),
        .ACT_SHIFT(DENSE1_SHIFT),
        .LOAD_FC_W(LOAD_DENSE1_W),
        .LOAD_FC_B(LOAD_DENSE1_B),
        .FC_W_FILE(DENSE1_W_FILE),
        .FC_B_FILE(DENSE1_B_FILE)
    ) u_dense1 (
        .clk(clk),
        .rst_n(rst_n),
        .start(state == ST_DENSE1_START),
        .in_base(ADDR_ZERO),
        .out_base(ADDR_ZERO),
        .feat_rd_addr(dense1_feat_rd_addr),
        .feat_rd_data(feat_out_data),
        .feat_wr_en(dense1_wr_en),
        .feat_wr_addr(dense1_wr_addr),
        .feat_wr_data(dense1_wr_data),
        .busy(),
        .done(dense1_done)
    );

    (* DONT_TOUCH = "TRUE" *) cnn1d_vector_sr_core_fpga_mix12 #(
        .VEC_LEN(DENSE1_DIM),
        .GATE_SHIFT(SR_SHIFT),
        .LOAD_GATE(LOAD_SR10_GATE),
        .GATE_FILE(SR10_GATE_FILE)
    ) u_sr10 (
        .clk(clk),
        .rst_n(rst_n),
        .start(state == ST_SR10_START),
        .in_base(ADDR_ZERO),
        .out_base(ADDR_ZERO),
        .feat_rd_addr(sr10_rd_addr),
        .feat_rd_data(dense1_mem_rd_data),
        .feat_wr_en(sr10_wr_en),
        .feat_wr_addr(sr10_wr_addr),
        .feat_wr_data(sr10_wr_data),
        .busy(),
        .done(sr10_done)
    );

    (* DONT_TOUCH = "TRUE" *) cnn1d_dense_argmax_core_fpga_mix12 #(
        .IN_DIM(DENSE1_DIM),
        .NUM_CLASSES(NUM_CLASSES),
        .ACT_SHIFT(DENSE2_SHIFT),
        .LOAD_FC_W(LOAD_DENSE2_W),
        .LOAD_FC_B(LOAD_DENSE2_B),
        .LOAD_FC_SHIFT(LOAD_DENSE2_SHIFT),
        .FC_W_FILE(DENSE2_W_FILE),
        .FC_B_FILE(DENSE2_B_FILE),
        .FC_SHIFT_FILE(DENSE2_SHIFT_FILE)
    ) u_dense2 (
        .clk(clk),
        .rst_n(rst_n),
        .start(state == ST_DENSE2_START),
        .in_base(ADDR_ZERO),
        .feat_rd_addr(dense2_feat_rd_addr),
        .feat_rd_data(sr10_mem_rd_data),
        .score_valid(dense2_score_valid),
        .score_class_idx(dense2_score_class_idx),
        .score_class_value(dense2_score_class_value),
        .class_valid(),
        .class_idx(dense2_class_idx),
        .class_score(dense2_class_score),
        .busy(),
        .done(dense2_done)
    );

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state     <= ST_IDLE;
            out_class <= {`CNN_CLASS_W{1'b0}};
            out_score <= {`CNN_ACC_W{1'b0}};
        end else begin
            case (state)
                ST_IDLE: begin
                    if (start) begin
                        state <= ST_FEAT_START;
                    end
                end

                ST_FEAT_START: begin
                    state <= ST_FEAT_WAIT;
                end

                ST_FEAT_WAIT: begin
                    if (feat_done) begin
                        state <= ST_DENSE1_START;
                    end
                end

                ST_DENSE1_START: begin
                    state <= ST_DENSE1_WAIT;
                end

                ST_DENSE1_WAIT: begin
                    if (dense1_done) begin
                        state <= ST_SR10_START;
                    end
                end

                ST_SR10_START: begin
                    state <= ST_SR10_WAIT;
                end

                ST_SR10_WAIT: begin
                    if (sr10_done) begin
                        state <= ST_DENSE2_START;
                    end
                end

                ST_DENSE2_START: begin
                    state <= ST_DENSE2_WAIT;
                end

                ST_DENSE2_WAIT: begin
                    if (dense2_done) begin
                        out_class <= dense2_class_idx;
                        out_score <= dense2_class_score;
                        state     <= ST_DONE;
                    end
                end

                ST_DONE: begin
                    state <= ST_IDLE;
                end

                default: begin
                    state <= ST_IDLE;
                end
            endcase
        end
    end

endmodule
