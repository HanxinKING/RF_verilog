`timescale 1ns/1ps
`include "cnn1d_defs_fpga_mix12.vh"

// ============================================================
// 面向 Keras ComplexCNN 的 9 层复值特征提取顶层
// ------------------------------------------------------------
// 1. Conv1D 使用复卷积，但输出对实部/虚部分别做 ReLU
// 2. BN 使用从 Keras 推理图折叠出来的 2x2 仿射
// 3. SR 使用每个复通道一组 gate，实部/虚部共用
// 4. MaxPooling1D 按 Keras 的普通通道池化语义执行
/*
Keras 的 MaxPooling1D 是“按每个平面通道独立池化”
它不会按复数模值比较
因此这里池化阶段直接复用实值的 cnn1d_pool_core，
但通道数设置为 2 * STAGE_COMPLEX_CH
*/
// ============================================================

(* DONT_TOUCH = "TRUE", KEEP_HIERARCHY = "TRUE" *)
module cnn1d_complex_feature_extractor9_deploy_top_fpga_mix12 #(
    parameter integer INPUT_LEN          = 1000,
    parameter integer INPUT_COMPLEX_CH   = 1,
    parameter integer STAGE_COMPLEX_CH   = 64,
    parameter integer KERNEL             = 3,
    parameter integer ACT_SHIFT          = 7,
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
    parameter integer RAM_DEPTH          = `CNN_FEATURE_RAM_DEPTH,
    parameter integer LOAD_CONV_W_REAL   = 0,
    parameter integer LOAD_CONV_W_IMAG   = 0,
    parameter integer LOAD_CONV_B_REAL   = 0,
    parameter integer LOAD_CONV_B_IMAG   = 0,
    parameter CONV_W_REAL_FILE           = "",
    parameter CONV_W_IMAG_FILE           = "",
    parameter CONV_B_REAL_FILE           = "",
    parameter CONV_B_IMAG_FILE           = ""
) (
    input                                clk,
    input                                rst_n,
    input                                start,
    input                                load_en,
    input      [`CNN_ADDR_W-1:0]         load_addr,
    input signed [`CNN_DATA_W-1:0]       load_data,
    input      [`CNN_ADDR_W-1:0]         feat_out_addr,
    output signed [`CNN_FEAT_W-1:0]      feat_out_data,
    output                               busy,
    output                               done
);

    localparam [`CNN_ADDR_W-1:0] ADDR_ZERO = {`CNN_ADDR_W{1'b0}};
    localparam integer FLAT_STAGE_CH = STAGE_COMPLEX_CH << 1;

    localparam integer S1_LEN = INPUT_LEN;
    localparam integer S2_LEN = S1_LEN >> 1;
    localparam integer S3_LEN = S2_LEN >> 1;
    localparam integer S4_LEN = S3_LEN >> 1;
    localparam integer S5_LEN = S4_LEN >> 1;
    localparam integer S6_LEN = S5_LEN >> 1;
    localparam integer S7_LEN = S6_LEN >> 1;
    localparam integer S8_LEN = S7_LEN >> 1;
    localparam integer S9_LEN = S8_LEN >> 1;

    localparam integer L1_W_BASE = 0;
    localparam integer L1_W_NUM  = STAGE_COMPLEX_CH * INPUT_COMPLEX_CH * KERNEL;
    localparam integer L2_W_BASE = L1_W_BASE + L1_W_NUM;
    localparam integer L2_W_NUM  = STAGE_COMPLEX_CH * STAGE_COMPLEX_CH * KERNEL;
    localparam integer L3_W_BASE = L2_W_BASE + L2_W_NUM;
    localparam integer L4_W_BASE = L3_W_BASE + L2_W_NUM;
    localparam integer L5_W_BASE = L4_W_BASE + L2_W_NUM;
    localparam integer L6_W_BASE = L5_W_BASE + L2_W_NUM;
    localparam integer L7_W_BASE = L6_W_BASE + L2_W_NUM;
    localparam integer L8_W_BASE = L7_W_BASE + L2_W_NUM;
    localparam integer L9_W_BASE = L8_W_BASE + L2_W_NUM;
    localparam integer CONV_W_DEPTH = L9_W_BASE + L2_W_NUM;

    localparam integer L1_B_BASE = 0;
    localparam integer L2_B_BASE = L1_B_BASE + STAGE_COMPLEX_CH;
    localparam integer L3_B_BASE = L2_B_BASE + STAGE_COMPLEX_CH;
    localparam integer L4_B_BASE = L3_B_BASE + STAGE_COMPLEX_CH;
    localparam integer L5_B_BASE = L4_B_BASE + STAGE_COMPLEX_CH;
    localparam integer L6_B_BASE = L5_B_BASE + STAGE_COMPLEX_CH;
    localparam integer L7_B_BASE = L6_B_BASE + STAGE_COMPLEX_CH;
    localparam integer L8_B_BASE = L7_B_BASE + STAGE_COMPLEX_CH;
    localparam integer L9_B_BASE = L8_B_BASE + STAGE_COMPLEX_CH;
    localparam integer CONV_B_DEPTH = L9_B_BASE + STAGE_COMPLEX_CH;

    localparam ST_IDLE       = 4'd0;
    localparam ST_CONV_START = 4'd1;
    localparam ST_CONV_WAIT  = 4'd2;
    localparam ST_BN_START   = 4'd3;
    localparam ST_BN_WAIT    = 4'd4;
    localparam ST_POOL_START = 4'd5;
    localparam ST_POOL_WAIT  = 4'd6;
    localparam ST_DONE       = 4'd7;

    reg [3:0] state;
    reg [3:0] stage_idx;

    (* DONT_TOUCH = "TRUE", KEEP = "TRUE" *) wire conv_active = (state == ST_CONV_START) || (state == ST_CONV_WAIT);
    (* DONT_TOUCH = "TRUE", KEEP = "TRUE" *) wire bn_active   = (state == ST_BN_START)   || (state == ST_BN_WAIT);
    (* DONT_TOUCH = "TRUE", KEEP = "TRUE" *) wire pool_active = (state == ST_POOL_START) || (state == ST_POOL_WAIT);

    wire [`CNN_ADDR_W-1:0] mem0_rd_addr;
    wire [`CNN_ADDR_W-1:0] mem1_rd_addr;
    wire [`CNN_ADDR_W-1:0] mem2_rd_addr;
    wire signed [`CNN_FEAT_W-1:0] mem0_rd_data;
    wire signed [`CNN_FEAT_W-1:0] mem1_rd_data;
    wire signed [`CNN_FEAT_W-1:0] mem2_rd_data;
    wire mem0_wr_en, mem1_wr_en, mem2_wr_en;
    wire [`CNN_ADDR_W-1:0] mem0_wr_addr, mem1_wr_addr, mem2_wr_addr;
    wire signed [`CNN_FEAT_W-1:0] mem0_wr_data, mem1_wr_data, mem2_wr_data;
    wire signed [`CNN_FEAT_W-1:0] load_data_ext;

    wire [`CNN_ADDR_W-1:0] conv_rd_addr [0:8];
    wire                   conv_wr_en   [0:8];
    wire [`CNN_ADDR_W-1:0] conv_wr_addr [0:8];
    wire signed [`CNN_FEAT_W-1:0] conv_wr_data [0:8];
    wire [`CNN_WADDR_W-1:0] conv_weight_addr [0:8];
    wire [`CNN_BADDR_W-1:0] conv_bias_addr [0:8];
    (* DONT_TOUCH = "TRUE", KEEP = "TRUE" *) wire conv_done [0:8];

    wire [`CNN_ADDR_W-1:0] bn_rd_addr [0:8];
    wire                   bn_wr_en   [0:8];
    wire [`CNN_ADDR_W-1:0] bn_wr_addr [0:8];
    wire signed [`CNN_FEAT_W-1:0] bn_wr_data [0:8];
    (* DONT_TOUCH = "TRUE", KEEP = "TRUE" *) wire bn_done [0:8];

    wire [`CNN_ADDR_W-1:0] pool_rd_addr [0:8];
    wire                   pool_wr_en   [0:8];
    wire [`CNN_ADDR_W-1:0] pool_wr_addr [0:8];
    wire signed [`CNN_FEAT_W-1:0] pool_wr_data [0:8];
    (* DONT_TOUCH = "TRUE", KEEP = "TRUE" *) wire pool_done [0:8];

    wire [`CNN_WADDR_W-1:0] weight_addr = conv_active ? conv_weight_addr[stage_idx] : {`CNN_WADDR_W{1'b0}};
    wire [`CNN_BADDR_W-1:0] bias_addr   = conv_active ? conv_bias_addr[stage_idx]   : {`CNN_BADDR_W{1'b0}};
    wire signed [`CNN_DATA_W-1:0] weight_real_data;
    wire signed [`CNN_DATA_W-1:0] weight_imag_data;
    wire signed [`CNN_ACC_W-1:0]  bias_real_data;
    wire signed [`CNN_ACC_W-1:0]  bias_imag_data;

    assign load_data_ext = $signed(load_data);

    assign busy = (state != ST_IDLE) && (state != ST_DONE);
    assign done = (state == ST_DONE);
    assign feat_out_data = mem0_rd_data;

    assign mem0_rd_addr = conv_active ? conv_rd_addr[stage_idx] : feat_out_addr;
    assign mem1_rd_addr = bn_active   ? bn_rd_addr[stage_idx]   : ADDR_ZERO;
    assign mem2_rd_addr = pool_active ? pool_rd_addr[stage_idx] : ADDR_ZERO;

    assign mem0_wr_en   = ((state == ST_IDLE) && load_en) || (pool_active && pool_wr_en[stage_idx]);
    assign mem0_wr_addr = ((state == ST_IDLE) && load_en) ? load_addr : pool_wr_addr[stage_idx];
    assign mem0_wr_data = ((state == ST_IDLE) && load_en) ? load_data_ext : pool_wr_data[stage_idx];
    assign mem1_wr_en   = conv_active && conv_wr_en[stage_idx];
    assign mem1_wr_addr = conv_wr_addr[stage_idx];
    assign mem1_wr_data = conv_wr_data[stage_idx];
    assign mem2_wr_en   = bn_active && bn_wr_en[stage_idx];
    assign mem2_wr_addr = bn_wr_addr[stage_idx];
    assign mem2_wr_data = bn_wr_data[stage_idx];

    (* DONT_TOUCH = "TRUE" *) cnn1d_feature_ram_fpga_mix12 #(.DATA_W(`CNN_FEAT_W), .DEPTH(RAM_DEPTH)) u_mem0 (
        .clk(clk), .wr_en(mem0_wr_en), .wr_addr(mem0_wr_addr), .wr_data(mem0_wr_data), .rd_addr(mem0_rd_addr), .rd_data(mem0_rd_data)
    );

    (* DONT_TOUCH = "TRUE" *) cnn1d_feature_ram_fpga_mix12 #(.DATA_W(`CNN_FEAT_W), .DEPTH(RAM_DEPTH)) u_mem1 (
        .clk(clk), .wr_en(mem1_wr_en), .wr_addr(mem1_wr_addr), .wr_data(mem1_wr_data), .rd_addr(mem1_rd_addr), .rd_data(mem1_rd_data)
    );

    (* DONT_TOUCH = "TRUE" *) cnn1d_feature_ram_fpga_mix12 #(.DATA_W(`CNN_FEAT_W), .DEPTH(RAM_DEPTH)) u_mem2 (
        .clk(clk), .wr_en(mem2_wr_en), .wr_addr(mem2_wr_addr), .wr_data(mem2_wr_data), .rd_addr(mem2_rd_addr), .rd_data(mem2_rd_data)
    );

    (* DONT_TOUCH = "TRUE" *) cnn1d_complex_weight_rom_fpga_mix12 #(
        .CONV_W_DEPTH(CONV_W_DEPTH),
        .CONV_B_DEPTH(CONV_B_DEPTH),
        .LOAD_CONV_W_REAL(LOAD_CONV_W_REAL),
        .LOAD_CONV_W_IMAG(LOAD_CONV_W_IMAG),
        .LOAD_CONV_B_REAL(LOAD_CONV_B_REAL),
        .LOAD_CONV_B_IMAG(LOAD_CONV_B_IMAG),
        .CONV_W_REAL_FILE(CONV_W_REAL_FILE),
        .CONV_W_IMAG_FILE(CONV_W_IMAG_FILE),
        .CONV_B_REAL_FILE(CONV_B_REAL_FILE),
        .CONV_B_IMAG_FILE(CONV_B_IMAG_FILE)
    ) u_weight_rom (
        .weight_addr(weight_addr),
        .bias_addr(bias_addr),
        .weight_real_data(weight_real_data),
        .weight_imag_data(weight_imag_data),
        .bias_real_data(bias_real_data),
        .bias_imag_data(bias_imag_data)
    );

`define INST_STAGE_DEPLOY(IDX, LEN, INCH, ASHIFT, WBASE, BBASE) \
    cnn1d_complex_conv_same_layer_fpga_mix12 #(.IN_LEN(LEN), .IN_COMPLEX_CH(INCH), .OUT_COMPLEX_CH(STAGE_COMPLEX_CH), .KERNEL(KERNEL), .USE_RELU(1), .ACT_SHIFT(ASHIFT), .W_BASE(WBASE), .B_BASE(BBASE)) u_conv``IDX ( \
        .clk(clk), .rst_n(rst_n), .start(state == ST_CONV_START && stage_idx == IDX-1), .in_base(ADDR_ZERO), .out_base(ADDR_ZERO), \
        .feat_rd_addr(conv_rd_addr[IDX-1]), .feat_rd_data(mem0_rd_data), .feat_wr_en(conv_wr_en[IDX-1]), .feat_wr_addr(conv_wr_addr[IDX-1]), .feat_wr_data(conv_wr_data[IDX-1]), \
        .weight_addr(conv_weight_addr[IDX-1]), .bias_addr(conv_bias_addr[IDX-1]), .weight_real_data(weight_real_data), .weight_imag_data(weight_imag_data), \
        .bias_real_data(bias_real_data), .bias_imag_data(bias_imag_data), .busy(), .done(conv_done[IDX-1]) ); \
    cnn1d_complex_bn_sr_folded_core_fpga_mix12 #(.FEAT_LEN(LEN), .COMPLEX_CH(STAGE_COMPLEX_CH), .AFFINE_SHIFT(BN_SHIFT), .GATE_SHIFT(SR_SHIFT), .SR_THRESH(SR_THRESH)) u_bn_sr``IDX ( \
        .clk(clk), .rst_n(rst_n), .start(state == ST_BN_START && stage_idx == IDX-1), .in_base(ADDR_ZERO), .out_base(ADDR_ZERO), \
        .feat_rd_addr(bn_rd_addr[IDX-1]), .feat_rd_data(mem1_rd_data), .feat_wr_en(bn_wr_en[IDX-1]), .feat_wr_addr(bn_wr_addr[IDX-1]), .feat_wr_data(bn_wr_data[IDX-1]), \
        .busy(), .done(bn_done[IDX-1]) ); \
    cnn1d_pool_core_fpga_mix12 #(.IN_LEN(LEN), .CHANNELS(FLAT_STAGE_CH), .REQUANT_EN(((IDX) > WIDE_STAGES) ? 1 : 0), .REQUANT_BITS(REQUANT_BITS)) u_pool``IDX ( \
        .clk(clk), .rst_n(rst_n), .start(state == ST_POOL_START && stage_idx == IDX-1), .in_base(ADDR_ZERO), .out_base(ADDR_ZERO), \
        .feat_rd_addr(pool_rd_addr[IDX-1]), .feat_rd_data(mem2_rd_data), .feat_wr_en(pool_wr_en[IDX-1]), .feat_wr_addr(pool_wr_addr[IDX-1]), .feat_wr_data(pool_wr_data[IDX-1]), \
        .busy(), .done(pool_done[IDX-1]) );

    `INST_STAGE_DEPLOY(1, S1_LEN, INPUT_COMPLEX_CH,  ACT_SHIFT1, L1_W_BASE, L1_B_BASE)
    `INST_STAGE_DEPLOY(2, S2_LEN, STAGE_COMPLEX_CH,  ACT_SHIFT2, L2_W_BASE, L2_B_BASE)
    `INST_STAGE_DEPLOY(3, S3_LEN, STAGE_COMPLEX_CH,  ACT_SHIFT3, L3_W_BASE, L3_B_BASE)
    `INST_STAGE_DEPLOY(4, S4_LEN, STAGE_COMPLEX_CH,  ACT_SHIFT4, L4_W_BASE, L4_B_BASE)
    `INST_STAGE_DEPLOY(5, S5_LEN, STAGE_COMPLEX_CH,  ACT_SHIFT5, L5_W_BASE, L5_B_BASE)
    `INST_STAGE_DEPLOY(6, S6_LEN, STAGE_COMPLEX_CH,  ACT_SHIFT6, L6_W_BASE, L6_B_BASE)
    `INST_STAGE_DEPLOY(7, S7_LEN, STAGE_COMPLEX_CH,  ACT_SHIFT7, L7_W_BASE, L7_B_BASE)
    `INST_STAGE_DEPLOY(8, S8_LEN, STAGE_COMPLEX_CH,  ACT_SHIFT8, L8_W_BASE, L8_B_BASE)
    `INST_STAGE_DEPLOY(9, S9_LEN, STAGE_COMPLEX_CH,  ACT_SHIFT9, L9_W_BASE, L9_B_BASE)

`undef INST_STAGE_DEPLOY

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            stage_idx <= 4'd0;
        end else begin
            case (state)
                ST_IDLE: begin
                    if (start) begin
                        stage_idx <= 4'd0;
                        state <= ST_CONV_START;
                    end
                end
                ST_CONV_START: state <= ST_CONV_WAIT;
                ST_BN_START:   state <= ST_BN_WAIT;
                ST_POOL_START: state <= ST_POOL_WAIT;
                ST_CONV_WAIT: if (conv_done[stage_idx]) state <= ST_BN_START;
                ST_BN_WAIT:   if (bn_done[stage_idx])   state <= ST_POOL_START;
                ST_POOL_WAIT: begin
                    if (pool_done[stage_idx]) begin
                        if (stage_idx == 4'd8) state <= ST_DONE;
                        else begin
                            stage_idx <= stage_idx + 1'b1;
                            state <= ST_CONV_START;
                        end
                    end
                end
                ST_DONE: state <= ST_IDLE;
                default: state <= ST_IDLE;
            endcase
        end
    end

endmodule
