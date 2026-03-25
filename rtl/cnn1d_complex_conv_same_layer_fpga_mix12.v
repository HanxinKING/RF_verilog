`timescale 1ns/1ps
`include "cnn1d_defs_fpga_mix12.vh"

// ============================================================
// 单层复值 1D SAME 卷积模块
// ------------------------------------------------------------
// 这个模块对应“复值卷积 + same padding + stride=1”。
//
// 一、特征图存储方式
// ------------------------------------------------------------
// 输入输出都按 [channel][position] 展平存储，但一个“复通道”会拆成两个相邻通道：
// - real(ic0), imag(ic0), real(ic1), imag(ic1), ...
//
// 因此访问某个复输入通道 ic 的：
// - 实部地址基址 = in_base + (2*ic)   * IN_LEN
// - 虚部地址基址 = in_base + (2*ic+1) * IN_LEN
//
// 二、复卷积公式
// ------------------------------------------------------------
// 若输入 x = xr + j*xi，权重 w = wr + j*wi，则：
//   y_real += xr * wr - xi * wi
//   y_imag += xr * wi + xi * wr
//
// 三、硬件实现风格
// ------------------------------------------------------------
// 这里沿用你现有工程里“单 MAC、顺序扫描”的写法：
// - 每次只处理一个输出通道、一个输出位置、一个输入通道、一个 tap
// - 每个复 tap 分 3 步完成：
//   1. 读输入实部
//   2. 读输入虚部
//   3. 做一次复乘加
//
// 这种写法资源占用低，时序关系也比较清晰，适合先把功能验证完整。
// ============================================================

module cnn1d_complex_conv_same_layer_fpga_mix12 #(
    parameter integer IN_LEN          = 1000,
    parameter integer IN_COMPLEX_CH   = 1,
    parameter integer OUT_COMPLEX_CH  = 64,
    parameter integer KERNEL          = 3,
    parameter integer USE_RELU        = 0,
    parameter integer ACT_SHIFT       = 0,
    parameter integer W_BASE          = 0,
    parameter integer B_BASE          = 0
) (
    input                                clk,
    input                                rst_n,
    input                                start,
    input      [`CNN_ADDR_W-1:0]         in_base,
    input      [`CNN_ADDR_W-1:0]         out_base,
    output reg [`CNN_ADDR_W-1:0]         feat_rd_addr,
    input signed [`CNN_FEAT_W-1:0]       feat_rd_data,
    output reg                           feat_wr_en,
    output reg [`CNN_ADDR_W-1:0]         feat_wr_addr,
    output reg signed [`CNN_FEAT_W-1:0]  feat_wr_data,
    output reg [`CNN_WADDR_W-1:0]        weight_addr,
    output reg [`CNN_BADDR_W-1:0]        bias_addr,
    input signed [`CNN_DATA_W-1:0]       weight_real_data,
    input signed [`CNN_DATA_W-1:0]       weight_imag_data,
    input signed [`CNN_ACC_W-1:0]        bias_real_data,
    input signed [`CNN_ACC_W-1:0]        bias_imag_data,
    output                               busy,
    output                               done
);

    // same padding 时，左右各补 PAD 个 0。
    localparam integer PAD = KERNEL >> 1;

    // 状态机说明：
    // S_BIAS    : 装载当前输出复通道的复偏置
    // S_REQ_R   : 计算当前 tap 对应的输入实部地址，并同时给出权重地址
    // S_CAP_R   : 捕获输入实部，并发起输入虚部读取
    // S_MAC     : 完成一次复数乘加
    // S_WRITE_R : 写回当前输出点实部
    // S_WRITE_I : 写回当前输出点虚部
    // S_NEXT    : 更新位置 / 输出通道计数
    localparam S_IDLE    = 4'd0;
    localparam S_BIAS    = 4'd1;
    localparam S_REQ_R   = 4'd2;
    localparam S_CAP_R   = 4'd3;
    localparam S_MAC     = 4'd4;
    localparam S_WRITE_R = 4'd5;
    localparam S_WRITE_I = 4'd6;
    localparam S_NEXT    = 4'd7;
    localparam S_DONE    = 4'd8;

    reg [3:0] state;

    // oc_cnt  : 当前输出复通道编号
    // pos_cnt : 当前输出位置
    // ic_cnt  : 当前输入复通道编号
    // k_cnt   : 当前卷积核 tap 编号
    reg [7:0]  oc_cnt;
    reg [15:0] pos_cnt;
    reg [7:0]  ic_cnt;
    reg [7:0]  k_cnt;

    // 复累加器：分别保存当前输出点的实部和虚部部分和。
    reg signed [`CNN_ACC_W-1:0] acc_real_reg;
    reg signed [`CNN_ACC_W-1:0] acc_imag_reg;

    // 由于特征 RAM 只有一个读口，所以先读实部、暂存，再读虚部。
    reg signed [`CNN_FEAT_W-1:0] sample_real_reg;
    reg signed [31:0] sample_idx_reg;
    reg in_range_reg;

    integer sample_idx;

    wire signed [`CNN_FEAT_W-1:0] sample_imag_data;
    wire signed [(`CNN_FEAT_W+`CNN_DATA_W):0] mul_rr_term;
    wire signed [(`CNN_FEAT_W+`CNN_DATA_W):0] mul_ii_term;
    wire signed [(`CNN_FEAT_W+`CNN_DATA_W):0] mul_ri_term;
    wire signed [(`CNN_FEAT_W+`CNN_DATA_W):0] mul_ir_term;
    wire signed [(`CNN_FEAT_W+`CNN_DATA_W+1):0] mac_real_term;
    wire signed [(`CNN_FEAT_W+`CNN_DATA_W+1):0] mac_imag_term;
    wire signed [(`CNN_FEAT_W+`CNN_DATA_W+2):0] acc_real_next_full;
    wire signed [(`CNN_FEAT_W+`CNN_DATA_W+2):0] acc_imag_next_full;
    wire signed [`CNN_ACC_W-1:0] act_real_value;
    wire signed [`CNN_ACC_W-1:0] act_imag_value;

    // same padding 越界时，本拍输入样本按 0 处理。
    assign sample_imag_data = in_range_reg ? feat_rd_data : {`CNN_FEAT_W{1'b0}};

    // 复乘法展开：
    // (xr + jxi) * (wr + jwi)
    // = (xr*wr - xi*wi) + j(xr*wi + xi*wr)
    assign mul_rr_term = $signed(sample_real_reg) * $signed(weight_real_data);
    assign mul_ii_term = $signed(sample_imag_data) * $signed(weight_imag_data);
    assign mul_ri_term = $signed(sample_real_reg) * $signed(weight_imag_data);
    assign mul_ir_term = $signed(sample_imag_data) * $signed(weight_real_data);
    assign mac_real_term = $signed(mul_rr_term) - $signed(mul_ii_term);
    assign mac_imag_term = $signed(mul_ri_term) + $signed(mul_ir_term);
    assign acc_real_next_full = $signed(acc_real_reg) + $signed(mac_real_term);
    assign acc_imag_next_full = $signed(acc_imag_reg) + $signed(mac_imag_term);

    // 卷积累加完成后，通过右移完成定点缩放。
    assign act_real_value = acc_real_reg >>> ACT_SHIFT;
    assign act_imag_value = acc_imag_reg >>> ACT_SHIFT;

    assign busy = (state != S_IDLE) && (state != S_DONE);
    assign done = (state == S_DONE);

    // 把 32 位累加结果裁剪回 int8。
    function signed [`CNN_ACC_W-1:0] sat_acc_from_wide;
        input signed [(`CNN_FEAT_W+`CNN_DATA_W+2):0] value;
        begin
            if (value > $signed({1'b0, {(`CNN_ACC_W-1){1'b1}}})) begin
                sat_acc_from_wide = {1'b0, {(`CNN_ACC_W-1){1'b1}}};
            end else if (value < $signed({1'b1, {(`CNN_ACC_W-1){1'b0}}})) begin
                sat_acc_from_wide = {1'b1, {(`CNN_ACC_W-1){1'b0}}};
            end else begin
                sat_acc_from_wide = value[`CNN_ACC_W-1:0];
            end
        end
    endfunction

    // 复卷积若要对齐 Keras 的 activation='relu'，
    // 则需要对实部/虚部分别做逐元素 ReLU。
    function signed [`CNN_FEAT_W-1:0] sat_feat;
        input signed [`CNN_ACC_W-1:0] value;
        begin
            if (value > $signed({1'b0, {(`CNN_ACC_W-1){1'b1}}})) begin
                sat_feat = {1'b0, {(`CNN_ACC_W-1){1'b1}}};
            end else if (value < $signed({1'b1, {(`CNN_ACC_W-1){1'b0}}})) begin
                sat_feat = {1'b1, {(`CNN_ACC_W-1){1'b0}}};
            end else begin
                sat_feat = value[`CNN_ACC_W-1:0];
            end
        end
    endfunction

    function signed [`CNN_FEAT_W-1:0] relu_sat_feat;
        input signed [`CNN_ACC_W-1:0] value;
        begin
            if (value < 0) begin
                relu_sat_feat = {`CNN_FEAT_W{1'b0}};
            end else begin
                relu_sat_feat = sat_feat(value);
            end
        end
    endfunction

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state           <= S_IDLE;
            oc_cnt          <= 8'd0;
            pos_cnt         <= 16'd0;
            ic_cnt          <= 8'd0;
            k_cnt           <= 8'd0;
            acc_real_reg    <= {`CNN_ACC_W{1'b0}};
            acc_imag_reg    <= {`CNN_ACC_W{1'b0}};
            sample_real_reg <= {`CNN_FEAT_W{1'b0}};
            sample_idx_reg  <= 32'sd0;
            in_range_reg    <= 1'b0;
            feat_rd_addr    <= {`CNN_ADDR_W{1'b0}};
            feat_wr_en      <= 1'b0;
            feat_wr_addr    <= {`CNN_ADDR_W{1'b0}};
            feat_wr_data    <= {`CNN_FEAT_W{1'b0}};
            weight_addr     <= {`CNN_WADDR_W{1'b0}};
            bias_addr       <= {`CNN_BADDR_W{1'b0}};
        end else begin
            feat_wr_en <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (start) begin
                        oc_cnt       <= 8'd0;
                        pos_cnt      <= 16'd0;
                        ic_cnt       <= 8'd0;
                        k_cnt        <= 8'd0;
                        acc_real_reg <= {`CNN_ACC_W{1'b0}};
                        acc_imag_reg <= {`CNN_ACC_W{1'b0}};
                        bias_addr    <= B_BASE[`CNN_BADDR_W-1:0];
                        state        <= S_BIAS;
                    end
                end

                S_BIAS: begin
                    // 一个新的输出点开始前，先把该输出复通道的复偏置装入累加器。
                    acc_real_reg <= bias_real_data;
                    acc_imag_reg <= bias_imag_data;
                    ic_cnt       <= 8'd0;
                    k_cnt        <= 8'd0;
                    state        <= S_REQ_R;
                end

                S_REQ_R: begin
                    // 当前要访问的输入位置：pos + k - PAD。
                    sample_idx = pos_cnt + k_cnt - PAD;
                    sample_idx_reg <= sample_idx;

                    if ((sample_idx >= 0) && (sample_idx < IN_LEN)) begin
                        in_range_reg <= 1'b1;
                        feat_rd_addr <= in_base + ((ic_cnt << 1) * IN_LEN) + sample_idx[`CNN_ADDR_W-1:0];
                    end else begin
                        in_range_reg <= 1'b0;
                        feat_rd_addr <= in_base;
                    end

                    // 权重地址只按“复通道”寻址，实部/虚部在 ROM 中天然同址对齐。
                    weight_addr <= W_BASE + (oc_cnt * IN_COMPLEX_CH * KERNEL) + (ic_cnt * KERNEL) + k_cnt;
                    state       <= S_CAP_R;
                end

                S_CAP_R: begin
                    // 先保存输入实部，再发起输入虚部读取。
                    sample_real_reg <= in_range_reg ? feat_rd_data : {`CNN_FEAT_W{1'b0}};

                    if (in_range_reg) begin
                        feat_rd_addr <= in_base + (((ic_cnt << 1) + 1'b1) * IN_LEN) + sample_idx_reg[`CNN_ADDR_W-1:0];
                    end else begin
                        feat_rd_addr <= in_base;
                    end

                    state <= S_MAC;
                end

                S_MAC: begin
                    // 完成一次复乘加，把当前 tap 的贡献累加到实部/虚部累加器。
                    acc_real_reg <= sat_acc_from_wide(acc_real_next_full);
                    acc_imag_reg <= sat_acc_from_wide(acc_imag_next_full);

                    if ((ic_cnt == IN_COMPLEX_CH - 1) && (k_cnt == KERNEL - 1)) begin
                        state <= S_WRITE_R;
                    end else begin
                        if (k_cnt == KERNEL - 1) begin
                            k_cnt  <= 8'd0;
                            ic_cnt <= ic_cnt + 1'b1;
                        end else begin
                            k_cnt <= k_cnt + 1'b1;
                        end
                        state <= S_REQ_R;
                    end
                end

                S_WRITE_R: begin
                    // 输出也按“实部通道、虚部通道相邻”的方式展开写回。
                    feat_wr_en   <= 1'b1;
                    feat_wr_addr <= out_base + ((oc_cnt << 1) * IN_LEN) + pos_cnt;
                    feat_wr_data <= (USE_RELU != 0) ? relu_sat_feat(act_real_value) : sat_feat(act_real_value);
                    state        <= S_WRITE_I;
                end

                S_WRITE_I: begin
                    feat_wr_en   <= 1'b1;
                    feat_wr_addr <= out_base + (((oc_cnt << 1) + 1'b1) * IN_LEN) + pos_cnt;
                    feat_wr_data <= (USE_RELU != 0) ? relu_sat_feat(act_imag_value) : sat_feat(act_imag_value);
                    state        <= S_NEXT;
                end

                S_NEXT: begin
                    // 先扫完一个输出复通道的所有位置，再切到下一个输出复通道。
                    if (pos_cnt == IN_LEN - 1) begin
                        if (oc_cnt == OUT_COMPLEX_CH - 1) begin
                            state <= S_DONE;
                        end else begin
                            oc_cnt    <= oc_cnt + 1'b1;
                            pos_cnt   <= 16'd0;
                            bias_addr <= B_BASE + oc_cnt + 1'b1;
                            state     <= S_BIAS;
                        end
                    end else begin
                        pos_cnt   <= pos_cnt + 1'b1;
                        bias_addr <= B_BASE + oc_cnt;
                        state     <= S_BIAS;
                    end
                end

                S_DONE: begin
                    state <= S_IDLE;
                end

                default: begin
                    state <= S_IDLE;
                end
            endcase
        end
    end

endmodule
