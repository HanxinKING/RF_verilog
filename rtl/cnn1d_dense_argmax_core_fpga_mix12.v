`timescale 1ns/1ps
`include "cnn1d_defs_fpga_mix12.vh"

// ============================================================
// Dense + Argmax 分类模块
// ------------------------------------------------------------
// 这个模块用于部署 dense2：
//
//   score[class] = ( bias[class] + sum(x[i] * w[class][i]) ) >>> ACT_SHIFT
//
// 然后在所有类别里做 argmax，输出：
// - class_idx
// - class_score
//
// 这里不做 softmax，原因是：
// - 分类结果只需要 argmax
// - softmax 需要指数和归一化，硬件代价更高
// - 对最终预测类别来说，softmax 前后的 argmax 一致
//
// 权重存储布局：
// - 地址顺序为 [class][feature]
// - weight_addr = class * IN_DIM + feature
// ============================================================

module cnn1d_dense_argmax_core_fpga_mix12 #(
    parameter integer IN_DIM      = 1024,
    parameter integer NUM_CLASSES = 7,
    parameter integer ACT_SHIFT   = 9,
    parameter integer LOAD_FC_W   = 0,
    parameter integer LOAD_FC_B   = 0,
    parameter FC_W_FILE           = "",
    parameter FC_B_FILE           = ""
) (
    input                                 clk,
    input                                 rst_n,
    input                                 start,
    input      [`CNN_ADDR_W-1:0]          in_base,
    output reg [`CNN_ADDR_W-1:0]          feat_rd_addr,
    input signed [`CNN_FEAT_W-1:0]        feat_rd_data,
    output reg                            score_valid,
    output reg [`CNN_CLASS_W-1:0]         score_class_idx,
    output reg signed [`CNN_ACC_W-1:0]    score_class_value,
    output                                class_valid,
    output reg [`CNN_CLASS_W-1:0]         class_idx,
    output reg signed [`CNN_ACC_W-1:0]    class_score,
    output                                busy,
    output                                done
);

    localparam S_IDLE  = 3'd0;
    localparam S_BIAS  = 3'd1;
    localparam S_REQ   = 3'd2;
    localparam S_MAC   = 3'd3;
    localparam S_NEXT  = 3'd4;
    localparam S_DONE  = 3'd5;

    reg [2:0] state;
    reg [7:0] cls_cnt;
    reg [10:0] feat_cnt;
    reg signed [`CNN_ACC_W-1:0] acc_reg;
    reg signed [`CNN_ACC_W-1:0] best_score;
    reg [`CNN_CLASS_W-1:0] best_class;

    reg signed [`CNN_MIX12_DENSE2_W_W-1:0] fc_w_mem [0:(NUM_CLASSES*IN_DIM)-1];
    reg signed [`CNN_ACC_W-1:0]  fc_b_mem [0:NUM_CLASSES-1];

    integer idx;

    wire [`CNN_WADDR_W-1:0] weight_index;
    wire signed [(`CNN_FEAT_W+`CNN_MIX12_DENSE2_W_W+1):0] fc_sum_full;
    wire signed [`CNN_ACC_W-1:0] fc_sum;
    wire signed [`CNN_ACC_W-1:0] score_value;

    assign weight_index = (cls_cnt * IN_DIM) + feat_cnt;
    assign fc_sum_full = $signed(acc_reg) + ($signed(feat_rd_data) * $signed(fc_w_mem[weight_index]));
    assign fc_sum = sat_acc_from_full(fc_sum_full);
    assign score_value = fc_sum >>> ACT_SHIFT;

    function signed [`CNN_ACC_W-1:0] sat_acc_from_full;
        input signed [(`CNN_FEAT_W+`CNN_MIX12_DENSE2_W_W+1):0] value;
        begin
            if (value > $signed({1'b0, {(`CNN_ACC_W-1){1'b1}}})) begin
                sat_acc_from_full = {1'b0, {(`CNN_ACC_W-1){1'b1}}};
            end else if (value < $signed({1'b1, {(`CNN_ACC_W-1){1'b0}}})) begin
                sat_acc_from_full = {1'b1, {(`CNN_ACC_W-1){1'b0}}};
            end else begin
                sat_acc_from_full = value[`CNN_ACC_W-1:0];
            end
        end
    endfunction

    assign class_valid = (state == S_DONE);
    assign busy = (state != S_IDLE) && (state != S_DONE);
    assign done = (state == S_DONE);

    initial begin
        for (idx = 0; idx < NUM_CLASSES*IN_DIM; idx = idx + 1) begin
            fc_w_mem[idx] = {`CNN_MIX12_DENSE2_W_W{1'b0}};
        end
        for (idx = 0; idx < NUM_CLASSES; idx = idx + 1) begin
            fc_b_mem[idx] = {`CNN_ACC_W{1'b0}};
        end

        if (LOAD_FC_W != 0) begin
            $display("INFO: loading DENSE_ARGMAX_FC_W from %s", FC_W_FILE);
            $readmemh(FC_W_FILE, fc_w_mem);
        end

        if (LOAD_FC_B != 0) begin
            $display("INFO: loading DENSE_ARGMAX_FC_B from %s", FC_B_FILE);
            $readmemh(FC_B_FILE, fc_b_mem);
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state      <= S_IDLE;
            cls_cnt    <= 8'd0;
            feat_cnt   <= 11'd0;
            acc_reg    <= {`CNN_ACC_W{1'b0}};
            best_score <= {1'b1, {(`CNN_ACC_W-1){1'b0}}};
            best_class <= {`CNN_CLASS_W{1'b0}};
            class_idx  <= {`CNN_CLASS_W{1'b0}};
            class_score<= {`CNN_ACC_W{1'b0}};
            score_valid <= 1'b0;
            score_class_idx <= {`CNN_CLASS_W{1'b0}};
            score_class_value <= {`CNN_ACC_W{1'b0}};
            feat_rd_addr<= {`CNN_ADDR_W{1'b0}};
        end else begin
            score_valid <= 1'b0;
            case (state)
                S_IDLE: begin
                    if (start) begin
                        cls_cnt    <= 8'd0;
                        feat_cnt   <= 11'd0;
                        best_score <= {1'b1, {(`CNN_ACC_W-1){1'b0}}};
                        best_class <= {`CNN_CLASS_W{1'b0}};
                        state      <= S_BIAS;
                    end
                end

                S_BIAS: begin
                    acc_reg  <= fc_b_mem[cls_cnt];
                    feat_cnt <= 11'd0;
                    state    <= S_REQ;
                end

                S_REQ: begin
                    feat_rd_addr <= in_base + feat_cnt;
                    state        <= S_MAC;
                end

                S_MAC: begin
                    acc_reg <= fc_sum;

                    if (feat_cnt == IN_DIM - 1) begin
                        score_valid       <= 1'b1;
                        score_class_idx   <= cls_cnt[`CNN_CLASS_W-1:0];
                        score_class_value <= score_value;
                        if ((cls_cnt == 8'd0) || (score_value > best_score)) begin
                            best_score <= score_value;
                            best_class <= cls_cnt[`CNN_CLASS_W-1:0];
                        end
                        state <= S_NEXT;
                    end else begin
                        feat_cnt <= feat_cnt + 1'b1;
                        state    <= S_REQ;
                    end
                end

                S_NEXT: begin
                    if (cls_cnt == NUM_CLASSES - 1) begin
                        class_idx   <= best_class;
                        class_score <= best_score;
                        state       <= S_DONE;
                    end else begin
                        cls_cnt <= cls_cnt + 1'b1;
                        state   <= S_BIAS;
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
