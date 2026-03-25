`timescale 1ns/1ps
`include "cnn1d_defs_fpga_mix12.vh"

// ============================================================
// Flatten + Dense 核
// ------------------------------------------------------------
// 输入特征按 [channel][position] 展平存储：
//   addr = in_base + channel * FEAT_LEN + pos
//
// 本模块对整个特征图做 flatten，然后计算：
//   y[out_idx] = ReLU( bias[out_idx] + sum(flat[i] * w[out_idx][i]) )
//
// 这是一个顺序 MAC 实现，适合先搭通完整网络结构。
// ============================================================

module cnn1d_flatten_dense_core_fpga_mix12 #(
    parameter integer FEAT_LEN     = 9,
    parameter integer CHANNELS     = 64,
    parameter integer OUT_DIM      = 1024,
    parameter integer ACT_SHIFT    = 7,
    parameter integer LOAD_FC_W    = 0,
    parameter integer LOAD_FC_B    = 0,
    parameter FC_W_FILE            = "",
    parameter FC_B_FILE            = ""
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
    output                               busy,
    output                               done
);

    localparam integer FLAT_DIM = FEAT_LEN * CHANNELS;

    localparam S_IDLE  = 3'd0;
    localparam S_BIAS  = 3'd1;
    localparam S_REQ   = 3'd2;
    localparam S_MAC   = 3'd3;
    localparam S_WRITE = 3'd4;
    localparam S_NEXT  = 3'd5;
    localparam S_DONE  = 3'd6;

    reg [2:0] state;
    reg [10:0] out_cnt;
    reg [`CNN_WADDR_W-1:0] flat_idx;
    reg signed [`CNN_ACC_W-1:0] acc_reg;

    reg signed [`CNN_MIX12_DENSE1_W_W-1:0] fc_w_mem [0:(OUT_DIM*FLAT_DIM)-1];
    reg signed [`CNN_ACC_W-1:0]  fc_b_mem [0:OUT_DIM-1];

    integer idx;

    wire signed [(`CNN_FEAT_W+`CNN_MIX12_DENSE1_W_W):0] mac_term;
    wire signed [(`CNN_FEAT_W+`CNN_MIX12_DENSE1_W_W+1):0] acc_next_full;
    wire signed [`CNN_ACC_W-1:0] act_value;
    wire [`CNN_WADDR_W-1:0]      fc_weight_index;
    wire [`CNN_WADDR_W-1:0]      flat_ch_index;
    wire [`CNN_WADDR_W-1:0]      flat_pos_index;
    wire [`CNN_WADDR_W-1:0]      src_ch_index;
    wire [`CNN_ADDR_W-1:0]       feat_src_addr;

    assign fc_weight_index = out_cnt * FLAT_DIM + flat_idx;
    assign mac_term = $signed(feat_rd_data) * $signed(fc_w_mem[fc_weight_index]);
    assign acc_next_full = $signed(acc_reg) + $signed(mac_term);
    assign act_value = acc_reg >>> ACT_SHIFT;
    assign flat_ch_index = flat_idx / FEAT_LEN;
    assign flat_pos_index = flat_idx % FEAT_LEN;
    assign src_ch_index = (FEAT_LEN == 1 && CHANNELS[0] == 1'b0) ?
                          ((flat_ch_index < (CHANNELS >> 1)) ? (flat_ch_index << 1) :
                           (((flat_ch_index - (CHANNELS >> 1)) << 1) + 1'b1)) :
                          flat_ch_index;
    assign feat_src_addr = in_base + (src_ch_index * FEAT_LEN) + flat_pos_index;

    assign busy = (state != S_IDLE) && (state != S_DONE);
    assign done = (state == S_DONE);

    function signed [`CNN_ACC_W-1:0] sat_acc_from_full;
        input signed [(`CNN_FEAT_W+`CNN_MIX12_DENSE1_W_W+1):0] value;
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

    function signed [`CNN_FEAT_W-1:0] relu_sat_feat;
        input signed [`CNN_ACC_W-1:0] value;
        begin
            if (value < 0) begin
                relu_sat_feat = {`CNN_FEAT_W{1'b0}};
            end else if (value > $signed({1'b0, {(`CNN_ACC_W-1){1'b1}}})) begin
                relu_sat_feat = {1'b0, {(`CNN_ACC_W-1){1'b1}}};
            end else begin
                relu_sat_feat = value[`CNN_ACC_W-1:0];
            end
        end
    endfunction

    initial begin
        for (idx = 0; idx < OUT_DIM*FLAT_DIM; idx = idx + 1) begin
            fc_w_mem[idx] = {`CNN_MIX12_DENSE1_W_W{1'b0}};
        end
        for (idx = 0; idx < OUT_DIM; idx = idx + 1) begin
            fc_b_mem[idx] = {`CNN_ACC_W{1'b0}};
        end

        if (LOAD_FC_W != 0) begin
            $display("INFO: loading FLATTEN_FC_W from %s", FC_W_FILE);
            $readmemh(FC_W_FILE, fc_w_mem);
        end

        if (LOAD_FC_B != 0) begin
            $display("INFO: loading FLATTEN_FC_B from %s", FC_B_FILE);
            $readmemh(FC_B_FILE, fc_b_mem);
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state        <= S_IDLE;
            out_cnt      <= 11'd0;
            flat_idx     <= {`CNN_WADDR_W{1'b0}};
            acc_reg      <= {`CNN_ACC_W{1'b0}};
            feat_rd_addr <= {`CNN_ADDR_W{1'b0}};
            feat_wr_en   <= 1'b0;
            feat_wr_addr <= {`CNN_ADDR_W{1'b0}};
            feat_wr_data <= {`CNN_FEAT_W{1'b0}};
        end else begin
            feat_wr_en <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (start) begin
                        out_cnt  <= 11'd0;
                        flat_idx <= {`CNN_WADDR_W{1'b0}};
                        state    <= S_BIAS;
                    end
                end

                S_BIAS: begin
                    acc_reg  <= fc_b_mem[out_cnt];
                    flat_idx <= {`CNN_WADDR_W{1'b0}};
                    state    <= S_REQ;
                end

                S_REQ: begin
                    feat_rd_addr <= feat_src_addr;
                    state        <= S_MAC;
                end

                S_MAC: begin
                    acc_reg <= sat_acc_from_full(acc_next_full);

                    if (flat_idx == FLAT_DIM - 1) begin
                        state <= S_WRITE;
                    end else begin
                        flat_idx <= flat_idx + 1'b1;
                        state    <= S_REQ;
                    end
                end

                S_WRITE: begin
                    feat_wr_en   <= 1'b1;
                    feat_wr_addr <= out_base + out_cnt;
                    feat_wr_data <= relu_sat_feat(act_value);
                    state        <= S_NEXT;
                end

                S_NEXT: begin
                    if (out_cnt == OUT_DIM - 1) begin
                        state <= S_DONE;
                    end else begin
                        out_cnt <= out_cnt + 1'b1;
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
