`timescale 1ns/1ps
`include "cnn1d_defs_fpga_mix12.vh"

// ============================================================
// 实值向量 SR 门控模块
// ------------------------------------------------------------
// 这个模块用于部署 dense1 之后的 sr10：
//
//   y[idx] = sat8( (x[idx] * gate[idx]) >>> GATE_SHIFT )
//
// 说明：
// - 输入是实值向量，不再是复值通道对
// - gate 由 Python 侧导出的 sr10 gamma 量化后得到
// - 这里不再做阈值判断，因为导出后的 sr10 gate 已经是
//   推理阶段真正使用的离散门控值
//
// 数据布局：
// - 输入输出都按一维向量连续存储
// - addr = base + idx
// ============================================================

(* KEEP_HIERARCHY = "TRUE" *)
module cnn1d_vector_sr_core_fpga_mix12 #(
    parameter integer VEC_LEN     = 1024,
    parameter integer GATE_SHIFT  = 0,
    parameter integer LOAD_GATE   = 0,
    parameter GATE_FILE           = ""
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

    localparam S_IDLE  = 3'd0;
    localparam S_REQ   = 3'd1;
    localparam S_EXEC  = 3'd2;
    localparam S_WRITE = 3'd3;
    localparam S_NEXT  = 3'd4;
    localparam S_DONE  = 3'd5;

    reg [2:0] state;
    reg [10:0] idx_cnt;

    reg signed [`CNN_MIX12_SR_W-1:0] gate_mem [0:VEC_LEN-1];
    reg signed [`CNN_FEAT_W-1:0] out_reg;

    integer idx;

    wire signed [(`CNN_FEAT_W+`CNN_MIX12_SR_W):0] gated_value;

    assign gated_value = $signed(feat_rd_data) * $signed(gate_mem[idx_cnt]);

    assign busy = (state != S_IDLE) && (state != S_DONE);
    assign done = (state == S_DONE);

    function signed [`CNN_FEAT_W-1:0] sat_feat_from_full;
        input signed [(`CNN_FEAT_W+`CNN_MIX12_SR_W):0] value;
        begin
            if (value > $signed({1'b0, {(`CNN_ACC_W-1){1'b1}}})) begin
                sat_feat_from_full = {1'b0, {(`CNN_ACC_W-1){1'b1}}};
            end else if (value < $signed({1'b1, {(`CNN_ACC_W-1){1'b0}}})) begin
                sat_feat_from_full = {1'b1, {(`CNN_ACC_W-1){1'b0}}};
            end else begin
                sat_feat_from_full = value[`CNN_ACC_W-1:0];
            end
        end
    endfunction

    initial begin
        for (idx = 0; idx < VEC_LEN; idx = idx + 1) begin
            gate_mem[idx] = 8'sd1;
        end

        if (LOAD_GATE != 0) begin
            $display("INFO: loading VECTOR_SR gate from %s", GATE_FILE);
            $readmemh(GATE_FILE, gate_mem);
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state        <= S_IDLE;
            idx_cnt      <= 11'd0;
            out_reg      <= {`CNN_FEAT_W{1'b0}};
            feat_rd_addr <= {`CNN_ADDR_W{1'b0}};
            feat_wr_en   <= 1'b0;
            feat_wr_addr <= {`CNN_ADDR_W{1'b0}};
            feat_wr_data <= {`CNN_FEAT_W{1'b0}};
        end else begin
            feat_wr_en <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (start) begin
                        idx_cnt <= 11'd0;
                        state   <= S_REQ;
                    end
                end

                S_REQ: begin
                    feat_rd_addr <= in_base + idx_cnt;
                    state        <= S_EXEC;
                end

                S_EXEC: begin
                    out_reg <= sat_feat_from_full($signed(gated_value) >>> GATE_SHIFT);
                    state   <= S_WRITE;
                end

                S_WRITE: begin
                    feat_wr_en   <= 1'b1;
                    feat_wr_addr <= out_base + idx_cnt;
                    feat_wr_data <= out_reg;
                    state        <= S_NEXT;
                end

                S_NEXT: begin
                    if (idx_cnt == VEC_LEN - 1) begin
                        state <= S_DONE;
                    end else begin
                        idx_cnt <= idx_cnt + 1'b1;
                        state   <= S_REQ;
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
