`timescale 1ns/1ps
`include "cnn1d_defs_fpga_mix12.vh"

// ============================================================
// 复值 BN 折叠 + SR 门控模块
// ------------------------------------------------------------
// 这个模块面向“从 Keras 推理图折叠后”的复值 BN。
//
// 原始 ComplexBatchNormalization 的推理形式可以写成：
//   y = A * x + b
//
// 其中：
//   x = [x_real, x_imag]^T
//   A = 2x2 实矩阵
//   b = 2x1 实偏置向量
//
// 展开后就是：
//   y_real = a_rr * x_real + a_ri * x_imag + b_real
//   y_imag = a_ir * x_real + a_ii * x_imag + b_imag
//
// 这里把 4 个仿射系数、2 个偏置和 1 个 SR gate
// 都按“每个复通道一组参数”的方式存成内部存储器。
//
// SR 门控位于 BN 折叠仿射之后：
// - 当 |gate| <= SR_THRESH 时，当前复通道输出直接清零
// - 否则输出乘上 gate，并再经 GATE_SHIFT 右移缩放
//
// 与简化版 cnn1d_complex_bn_sr_core 的区别：
// - 这里支持完整的 2x2 复仿射，不再只保留对角项
// - BN 系数位宽使用 32 位，适合承载折叠后的较大系数
// ============================================================

module cnn1d_complex_bn_sr_folded_core_fpga_mix12 #(
    parameter integer FEAT_LEN          = 1000,
    parameter integer COMPLEX_CH        = 64,
    parameter integer AFFINE_SHIFT      = 10,
    parameter integer GATE_SHIFT        = 0,
    parameter integer SR_THRESH         = 0,
    parameter integer LOAD_A_RR         = 0,
    parameter integer LOAD_A_RI         = 0,
    parameter integer LOAD_A_IR         = 0,
    parameter integer LOAD_A_II         = 0,
    parameter integer LOAD_BIAS_REAL    = 0,
    parameter integer LOAD_BIAS_IMAG    = 0,
    parameter integer LOAD_GATE         = 0,
    parameter A_RR_FILE                 = "",
    parameter A_RI_FILE                 = "",
    parameter A_IR_FILE                 = "",
    parameter A_II_FILE                 = "",
    parameter BIAS_REAL_FILE            = "",
    parameter BIAS_IMAG_FILE            = "",
    parameter GATE_FILE                 = ""
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

    localparam S_IDLE    = 4'd0;
    localparam S_REQ_R   = 4'd1;
    localparam S_CAP_R   = 4'd2;
    localparam S_EXEC    = 4'd3;
    localparam S_WRITE_R = 4'd4;
    localparam S_WRITE_I = 4'd5;
    localparam S_NEXT    = 4'd6;
    localparam S_DONE    = 4'd7;

    reg [3:0] state;
    reg [7:0]  ch_cnt;
    reg [15:0] pos_cnt;

    reg signed [`CNN_FEAT_W-1:0] sample_real_reg;
    reg signed [`CNN_FEAT_W-1:0] out_real_reg;
    reg signed [`CNN_FEAT_W-1:0] out_imag_reg;

    reg signed [`CNN_MIX12_BN_W-1:0] a_rr_mem     [0:COMPLEX_CH-1];
    reg signed [`CNN_MIX12_BN_W-1:0] a_ri_mem     [0:COMPLEX_CH-1];
    reg signed [`CNN_MIX12_BN_W-1:0] a_ir_mem     [0:COMPLEX_CH-1];
    reg signed [`CNN_MIX12_BN_W-1:0] a_ii_mem     [0:COMPLEX_CH-1];
    reg signed [`CNN_MIX12_BN_W-1:0] bias_real_mem[0:COMPLEX_CH-1];
    reg signed [`CNN_MIX12_BN_W-1:0] bias_imag_mem[0:COMPLEX_CH-1];
    reg signed [`CNN_MIX12_SR_W-1:0] gate_mem    [0:COMPLEX_CH-1];

    integer idx;

    wire signed [95:0] affine_real_mul;
    wire signed [95:0] affine_imag_mul;
    wire signed [95:0] affine_real_value;
    wire signed [95:0] affine_imag_value;
    wire signed [103:0] affine_bias_real_full;
    wire signed [103:0] affine_bias_imag_full;
    wire signed [`CNN_FEAT_W-1:0] affine_bias_real_clipped;
    wire signed [`CNN_FEAT_W-1:0] affine_bias_imag_clipped;
    wire signed [103:0] gated_real_value;
    wire signed [103:0] gated_imag_value;
    wire [`CNN_MIX12_SR_W-1:0] gate_abs_value;

    assign affine_real_mul =
        ($signed(sample_real_reg) * $signed(a_rr_mem[ch_cnt])) +
        ($signed(feat_rd_data)    * $signed(a_ri_mem[ch_cnt]));
    assign affine_imag_mul =
        ($signed(sample_real_reg) * $signed(a_ir_mem[ch_cnt])) +
        ($signed(feat_rd_data)    * $signed(a_ii_mem[ch_cnt]));
    assign affine_real_value = $signed(affine_real_mul) >>> AFFINE_SHIFT;
    assign affine_imag_value = $signed(affine_imag_mul) >>> AFFINE_SHIFT;

    assign affine_bias_real_full = $signed(affine_real_value) + $signed(bias_real_mem[ch_cnt]);
    assign affine_bias_imag_full = $signed(affine_imag_value) + $signed(bias_imag_mem[ch_cnt]);
    assign affine_bias_real_clipped = sat_feat_from104(affine_bias_real_full);
    assign affine_bias_imag_clipped = sat_feat_from104(affine_bias_imag_full);
    assign gated_real_value = $signed(affine_bias_real_clipped) * $signed(gate_mem[ch_cnt]);
    assign gated_imag_value = $signed(affine_bias_imag_clipped) * $signed(gate_mem[ch_cnt]);
    assign gate_abs_value = gate_mem[ch_cnt][`CNN_MIX12_SR_W-1] ? -gate_mem[ch_cnt] : gate_mem[ch_cnt];

    assign busy = (state != S_IDLE) && (state != S_DONE);
    assign done = (state == S_DONE);

    function signed [`CNN_FEAT_W-1:0] sat_feat_from104;
        input signed [103:0] value;
        begin
            if (value > $signed({1'b0, {(`CNN_ACC_W-1){1'b1}}})) begin
                sat_feat_from104 = {1'b0, {(`CNN_ACC_W-1){1'b1}}};
            end else if (value < $signed({1'b1, {(`CNN_ACC_W-1){1'b0}}})) begin
                sat_feat_from104 = {1'b1, {(`CNN_ACC_W-1){1'b0}}};
            end else begin
                sat_feat_from104 = value[`CNN_ACC_W-1:0];
            end
        end
    endfunction

    initial begin
        for (idx = 0; idx < COMPLEX_CH; idx = idx + 1) begin
            a_rr_mem[idx]      = 32'sd1 <<< AFFINE_SHIFT;
            a_ri_mem[idx]      = 32'sd0;
            a_ir_mem[idx]      = 32'sd0;
            a_ii_mem[idx]      = 32'sd1 <<< AFFINE_SHIFT;
            bias_real_mem[idx] = 32'sd0;
            bias_imag_mem[idx] = 32'sd0;
            gate_mem[idx]      = 8'sd1;
        end

        if (LOAD_A_RR != 0) begin
            $display("INFO: loading folded BN A_RR from %s", A_RR_FILE);
            $readmemh(A_RR_FILE, a_rr_mem);
        end

        if (LOAD_A_RI != 0) begin
            $display("INFO: loading folded BN A_RI from %s", A_RI_FILE);
            $readmemh(A_RI_FILE, a_ri_mem);
        end

        if (LOAD_A_IR != 0) begin
            $display("INFO: loading folded BN A_IR from %s", A_IR_FILE);
            $readmemh(A_IR_FILE, a_ir_mem);
        end

        if (LOAD_A_II != 0) begin
            $display("INFO: loading folded BN A_II from %s", A_II_FILE);
            $readmemh(A_II_FILE, a_ii_mem);
        end

        if (LOAD_BIAS_REAL != 0) begin
            $display("INFO: loading folded BN bias_real from %s", BIAS_REAL_FILE);
            $readmemh(BIAS_REAL_FILE, bias_real_mem);
        end

        if (LOAD_BIAS_IMAG != 0) begin
            $display("INFO: loading folded BN bias_imag from %s", BIAS_IMAG_FILE);
            $readmemh(BIAS_IMAG_FILE, bias_imag_mem);
        end

        if (LOAD_GATE != 0) begin
            $display("INFO: loading SR gate from %s", GATE_FILE);
            $readmemh(GATE_FILE, gate_mem);
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state           <= S_IDLE;
            ch_cnt          <= 8'd0;
            pos_cnt         <= 16'd0;
            sample_real_reg <= {`CNN_FEAT_W{1'b0}};
            out_real_reg    <= {`CNN_FEAT_W{1'b0}};
            out_imag_reg    <= {`CNN_FEAT_W{1'b0}};
            feat_rd_addr    <= {`CNN_ADDR_W{1'b0}};
            feat_wr_en      <= 1'b0;
            feat_wr_addr    <= {`CNN_ADDR_W{1'b0}};
            feat_wr_data    <= {`CNN_FEAT_W{1'b0}};
        end else begin
            feat_wr_en <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (start) begin
                        ch_cnt  <= 8'd0;
                        pos_cnt <= 16'd0;
                        state   <= S_REQ_R;
                    end
                end

                S_REQ_R: begin
                    feat_rd_addr <= in_base + ((ch_cnt << 1) * FEAT_LEN) + pos_cnt;
                    state        <= S_CAP_R;
                end

                S_CAP_R: begin
                    sample_real_reg <= feat_rd_data;
                    feat_rd_addr    <= in_base + (((ch_cnt << 1) + 1'b1) * FEAT_LEN) + pos_cnt;
                    state           <= S_EXEC;
                end

                S_EXEC: begin
                    if (gate_abs_value <= SR_THRESH) begin
                        out_real_reg <= {`CNN_FEAT_W{1'b0}};
                        out_imag_reg <= {`CNN_FEAT_W{1'b0}};
                    end else begin
                        out_real_reg <= sat_feat_from104($signed(gated_real_value) >>> GATE_SHIFT);
                        out_imag_reg <= sat_feat_from104($signed(gated_imag_value) >>> GATE_SHIFT);
                    end
                    state <= S_WRITE_R;
                end

                S_WRITE_R: begin
                    feat_wr_en   <= 1'b1;
                    feat_wr_addr <= out_base + ((ch_cnt << 1) * FEAT_LEN) + pos_cnt;
                    feat_wr_data <= out_real_reg;
                    state        <= S_WRITE_I;
                end

                S_WRITE_I: begin
                    feat_wr_en   <= 1'b1;
                    feat_wr_addr <= out_base + (((ch_cnt << 1) + 1'b1) * FEAT_LEN) + pos_cnt;
                    feat_wr_data <= out_imag_reg;
                    state        <= S_NEXT;
                end

                S_NEXT: begin
                    if (pos_cnt == FEAT_LEN - 1) begin
                        if (ch_cnt == COMPLEX_CH - 1) begin
                            state <= S_DONE;
                        end else begin
                            ch_cnt  <= ch_cnt + 1'b1;
                            pos_cnt <= 16'd0;
                            state   <= S_REQ_R;
                        end
                    end else begin
                        pos_cnt <= pos_cnt + 1'b1;
                        state   <= S_REQ_R;
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
