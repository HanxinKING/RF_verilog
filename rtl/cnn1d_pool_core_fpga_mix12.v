`timescale 1ns/1ps
`include "cnn1d_defs_fpga_mix12.vh"

// ============================================================
// 单层 1D 最大池化模块
// ------------------------------------------------------------
// 当前实现的是最常见、也最省资源的一种 1D 最大池化：
// - kernel = 2
// - stride = 2
// - out[pos] = max(in[2*pos], in[2*pos+1])
//
// 参数说明：
// - IN_LEN   : 输入特征长度
// - CHANNELS : 输入通道数
//
// 输出长度：
// - OUT_LEN = IN_LEN / 2
// - 默认假设 IN_LEN 为偶数
//
// 数据组织方式：
// - 输入和输出都按 [channel][position] 展平存储
//
// 资源风格：
// - 单窗口顺序扫描
// - 每次只处理一个 2 点窗口
// ============================================================

module cnn1d_pool_core_fpga_mix12 #(
    parameter integer IN_LEN       = 4092,
    parameter integer CHANNELS     = 8,
    parameter integer REQUANT_EN   = 0,
    parameter integer REQUANT_BITS = `CNN_DATA_W
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

    // 池化后长度减半
    localparam integer OUT_LEN = IN_LEN >> 1;

    // 状态机：
    // S_REQ0  : 读取窗口第 1 个点
    // S_CAP0  : 保存第 1 个点，并发起第 2 个点读取
    // S_CAP1  : 保存第 2 个点
    // S_WRITE : 写回最大值
    // S_NEXT  : 更新窗口位置/通道计数
    // S_DONE  : 当前层池化结束
    localparam S_IDLE  = 3'd0;
    localparam S_REQ0  = 3'd1;
    localparam S_CAP0  = 3'd2;
    localparam S_CAP1  = 3'd3;
    localparam S_WRITE = 3'd4;
    localparam S_NEXT  = 3'd5;
    localparam S_DONE  = 3'd6;

    reg [2:0] state;

    // 当前通道编号
    reg [7:0]  ch_cnt;

    // 当前输出位置编号
    reg [15:0] pos_cnt;

    // 一个池化窗口内的两个采样点
    reg signed [`CNN_FEAT_W-1:0] sample0;
    reg signed [`CNN_FEAT_W-1:0] sample1;

    assign busy = (state != S_IDLE) && (state != S_DONE);
    assign done = (state == S_DONE);

    // 取两个输入中的较大值
    function signed [`CNN_FEAT_W-1:0] max2;
        input signed [`CNN_FEAT_W-1:0] a;
        input signed [`CNN_FEAT_W-1:0] b;
        begin
            if (a > b) begin
                max2 = a;
            end else begin
                max2 = b;
            end
        end
    endfunction

    function signed [`CNN_FEAT_W-1:0] requant_clip;
        input signed [`CNN_FEAT_W-1:0] value;
        reg signed [REQUANT_BITS-1:0] clipped;
        begin
            if (value > $signed({1'b0, {(REQUANT_BITS-1){1'b1}}})) begin
                clipped = {1'b0, {(REQUANT_BITS-1){1'b1}}};
            end else if (value < $signed({1'b1, {(REQUANT_BITS-1){1'b0}}})) begin
                clipped = {1'b1, {(REQUANT_BITS-1){1'b0}}};
            end else begin
                clipped = value[REQUANT_BITS-1:0];
            end
            requant_clip = {{(`CNN_FEAT_W-REQUANT_BITS){clipped[REQUANT_BITS-1]}}, clipped};
        end
    endfunction

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state        <= S_IDLE;
            ch_cnt       <= 8'd0;
            pos_cnt      <= 16'd0;
            sample0      <= {`CNN_FEAT_W{1'b0}};
            sample1      <= {`CNN_FEAT_W{1'b0}};
            feat_rd_addr <= {`CNN_ADDR_W{1'b0}};
            feat_wr_en   <= 1'b0;
            feat_wr_addr <= {`CNN_ADDR_W{1'b0}};
            feat_wr_data <= {`CNN_FEAT_W{1'b0}};
        end else begin
            feat_wr_en <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (start) begin
                        ch_cnt  <= 8'd0;
                        pos_cnt <= 16'd0;
                        state   <= S_REQ0;
                    end
                end

                S_REQ0: begin
                    // 当前池化窗口第 1 个点：
                    // in_base + ch * IN_LEN + 2 * pos
                    feat_rd_addr <= in_base + (ch_cnt * IN_LEN) + (pos_cnt << 1);
                    state        <= S_CAP0;
                end

                S_CAP0: begin
                    sample0      <= feat_rd_data;

                    // 当前池化窗口第 2 个点：
                    // in_base + ch * IN_LEN + 2 * pos + 1
                    feat_rd_addr <= in_base + (ch_cnt * IN_LEN) + (pos_cnt << 1) + 1'b1;
                    state        <= S_CAP1;
                end

                S_CAP1: begin
                    sample1 <= feat_rd_data;
                    state   <= S_WRITE;
                end

                S_WRITE: begin
                    // 输出仍按 [channel][position] 展平写回
                    feat_wr_en   <= 1'b1;
                    feat_wr_addr <= out_base + (ch_cnt * OUT_LEN) + pos_cnt;
                    feat_wr_data <= (REQUANT_EN != 0) ? requant_clip(max2(sample0, sample1)) : max2(sample0, sample1);
                    state        <= S_NEXT;
                end

                S_NEXT: begin
                    // 先扫完一个通道，再切换到下一个通道
                    if (pos_cnt == OUT_LEN - 1) begin
                        if (ch_cnt == CHANNELS - 1) begin
                            state <= S_DONE;
                        end else begin
                            ch_cnt  <= ch_cnt + 1'b1;
                            pos_cnt <= 16'd0;
                            state   <= S_REQ0;
                        end
                    end else begin
                        pos_cnt <= pos_cnt + 1'b1;
                        state   <= S_REQ0;
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
