`timescale 1ns/1ps
`include "cnn1d_defs_fpga_mix12.vh"

// ============================================================
// 特征图 RAM
// ------------------------------------------------------------
// 这个模块是整套网络里用于存放中间特征图的缓存。
//
// 当前接口风格：
// - 单写口
// - 单读口
// - 写为同步写
// - 读为组合读
//
// 这样写的原因：
// 1. 逻辑最容易看懂
// 2. 地址一给出来，数据就能直接看到，便于仿真联调
// 3. 后续如果要换成真正的 BRAM，再把读口改成同步读即可
//
// 数据组织方式约定：
// - 所有特征图都按 [channel][position] 展平存储
// - 常用地址公式：
//     addr = base + channel * feature_len + position
// ============================================================

module cnn1d_feature_ram_fpga_mix12 #(
    parameter DATA_W = `CNN_FEAT_W,
    parameter ADDR_W = `CNN_ADDR_W,
    parameter DEPTH  = `CNN_FEATURE_RAM_DEPTH
) (
    input                         clk,
    input                         wr_en,
    input      [ADDR_W-1:0]       wr_addr,
    input signed [DATA_W-1:0]     wr_data,
    input      [ADDR_W-1:0]       rd_addr,
    output signed [DATA_W-1:0]    rd_data
);

    // 实际存储阵列
    reg signed [DATA_W-1:0] mem [0:DEPTH-1];

    integer idx;

    initial begin
        // 仿真初始清零，便于观察波形
        for (idx = 0; idx < DEPTH; idx = idx + 1) begin
            mem[idx] = {DATA_W{1'b0}};
        end
    end

    // 同步写：
    // 在时钟上升沿把 wr_data 写到 wr_addr
    always @(posedge clk) begin
        if (wr_en) begin
            mem[wr_addr] <= wr_data;
        end
    end

    // 组合读：
    // rd_addr 改变后，rd_data 立即对应到该地址内容
    assign rd_data = mem[rd_addr];

endmodule
