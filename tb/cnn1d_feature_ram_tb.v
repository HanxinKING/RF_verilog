`timescale 1ns/1ps
`include "cnn1d_defs.vh"

// ============================================================
// cnn1d_feature_ram testbench
// ------------------------------------------------------------
// 验证目标：
// 1. 写使能有效时，数据能在时钟上升沿写入指定地址
// 2. 组合读口能正确读回指定地址的数据
// ============================================================

module cnn1d_feature_ram_tb;

    reg                       clk;
    reg                       wr_en;
    reg  [`CNN_ADDR_W-1:0]    wr_addr;
    reg  signed [`CNN_DATA_W-1:0] wr_data;
    reg  [`CNN_ADDR_W-1:0]    rd_addr;
    wire signed [`CNN_DATA_W-1:0] rd_data;

    cnn1d_feature_ram #(
        .DATA_W(`CNN_DATA_W),
        .ADDR_W(`CNN_ADDR_W),
        .DEPTH(64)
    ) dut (
        .clk(clk),
        .wr_en(wr_en),
        .wr_addr(wr_addr),
        .wr_data(wr_data),
        .rd_addr(rd_addr),
        .rd_data(rd_data)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    initial begin
        wr_en   = 1'b0;
        wr_addr = 0;
        wr_data = 0;
        rd_addr = 0;

        // 写地址 3
        @(posedge clk);
        wr_en   = 1'b1;
        wr_addr = 3;
        wr_data = 8'sd25;

        // 写地址 5
        @(posedge clk);
        wr_addr = 5;
        wr_data = -8'sd7;

        @(posedge clk);
        wr_en = 1'b0;

        // 读回地址 3
        rd_addr = 3;
        #1;
        if (rd_data !== 8'sd25) begin
            $display("TEST FAIL: RAM readback mismatch at addr 3, got %0d", rd_data);
            $finish;
        end

        // 读回地址 5
        rd_addr = 5;
        #1;
        if (rd_data !== -8'sd7) begin
            $display("TEST FAIL: RAM readback mismatch at addr 5, got %0d", rd_data);
            $finish;
        end

        $display("TEST PASS: cnn1d_feature_ram");
        $finish;
    end

endmodule
