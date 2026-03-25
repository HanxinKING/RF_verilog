`timescale 1ns/1ps
`include "cnn1d_defs.vh"

// ============================================================
// cnn1d_gap_fc_core testbench
// ------------------------------------------------------------
// 验证目标：
// 1. GAP 对各通道求和并右移是否正确
// 2. FC 地址生成是否正确
// 3. 最终 argmax 分类是否正确
//
// 测试配置：
// - IN_LEN = 2
// - CHANNELS = 2
// - NUM_CLASSES = 3
// - GAP_SHIFT = 0
//
// 输入特征：
//   ch0 = [1, 2] -> GAP = 3
//   ch1 = [4, 5] -> GAP = 9
//
// FC:
//   class0: [1, 0], bias = 0  -> 3
//   class1: [0, 1], bias = 0  -> 9
//   class2: [1, 1], bias = -1 -> 11
//
// 最终结果：
//   class = 2
//   score = 11
// ============================================================

module cnn1d_gap_fc_core_tb;

    localparam integer TB_IN_LEN      = 2;
    localparam integer TB_CHANNELS    = 2;
    localparam integer TB_NUM_CLASSES = 3;
    localparam integer TB_GAP_SHIFT   = 0;

    reg                              clk;
    reg                              rst_n;
    reg                              start;
    wire [`CNN_ADDR_W-1:0]           feat_rd_addr;
    wire signed [`CNN_DATA_W-1:0]    feat_rd_data;
    wire [1:0]                       weight_layer_sel;
    wire [10:0]                      weight_addr;
    wire [7:0]                       bias_addr;
    wire signed [`CNN_DATA_W-1:0]    weight_data;
    wire signed [`CNN_ACC_W-1:0]     bias_data;
    wire                             class_valid;
    wire [`CNN_CLASS_W-1:0]          class_idx;
    wire signed [`CNN_ACC_W-1:0]     class_score;
    wire                             busy;
    wire                             done;

    reg signed [`CNN_DATA_W-1:0] in_mem     [0:TB_IN_LEN*TB_CHANNELS-1];
    reg signed [`CNN_DATA_W-1:0] weight_mem [0:TB_NUM_CLASSES*TB_CHANNELS-1];
    reg signed [`CNN_ACC_W-1:0]  bias_mem   [0:TB_NUM_CLASSES-1];

    integer i;

    assign feat_rd_data = in_mem[feat_rd_addr];
    assign weight_data  = (weight_layer_sel == 2'd3) ? weight_mem[weight_addr] : {`CNN_DATA_W{1'b0}};
    assign bias_data    = (weight_layer_sel == 2'd3) ? bias_mem[bias_addr]     : {`CNN_ACC_W{1'b0}};

    cnn1d_gap_fc_core #(
        .IN_LEN(TB_IN_LEN),
        .CHANNELS(TB_CHANNELS),
        .NUM_CLASSES(TB_NUM_CLASSES),
        .GAP_SHIFT(TB_GAP_SHIFT)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .in_base({`CNN_ADDR_W{1'b0}}),
        .feat_rd_addr(feat_rd_addr),
        .feat_rd_data(feat_rd_data),
        .weight_layer_sel(weight_layer_sel),
        .weight_addr(weight_addr),
        .bias_addr(bias_addr),
        .weight_data(weight_data),
        .bias_data(bias_data),
        .class_valid(class_valid),
        .class_idx(class_idx),
        .class_score(class_score),
        .busy(busy),
        .done(done)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    initial begin
        rst_n = 1'b0;
        start = 1'b0;

        // 输入特征，按 [channel][position] 展平
        in_mem[0] = 8'sd1;
        in_mem[1] = 8'sd2;
        in_mem[2] = 8'sd4;
        in_mem[3] = 8'sd5;

        // class0: [1,0]
        weight_mem[0] = 8'sd1;
        weight_mem[1] = 8'sd0;

        // class1: [0,1]
        weight_mem[2] = 8'sd0;
        weight_mem[3] = 8'sd1;

        // class2: [1,1]
        weight_mem[4] = 8'sd1;
        weight_mem[5] = 8'sd1;

        bias_mem[0] = 32'sd0;
        bias_mem[1] = 32'sd0;
        bias_mem[2] = -32'sd1;

        repeat (4) @(posedge clk);
        rst_n = 1'b1;

        @(posedge clk);
        start = 1'b1;
        @(posedge clk);
        start = 1'b0;

        repeat (200) begin
            @(posedge clk);
            if (done) begin
                if (class_idx !== 8'd2 || class_score !== 32'sd11) begin
                    $display("TEST FAIL: cnn1d_gap_fc_core result mismatch");
                    $display("class_idx=%0d class_score=%0d", class_idx, class_score);
                    $finish;
                end

                $display("TEST PASS: cnn1d_gap_fc_core");
                $finish;
            end
        end

        $display("TEST FAIL: cnn1d_gap_fc_core timeout");
        $finish;
    end

endmodule
