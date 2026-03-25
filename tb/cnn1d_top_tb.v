`timescale 1ns/1ps
`include "cnn1d_defs.vh"

// ============================================================
// 顶层 testbench
// ------------------------------------------------------------
// 这一版 testbench 使用外部 .mem 参数文件，并且运行多组输入样例。
//
// 与之前单一常数偏置场景不同，这里卷积权重不再全为 0，而是构造了一条
// 简单但真实经过三层 Conv + 两层 Pool 的有效数据通路：
//
// 1. Conv1/Conv2/Conv3 的第 0 个输出通道都只传递“输入通道 0”的数据
// 2. Conv1/Conv2/Conv3 的第 1 个输出通道都只传递“输入通道 1”的数据
// 3. 池化会逐层取局部最大值
// 4. 最终 GAP/FC 会比较两个通道的最终特征大小
//
// 对于当前小网络：
//   Input(20) -> Conv1(18) -> Pool1(9) -> Conv2(7) -> Pool2(3) -> Conv3(1)
//
// 最终两个输出特征大致对应于：
// - 通道 0 前 4 个输入样本中的最大值
// - 通道 1 前 4 个输入样本中的最大值
//
// FC 参数设计为：
// - class0 分数 = feature0
// - class1 分数 = feature1
// - class2, class3 给一个很小的固定偏置
//
// 因此最终分类规则近似为：
// - 如果通道 0 特征更大，则输出 class0
// - 如果通道 1 特征更大，则输出 class1
//
// 这样做的好处：
// - 数据不再是“无论输入什么都一个输出”
// - 仿真里能看到不同输入样本得到不同分类结果
// - 同时仍然便于手工分析和验证
// ============================================================

module cnn1d_top_tb;

    localparam integer TB_NUM_CLASSES = 4;
    localparam integer TB_INPUT_LEN   = 20;
    localparam integer TB_INPUT_CH    = 2;

    localparam integer TB_L1_OUT_CH   = 2;
    localparam integer TB_L1_K        = 3;
    localparam integer TB_L1_SHIFT    = 0;
    localparam integer TB_L1_W_BASE   = 0;
    localparam integer TB_L1_B_BASE   = 0;

    localparam integer TB_L2_OUT_CH   = 2;
    localparam integer TB_L2_K        = 3;
    localparam integer TB_L2_SHIFT    = 0;
    localparam integer TB_L2_W_BASE   = 12;
    localparam integer TB_L2_B_BASE   = 2;

    localparam integer TB_L3_OUT_CH   = 2;
    localparam integer TB_L3_K        = 3;
    localparam integer TB_L3_SHIFT    = 0;
    localparam integer TB_L3_W_BASE   = 24;
    localparam integer TB_L3_B_BASE   = 4;

    localparam integer TB_GAP_SHIFT   = 0;

    reg                              clk;
    reg                              rst_n;
    reg                              start;
    reg                              load_en;
    reg  [`CNN_ADDR_W-1:0]           load_addr;
    reg  signed [`CNN_DATA_W-1:0]    load_data;
    wire                             busy;
    wire                             done;
    wire                             out_valid;
    wire [`CNN_CLASS_W-1:0]          out_class;
    wire signed [`CNN_ACC_W-1:0]     out_score;

    reg signed [`CNN_DATA_W-1:0] sample_ch0 [0:TB_INPUT_LEN-1];
    reg signed [`CNN_DATA_W-1:0] sample_ch1 [0:TB_INPUT_LEN-1];

    integer i;

    cnn1d_top #(
        .NUM_CLASSES(TB_NUM_CLASSES),
        .INPUT_LEN(TB_INPUT_LEN),
        .INPUT_CH(TB_INPUT_CH),
        .L1_OUT_CH(TB_L1_OUT_CH),
        .L1_K(TB_L1_K),
        .L1_SHIFT(TB_L1_SHIFT),
        .L1_W_BASE(TB_L1_W_BASE),
        .L1_B_BASE(TB_L1_B_BASE),
        .L2_OUT_CH(TB_L2_OUT_CH),
        .L2_K(TB_L2_K),
        .L2_SHIFT(TB_L2_SHIFT),
        .L2_W_BASE(TB_L2_W_BASE),
        .L2_B_BASE(TB_L2_B_BASE),
        .L3_OUT_CH(TB_L3_OUT_CH),
        .L3_K(TB_L3_K),
        .L3_SHIFT(TB_L3_SHIFT),
        .L3_W_BASE(TB_L3_W_BASE),
        .L3_B_BASE(TB_L3_B_BASE),
        .GAP_SHIFT(TB_GAP_SHIFT),
        .LOAD_CONV_W(1),
        .LOAD_CONV_B(1),
        .LOAD_FC_W(1),
        .LOAD_FC_B(1),
        .CONV_W_FILE("mem/tb_top_conv_w.mem"),
        .CONV_B_FILE("mem/tb_top_conv_b.mem"),
        .FC_W_FILE("mem/tb_top_fc_w.mem"),
        .FC_B_FILE("mem/tb_top_fc_b.mem")
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .load_en(load_en),
        .load_addr(load_addr),
        .load_data(load_data),
        .busy(busy),
        .done(done),
        .out_valid(out_valid),
        .out_class(out_class),
        .out_score(out_score)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    // ========================================================
    // 按当前 sample_ch0 / sample_ch1 装载输入
    // ========================================================
    task load_current_sample;
        begin
            load_en   = 1'b1;
            load_addr = {`CNN_ADDR_W{1'b0}};
            load_data = {`CNN_DATA_W{1'b0}};

            for (i = 0; i < TB_INPUT_LEN; i = i + 1) begin
                @(posedge clk);
                load_addr = i;
                load_data = sample_ch0[i];
            end

            for (i = 0; i < TB_INPUT_LEN; i = i + 1) begin
                @(posedge clk);
                load_addr = TB_INPUT_LEN + i;
                load_data = sample_ch1[i];
            end

            @(posedge clk);
            load_en   = 1'b0;
            load_addr = {`CNN_ADDR_W{1'b0}};
            load_data = {`CNN_DATA_W{1'b0}};
        end
    endtask

    // ========================================================
    // 发起一次推理
    // ========================================================
    task start_inference;
        begin
            @(posedge clk);
            start = 1'b1;
            @(posedge clk);
            start = 1'b0;
        end
    endtask

    // ========================================================
    // 运行一组样本并检查结果
    // ========================================================
    task run_case;
        input [127:0] case_name;
        input [`CNN_CLASS_W-1:0] expected_class;
        input signed [`CNN_ACC_W-1:0] expected_score;
        reg case_done;
        integer t;
        begin
            case_done = 1'b0;
            load_current_sample();
            start_inference();

            for (t = 0; t < 6000; t = t + 1) begin
                @(posedge clk);
                if (!case_done && done) begin
                    $display("----------------------------------------------");
                    $display("CASE      = %0s", case_name);
                    $display("out_valid = %0d", out_valid);
                    $display("out_class = %0d", out_class);
                    $display("out_score = %0d", out_score);

                    if ((out_class == expected_class) && (out_score == expected_score)) begin
                        $display("CASE PASS");
                    end else begin
                        $display("CASE FAIL");
                        $finish;
                    end

                    // 给顶层状态机一个周期回到 IDLE
                    @(posedge clk);
                    case_done = 1'b1;
                end
            end

            if (!case_done) begin
                $display("CASE FAIL: TIMEOUT");
                $finish;
            end
        end
    endtask

    // ========================================================
    // 构造样本 0
    // --------------------------------------------------------
    // 通道 0 前 4 个点最大值 = 7
    // 通道 1 前 4 个点最大值 = 2
    // 预期 class0, score=7
    // ========================================================
    task set_case0;
        begin
            sample_ch0[0]  = 8'sd1; sample_ch0[1]  = 8'sd7; sample_ch0[2]  = 8'sd2; sample_ch0[3]  = 8'sd3;
            sample_ch1[0]  = 8'sd0; sample_ch1[1]  = 8'sd1; sample_ch1[2]  = 8'sd2; sample_ch1[3]  = 8'sd1;

            for (i = 4; i < TB_INPUT_LEN; i = i + 1) begin
                sample_ch0[i] = i;
                sample_ch1[i] = i - 10;
            end
        end
    endtask

    // ========================================================
    // 构造样本 1
    // --------------------------------------------------------
    // 通道 0 前 4 个点最大值 = 2
    // 通道 1 前 4 个点最大值 = 6
    // 预期 class1, score=6
    // ========================================================
    task set_case1;
        begin
            sample_ch0[0]  = 8'sd1; sample_ch0[1]  = 8'sd2; sample_ch0[2]  = 8'sd1; sample_ch0[3]  = 8'sd0;
            sample_ch1[0]  = 8'sd4; sample_ch1[1]  = 8'sd3; sample_ch1[2]  = 8'sd6; sample_ch1[3]  = 8'sd2;

            for (i = 4; i < TB_INPUT_LEN; i = i + 1) begin
                sample_ch0[i] = 20 - i;
                sample_ch1[i] = i - 5;
            end
        end
    endtask

    // ========================================================
    // 构造样本 2
    // --------------------------------------------------------
    // 通道 0 前 4 个点最大值 = 9
    // 通道 1 前 4 个点最大值 = 5
    // 预期 class0, score=9
    // ========================================================
    task set_case2;
        begin
            sample_ch0[0]  = 8'sd9; sample_ch0[1]  = 8'sd1; sample_ch0[2]  = 8'sd4; sample_ch0[3]  = 8'sd2;
            sample_ch1[0]  = 8'sd2; sample_ch1[1]  = 8'sd5; sample_ch1[2]  = 8'sd1; sample_ch1[3]  = 8'sd0;

            for (i = 4; i < TB_INPUT_LEN; i = i + 1) begin
                sample_ch0[i] = (i % 4);
                sample_ch1[i] = (i % 3);
            end
        end
    endtask

    initial begin
        rst_n     = 1'b0;
        start     = 1'b0;
        load_en   = 1'b0;
        load_addr = {`CNN_ADDR_W{1'b0}};
        load_data = {`CNN_DATA_W{1'b0}};

        repeat (5) @(posedge clk);
        rst_n = 1'b1;

        set_case0();
        run_case("case0", 8'd0, 32'sd7);

        set_case1();
        run_case("case1", 8'd1, 32'sd6);

        set_case2();
        run_case("case2", 8'd0, 32'sd9);

        $display("==============================================");
        $display("ALL TOP-LEVEL READMEMH CASES PASS");
        $display("==============================================");
        $finish;
    end

endmodule
