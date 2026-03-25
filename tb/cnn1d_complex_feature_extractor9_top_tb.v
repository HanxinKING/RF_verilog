`timescale 1ns/1ps
`include "cnn1d_defs.vh"

// ============================================================
// cnn1d_complex_feature_extractor9_top testbench
// ------------------------------------------------------------
// 这是复值 9 层特征提取顶层的 smoke test。
//
// 测试思路：
// 1. 把 9 层卷积都配置成“复恒等映射”
//    也就是只保留输出复通道 0，且只取中心 tap，权重 = 1 + j0
// 2. BN/SR 保持默认参数
//    即 scale=1、bias=0、gate=1，相当于恒等变换
// 3. 这样整条链路真正起作用的就只剩下 9 次复值最大池化
//
// 因为复值池化按模平方选最大，所以经过 9 轮池化之后，
// 最终输出应当是原始输入中“模最大”的那个复样本。
// ============================================================

module cnn1d_complex_feature_extractor9_top_tb;

    localparam integer TB_INPUT_LEN        = 512;
    localparam integer TB_INPUT_COMPLEX_CH = 1;
    localparam integer TB_STAGE_COMPLEX_CH = 2;
    localparam integer TB_KERNEL           = 3;

    reg                            clk;
    reg                            rst_n;
    reg                            start;
    reg                            load_en;
    reg  [`CNN_ADDR_W-1:0]         load_addr;
    reg  signed [`CNN_DATA_W-1:0]  load_data;
    reg  [`CNN_ADDR_W-1:0]         feat_out_addr;
    wire signed [`CNN_DATA_W-1:0]  feat_out_data;
    wire                           busy;
    wire                           done;

    reg signed [`CNN_DATA_W-1:0] sample_real [0:TB_INPUT_LEN-1];
    reg signed [`CNN_DATA_W-1:0] sample_imag [0:TB_INPUT_LEN-1];

    integer i;
    integer base_addr;

    cnn1d_complex_feature_extractor9_top #(
        .INPUT_LEN(TB_INPUT_LEN),
        .INPUT_COMPLEX_CH(TB_INPUT_COMPLEX_CH),
        .STAGE_COMPLEX_CH(TB_STAGE_COMPLEX_CH),
        .KERNEL(TB_KERNEL),
        .ACT_SHIFT(0),
        .BN_SHIFT(0),
        .SR_SHIFT(0),
        .SR_THRESH(0),
        .RAM_DEPTH(4096)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .load_en(load_en),
        .load_addr(load_addr),
        .load_data(load_data),
        .feat_out_addr(feat_out_addr),
        .feat_out_data(feat_out_data),
        .busy(busy),
        .done(done)
    );

    initial begin
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    // 手工构造一组“复恒等卷积”参数。
    task init_weights;
        begin
            for (i = 0; i < dut.CONV_W_DEPTH; i = i + 1) begin
                dut.u_weight_rom.conv_w_real_mem[i] = 0;
                dut.u_weight_rom.conv_w_imag_mem[i] = 0;
            end

            for (i = 0; i < dut.CONV_B_DEPTH; i = i + 1) begin
                dut.u_weight_rom.conv_b_real_mem[i] = 0;
                dut.u_weight_rom.conv_b_imag_mem[i] = 0;
            end

            // 第 1 层：oc0 <- ic0，中心 tap = 1 + j0
            dut.u_weight_rom.conv_w_real_mem[1] = 8'sd1;

            // 第 2~9 层：oc0 <- ic0，中心 tap = 1 + j0
            base_addr = 6;
            repeat (8) begin
                dut.u_weight_rom.conv_w_real_mem[base_addr + 1] = 8'sd1;
                base_addr = base_addr + 12;
            end
        end
    endtask

    // 把一帧复输入样本写入 mem0：
    // - 前半段写实部平面
    // - 后半段写虚部平面
    task load_current_sample;
        begin
            load_en = 1'b1;

            for (i = 0; i < TB_INPUT_LEN; i = i + 1) begin
                @(posedge clk);
                load_addr = i;
                load_data = sample_real[i];
            end

            for (i = 0; i < TB_INPUT_LEN; i = i + 1) begin
                @(posedge clk);
                load_addr = TB_INPUT_LEN + i;
                load_data = sample_imag[i];
            end

            @(posedge clk);
            load_en = 1'b0;
            load_addr = 0;
            load_data = 0;
        end
    endtask

    task start_once;
        begin
            @(posedge clk);
            start = 1'b1;
            @(posedge clk);
            start = 1'b0;
        end
    endtask

    // 跑一组测试，并检查最终输出复样本是否符合预期。
    task run_case;
        input integer expect_real;
        input integer expect_imag;
        reg got_done;
        integer t;
        begin
            got_done = 1'b0;
            load_current_sample();
            start_once();

            for (t = 0; t < 500000; t = t + 1) begin
                @(posedge clk);
                if (!got_done && done) begin
                    got_done = 1'b1;

                    feat_out_addr = 0;
                    @(posedge clk);
                    if (feat_out_data !== expect_real) begin
                        $display("TEST FAIL: real feature mismatch, got %0d expect %0d", feat_out_data, expect_real);
                        $finish;
                    end

                    feat_out_addr = 1;
                    @(posedge clk);
                    if (feat_out_data !== expect_imag) begin
                        $display("TEST FAIL: imag feature mismatch, got %0d expect %0d", feat_out_data, expect_imag);
                        $finish;
                    end

                    feat_out_addr = 2;
                    @(posedge clk);
                    if (feat_out_data !== 0) begin
                        $display("TEST FAIL: unused channel real should stay 0, got %0d", feat_out_data);
                        $finish;
                    end

                    feat_out_addr = 3;
                    @(posedge clk);
                    if (feat_out_data !== 0) begin
                        $display("TEST FAIL: unused channel imag should stay 0, got %0d", feat_out_data);
                        $finish;
                    end
                end
            end

            if (!got_done) begin
                $display("TEST FAIL: complex feature_extractor9 timeout");
                $finish;
            end
        end
    endtask

    initial begin
        rst_n = 1'b0;
        start = 1'b0;
        load_en = 1'b0;
        load_addr = 0;
        load_data = 0;
        feat_out_addr = 0;

        for (i = 0; i < TB_INPUT_LEN; i = i + 1) begin
            sample_real[i] = 0;
            sample_imag[i] = 0;
        end

        repeat (5) @(posedge clk);
        rst_n = 1'b1;
        init_weights();

        // case0：最大模值样本位于 index 123，对应 3 + j10
        for (i = 0; i < TB_INPUT_LEN; i = i + 1) begin
            sample_real[i] = (i % 5) - 2;
            sample_imag[i] = 1 - (i % 3);
        end
        sample_real[123] = 8'sd3;
        sample_imag[123] = 8'sd10;
        sample_real[400] = -8'sd7;
        sample_imag[400] = 8'sd1;
        run_case(3, 10);

        // case1：最大模值样本位于 index 255，对应 -12 + j5
        for (i = 0; i < TB_INPUT_LEN; i = i + 1) begin
            sample_real[i] = (i % 7) - 3;
            sample_imag[i] = (i % 4) - 2;
        end
        sample_real[255] = -8'sd12;
        sample_imag[255] = 8'sd5;
        sample_real[300] = 8'sd8;
        sample_imag[300] = 8'sd2;
        run_case(-12, 5);

        $display("TEST PASS: cnn1d_complex_feature_extractor9_top");
        $finish;
    end

endmodule
