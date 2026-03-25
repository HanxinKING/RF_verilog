`timescale 1ns/1ps
`include "cnn1d_defs.vh"

// Smoke test for the 9-layer feature extractor.
// The test uses:
// - INPUT_LEN = 512
// - INPUT_CH  = 2
// - STAGE_CH  = 4
// - 9 stages of same-conv + bn/sr(identity) + maxpool
//
// Only channel 0 and channel 1 are propagated through the network.
// After 9 rounds of maxpool, the final feature should become the global max
// of each original input channel.

module cnn1d_feature_extractor9_top_tb;

    localparam integer TB_INPUT_LEN = 512;
    localparam integer TB_INPUT_CH  = 2;
    localparam integer TB_STAGE_CH  = 4;
    localparam integer TB_KERNEL    = 3;

    reg                              clk;
    reg                              rst_n;
    reg                              start;
    reg                              load_en;
    reg  [`CNN_ADDR_W-1:0]           load_addr;
    reg  signed [`CNN_DATA_W-1:0]    load_data;
    reg  [`CNN_ADDR_W-1:0]           feat_out_addr;
    wire signed [`CNN_DATA_W-1:0]    feat_out_data;
    wire                             busy;
    wire                             done;

    reg signed [`CNN_DATA_W-1:0] sample0 [0:TB_INPUT_LEN-1];
    reg signed [`CNN_DATA_W-1:0] sample1 [0:TB_INPUT_LEN-1];

    integer i;
    integer base_addr;

    cnn1d_feature_extractor9_top #(
        .INPUT_LEN(TB_INPUT_LEN),
        .INPUT_CH(TB_INPUT_CH),
        .STAGE_CH(TB_STAGE_CH),
        .KERNEL(TB_KERNEL),
        .ACT_SHIFT(0),
        .BN_SHIFT(0),
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

    task init_weights;
        begin
            // clear weights/biases
            for (i = 0; i < 408; i = i + 1) begin
                dut.u_weight_rom.conv_w_mem[i] = 0;
            end
            for (i = 0; i < 36; i = i + 1) begin
                dut.u_weight_rom.conv_b_mem[i] = 0;
            end

            // stage1: oc0 <- ic0(center), oc1 <- ic1(center)
            dut.u_weight_rom.conv_w_mem[1]  = 8'sd1;
            dut.u_weight_rom.conv_w_mem[10] = 8'sd1;

            // stages2..9: oc0 <- ic0(center), oc1 <- ic1(center)
            base_addr = 24;
            repeat (8) begin
                dut.u_weight_rom.conv_w_mem[base_addr + 1]  = 8'sd1;
                dut.u_weight_rom.conv_w_mem[base_addr + 16] = 8'sd1;
                base_addr = base_addr + 48;
            end
        end
    endtask

    task load_current_sample;
        begin
            load_en = 1'b1;
            for (i = 0; i < TB_INPUT_LEN; i = i + 1) begin
                @(posedge clk);
                load_addr = i;
                load_data = sample0[i];
            end
            for (i = 0; i < TB_INPUT_LEN; i = i + 1) begin
                @(posedge clk);
                load_addr = TB_INPUT_LEN + i;
                load_data = sample1[i];
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

    task run_case;
        input integer expect0;
        input integer expect1;
        reg got_done;
        integer t;
        begin
            got_done = 1'b0;
            load_current_sample();
            start_once();

            for (t = 0; t < 200000; t = t + 1) begin
                @(posedge clk);
                if (!got_done && done) begin
                    got_done = 1'b1;
                    feat_out_addr = 0;
                    @(posedge clk);
                    if (feat_out_data !== expect0) begin
                        $display("TEST FAIL: feature0 mismatch, got %0d expect %0d", feat_out_data, expect0);
                        $finish;
                    end

                    feat_out_addr = 1;
                    @(posedge clk);
                    if (feat_out_data !== expect1) begin
                        $display("TEST FAIL: feature1 mismatch, got %0d expect %0d", feat_out_data, expect1);
                        $finish;
                    end

                    @(posedge clk);
                end
            end

            if (!got_done) begin
                $display("TEST FAIL: feature_extractor9 timeout");
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
            sample0[i] = 0;
            sample1[i] = 0;
        end

        repeat (5) @(posedge clk);
        rst_n = 1'b1;
        init_weights();

        // case0: ch0 max=11, ch1 max=6
        for (i = 0; i < TB_INPUT_LEN; i = i + 1) begin
            sample0[i] = i % 8;
            sample1[i] = i % 5;
        end
        sample0[123] = 8'sd11;
        sample1[321] = 8'sd6;
        run_case(11, 6);

        // case1: ch0 max=4, ch1 max=13
        for (i = 0; i < TB_INPUT_LEN; i = i + 1) begin
            sample0[i] = i % 4;
            sample1[i] = i % 7;
        end
        sample0[100] = 8'sd4;
        sample1[77]  = 8'sd13;
        run_case(4, 13);

        $display("TEST PASS: cnn1d_feature_extractor9_top");
        $finish;
    end

endmodule
