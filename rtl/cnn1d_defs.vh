`ifndef CNN1D_DEFS_VH
`define CNN1D_DEFS_VH

// ============================================================
// 全局默认参数
// ============================================================

`define CNN_DATA_W             8
`define CNN_ACC_W              32
`define CNN_ADDR_W             18
`define CNN_CLASS_W            8
`define CNN_WADDR_W            21
`define CNN_BADDR_W            12

`define CNN_NUM_CLASSES        16

`define CNN_INPUT_LEN          16'd4096
`define CNN_INPUT_CH           8'd2

`define CNN_L1_K               8'd5
`define CNN_L1_OUT_CH          8'd8
`define CNN_L1_LEN             16'd4092
`define CNN_L1_POOL_LEN        16'd2046
`define CNN_L1_SHIFT           5'd7

`define CNN_L2_K               8'd3
`define CNN_L2_OUT_CH          8'd16
`define CNN_L2_LEN             16'd2044
`define CNN_L2_POOL_LEN        16'd1022
`define CNN_L2_SHIFT           5'd7

`define CNN_L3_K               8'd3
`define CNN_L3_OUT_CH          8'd32
`define CNN_L3_LEN             16'd1020
`define CNN_L3_SHIFT           5'd7

`define CNN_GAP_SHIFT          5'd10

`define CNN_FEATURE_RAM_DEPTH  262144

`define CNN_L1_W_BASE          19'd0
`define CNN_L2_W_BASE          19'd80
`define CNN_L3_W_BASE          19'd464

`define CNN_L1_B_BASE          11'd0
`define CNN_L2_B_BASE          11'd8
`define CNN_L3_B_BASE          11'd24

`define CNN_TOTAL_CONV_W       2000
`define CNN_TOTAL_CONV_B       56
`define CNN_TOTAL_FC_W         (`CNN_NUM_CLASSES * `CNN_L3_OUT_CH)
`define CNN_TOTAL_FC_B         `CNN_NUM_CLASSES

`endif
