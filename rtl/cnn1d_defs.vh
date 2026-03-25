`ifndef CNN1D_DEFS_VH
`define CNN1D_DEFS_VH

// ============================================================
// 全局默认参数
// ------------------------------------------------------------
// 这份头文件提供工程默认位宽和默认三层网络参数。
// 现在工程已经支持通过 parameter 覆盖这些默认值。
//
// 说明：
// - CNN_ADDR_W 现在提升到 18 位，足够覆盖 1000 x 128 级别特征图
// - CNN_WADDR_W / CNN_BADDR_W 用于更大模型的参数寻址
// - CNN_FEATURE_RAM_DEPTH 提升后可以兼容九层特征提取结构
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
