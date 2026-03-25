`ifndef CNN1D_DEFS_FPGA_MIX12_VH
`define CNN1D_DEFS_FPGA_MIX12_VH

// ============================================================
// FPGA mix12 专用位宽配置
// ------------------------------------------------------------
// 当前部署配置：
// - data        : 12 bit
// - acc         : 48 bit
// - BN param    : 32 bit
// - SR gate     : 8 bit
// - conv1~9 w   : 12 bit
// - dense1 w    : 8 bit
// - dense2 w    : 16 bit
// ============================================================

`define CNN_DATA_W             12
`define CNN_ACC_W              48
`define CNN_FEAT_W             `CNN_ACC_W
`define CNN_ADDR_W             18
`define CNN_CLASS_W            8
`define CNN_WADDR_W            21
`define CNN_BADDR_W            12

`define CNN_NUM_CLASSES        16
`define CNN_FEATURE_RAM_DEPTH  140000

`define CNN_MIX12_BN_W         32
`define CNN_MIX12_SR_W         8
`define CNN_MIX12_DENSE1_W_W   8
`define CNN_MIX12_DENSE2_W_W   16

`endif
