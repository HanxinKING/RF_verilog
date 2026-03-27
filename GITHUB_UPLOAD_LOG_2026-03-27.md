# GitHub 上传日志

## 时间

2026-03-27

## 上传目的

将 `fpga_mix12` 新流程相关的源码、批量 testbench、结果汇总文档、PPT 支撑材料和图表整理后上传到 GitHub，保留当前阶段的可复现工程状态。

## 本次准备上传的内容

### 1. RTL 与 testbench 源码

- `rtl/cnn1d_complex_classifier_deploy_top_fpga_mix12.v`
- `rtl/cnn1d_complex_feature_extractor9_deploy_top_fpga_mix12.v`
- `rtl/cnn1d_defs.vh`
- `rtl/cnn1d_dense_argmax_core_fpga_mix12.v`
- `rtl/cnn1d_flatten_dense_core_fpga_mix12.v`
- `rtl/cnn1d_vector_sr_core_fpga_mix12.v`
- `tb/cnn1d_complex_classifier_deploy_fpga_mix12_fast_tb.v`
- `tb/cnn1d_complex_classifier_deploy_fpga_mix12_tb.v`
- `tb/cnn1d_complex_classifier_batch_fpga_mix12_tb.v`
- `tb/generated/fpga_mix12_batch_cfg.vh`
- `modelsim_fpga_mix12_tb_batch.f`

### 2. 项目说明与交接文档

- `FPGA_MIX12_HANDOFF_2026-03-27.md`
- `FPGA_MIX12_交接说明_2026-03-27.md`
- `GITHUB_UPLOAD_LOG_2026-03-27.md`

### 3. 结果汇总与清洗后的表格

- `mem/fpga_mix12_batch_7x50/summary_float_fixed.json`
- `mem/fpga_mix12_batch_7x50/summary_final.json`
- `mem/fpga_mix12_batch_7x50/summary_final.md`
- `mem/fpga_mix12_batch_7x50/summary_rtl_vs_float.json`
- `mem/fpga_mix12_batch_7x50/summary_rtl_vs_float.md`
- `mem/fpga_mix12_batch_7x50/rtl_results_float_only_clean.csv`
- `mem/fpga_mix12_batch_7x50/rtl_results.csv`
- `mem/fpga_mix12_batch_7x50/rtl_logits.csv`

### 4. PPT 支撑材料与图表

- `项目汇报PPT大纲.md`
- `项目汇报PPT逐页讲稿.md`
- `整体技术路线流程图.svg`
- `逐层平均绝对误差折线图.svg`
- `stage_cosine_similarity_trend.png`

## 本次结果结论

- RTL 与 Python 定点结果已经达到完全一致。
- 7x50 批量结果已经整理出干净表格，可直接用于最终统计。
- 已生成中文汇总文档，可直接支撑 PPT 撰写与答辩说明。

## 本次未上传的内容

以下文件属于仿真缓存、工程临时产物或体积过大的中间文件，本次不推送到 GitHub：

- `.Xil/`
- `work/`
- `project_1.cache/`
- `project_1.runs/`
- `vivado.log`
- `vivado.jou`
- `transcript`
- `mem/fpga_mix12_batch_7x50/keras_intermediates.npz`
- `mem/fpga_mix12_batch_7x50/batch_samples_float.npy`
- `mem/fpga_mix12_batch_7x50/batch_input_iq.mem`
- 其他仅用于本地仿真或临时调试的大体积文件

## 备注

本次上传遵循“保留核心源码与可复现结果，排除临时缓存和超大中间文件”的原则，避免仓库被本地仿真产物污染。
