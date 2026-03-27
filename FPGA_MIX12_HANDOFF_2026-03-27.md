# FPGA Mix12 Handoff

This document summarizes the current "new flow" for the `fpga_mix12` deployment path, the files that matter, what has already been fixed, and what to watch out for when continuing the work in a new chat/session.

## 1. Goal

The active goal is to validate the 7-class SlimSEI model on the FPGA-oriented RTL path:

- Python float reference
- Python fixed-point reference
- ModelSim RTL result

The intended production path is the `fpga_mix12` path, not the old `CNN_RF` batch path.

## 2. Authoritative Model Source

Python float reference must come from this model only:

- `F:\RF_cnn\SlimSEI\SlimSEI-main\FastPSGD`
- `F:\RF_cnn\SlimSEI\SlimSEI-main\FastPSGD\SparseComplexCNNSameChannel35.hdf5`

The float helper scripts already use that path:

- `F:\RF_cnn\conv1d_verilog\CNN_RF\tools\slimsei64_keras_float_predict.py`
- `F:\RF_cnn\conv1d_verilog\CNN_RF\tools\slimsei64_keras_dump_intermediates.py`

## 3. New Flow vs Old Flow

### New flow (keep using this)

Python / export side:

- `F:\RF_cnn\conv1d_verilog\CNN_RF\tools\slimsei64_flow.py`
- `F:\RF_cnn\conv1d_verilog\CNN_RF\tools\slimsei64_auto_calibrate.py`
- `F:\RF_cnn\conv1d_verilog\CNN_RF\tools\export_fpga_mix12_demo.py`
- `F:\RF_cnn\conv1d_verilog\CNN_RF\tools\export_fpga_mix12_batch.py`
- `F:\RF_cnn\conv1d_verilog\CNN_RF\tools\summarize_fpga_mix12_batch.py`

RTL / ModelSim side:

- `F:\RF_cnn\CONV1_21\project_1\rtl\cnn1d_complex_classifier_deploy_top_fpga_mix12.v`
- `F:\RF_cnn\CONV1_21\project_1\rtl\cnn1d_dense_argmax_core_fpga_mix12.v`
- `F:\RF_cnn\CONV1_21\project_1\tb\cnn1d_complex_classifier_deploy_fpga_mix12_tb.v`
- `F:\RF_cnn\CONV1_21\project_1\tb\cnn1d_complex_classifier_deploy_fpga_mix12_fast_tb.v`
- `F:\RF_cnn\CONV1_21\project_1\tb\cnn1d_complex_classifier_batch_fpga_mix12_tb.v`
- `F:\RF_cnn\CONV1_21\project_1\modelsim_fpga_mix12_rtl.f`
- `F:\RF_cnn\CONV1_21\project_1\modelsim_fpga_mix12_tb_batch.f`
- `F:\RF_cnn\CONV1_21\project_1\run_fpga_mix12_batch.bat`

### Old flow (do not use as the main reference)

These were the older batch-path files and should not be treated as the current truth:

- old batch script in `CNN_RF/tools/slimsei64_batch_validate.py` was removed
- old non-`fpga_mix12` RTL path in `CNN_RF/rtl/*deploy_top.v`
- old batch outputs in `CNN_RF/mem/slimsei64_batch_*`

## 4. Key Fixes Already Applied

### 4.1 Fixed `.mem` export width bug

The important bug was that convolution weights were previously written out with the wrong width in an older path.

The current `fpga_mix12` export path writes mem files with explicit per-tensor widths:

- `F:\RF_cnn\conv1d_verilog\CNN_RF\tools\export_fpga_mix12_demo.py`
- `F:\RF_cnn\conv1d_verilog\CNN_RF\tools\export_fpga_mix12_batch.py`

Relevant helper:

- `write_quantized_mem_with_widths(...)`

### 4.2 Fixed dump testbench loop-variable interference

The single-sample dump testbench was fixed so the stage dump no longer gets corrupted by shared loop-variable interference:

- `F:\RF_cnn\CONV1_21\project_1\tb\cnn1d_complex_classifier_deploy_fpga_mix12_tb.v`

### 4.3 Fixed dense2 per-class shift mismatch

This was a critical bug discovered during the new batch smoke test.

Problem:

- Python fixed used per-class `dense2_shift`
- RTL used one scalar `DENSE2_SHIFT`
- Class 6 score could be wrong / amplified

Fix:

- RTL now loads `dense2_shift.mem`
- dense2 core now supports per-class shift

Files changed:

- `F:\RF_cnn\CONV1_21\project_1\rtl\cnn1d_dense_argmax_core_fpga_mix12.v`
- `F:\RF_cnn\CONV1_21\project_1\rtl\cnn1d_complex_classifier_deploy_top_fpga_mix12.v`
- `F:\RF_cnn\CONV1_21\project_1\tb\cnn1d_complex_classifier_deploy_fpga_mix12_tb.v`
- `F:\RF_cnn\CONV1_21\project_1\tb\cnn1d_complex_classifier_deploy_fpga_mix12_fast_tb.v`
- `F:\RF_cnn\CONV1_21\project_1\tb\cnn1d_complex_classifier_batch_fpga_mix12_tb.v`
- `F:\RF_cnn\conv1d_verilog\CNN_RF\tools\export_fpga_mix12_demo.py`
- `F:\RF_cnn\conv1d_verilog\CNN_RF\tools\export_fpga_mix12_batch.py`

New mem file:

- `dense2_shift.mem`

## 5. Single-Sample Status

Single-sample `fpga_mix12_demo` is aligned and remains the main "known-good" reference.

Directory:

- `F:\RF_cnn\CONV1_21\project_1\mem\fpga_mix12_demo`

Important comparison files:

- `F:\RF_cnn\CONV1_21\project_1\mem\fpga_mix12_demo\fpga_py_fixed_float_18_stage_compare.md`
- `F:\RF_cnn\CONV1_21\project_1\mem\fpga_mix12_demo\fpga_py_fixed_float_18_stage_compare.csv`

Interpretation:

- `FPGA vs Python fixed` for `conv1~pool9` is fully aligned on the demo sample
- `dense1 / sr10 / dense2_logits` fixed-vs-float error is small and expected

## 6. Batch Smoke Status

Directory:

- `F:\RF_cnn\CONV1_21\project_1\mem\fpga_mix12_batch_smoke7`

This is the best current proof that the new chain is working end-to-end.

Important files:

- `summary_float_fixed.json`
- `summary_final.json`
- `rtl_results.csv`
- `rtl_logits.csv`

Meaning:

- `summary_float_fixed.json` = Python float vs Python fixed summary
- `summary_final.json` = float / fixed / RTL combined summary

Expected smoke result:

- `float_fixed_agreement = 1.0`
- `fixed_rtl_agreement = 1.0`
- `float_rtl_agreement = 1.0`
- `mean_abs_fixed_rtl_logit_diff = 0.0`

## 7. 7x50 Batch Directory

Formal batch directory:

- `F:\RF_cnn\CONV1_21\project_1\mem\fpga_mix12_batch_7x50`

Current important files already present:

- `batch_samples_float.npy`
- `keras_float_outputs.npz`
- `keras_intermediates.npz`
- `comparison_arrays.npz`
- `summary_float_fixed.json`
- `batch_input_iq.mem`
- `batch_expect_class.mem`
- `batch_expect_score.mem`
- `dense2_shift.mem`
- `rtl_results.csv`
- `rtl_logits.csv`

## 8. Important Caveat: Mixed Old/New Run Artifacts

The directory `fpga_mix12_batch_7x50` has been written by more than one ModelSim process during debugging.

That means:

- `rtl_results.csv` may contain mixed old/new rows
- some rows are 5-column "new format"
- older rows may still use the older 7-column format

The clean new-format rows look like:

```csv
sample_idx,rtl_class,rtl_score,float_class,class_match
```

The old mixed-format rows looked like:

```csv
sample_idx,rtl_class,rtl_score,expect_class,expect_score,class_match,score_match
```

Because of this, raw line counts in `rtl_results.csv` are not always reliable.

If a clean snapshot is needed, use or regenerate:

- `F:\RF_cnn\CONV1_21\project_1\mem\fpga_mix12_batch_7x50\rtl_results_float_only_clean.csv`

## 9. Current Runtime State at Handoff

At the time of writing this handoff, there are still multiple `vsim / vsimk` processes alive.

That means the next session should begin by checking / stopping old ModelSim jobs before trusting the batch directory.

Typical symptoms of mixed runs:

- `rtl_results.csv` line count > 350
- malformed or half-written final line
- mismatch between `modelsim_stdout.log` progress and csv line count

## 10. Recommended Clean Restart Procedure

If continuing the formal 7x50 run in a new session, the safest sequence is:

1. Stop old `vsim / vsimk` processes.
2. Delete or move old runtime outputs in:
   - `F:\RF_cnn\CONV1_21\project_1\mem\fpga_mix12_batch_7x50\rtl_results.csv`
   - `F:\RF_cnn\CONV1_21\project_1\mem\fpga_mix12_batch_7x50\rtl_logits.csv`
   - `F:\RF_cnn\CONV1_21\project_1\mem\fpga_mix12_batch_7x50\modelsim_stdout*.log`
   - `F:\RF_cnn\CONV1_21\project_1\mem\fpga_mix12_batch_7x50\modelsim_stderr*.log`
3. Reuse the already-generated Python float outputs if they are still valid:
   - `keras_float_outputs.npz`
   - `keras_intermediates.npz`
4. Rewrite `batch_expect_class.mem` from float classes if the current policy is "RTL vs float class only".
5. Run:
   - `F:\RF_cnn\CONV1_21\project_1\run_fpga_mix12_batch.bat`
6. Watch:
   - `F:\RF_cnn\CONV1_21\project_1\mem\fpga_mix12_batch_7x50\modelsim_stdout_new.log`

## 11. Batch TB Behavior

The current batch tb has been modified to:

- emit progress every 14 samples
- flush result/logit csv files after each completed sample

File:

- `F:\RF_cnn\CONV1_21\project_1\tb\cnn1d_complex_classifier_batch_fpga_mix12_tb.v`

Progress line format:

```text
BATCH PROGRESS: N / 350, rtl_vs_float_class_match=M
```

## 12. What the Main Files Mean

### `rtl_results.csv`

Per-sample final classification result table.

New intended format:

- `sample_idx`
- `rtl_class`
- `rtl_score`
- `float_class`
- `class_match`

### `rtl_logits.csv`

Per-sample, per-class raw RTL scores:

- `sample_idx`
- `class_idx`
- `score`

This file is useful for comparing RTL class-score distribution against float logits.

### `summary_float_fixed.json`

Python float vs Python fixed summary only.

Includes:

- `float_accuracy`
- `fixed_accuracy`
- `float_fixed_agreement`
- per-layer metrics:
  - `mean_abs_err`
  - `max_abs_err`
  - `cosine_similarity`

### `summary_final.json`

Combined float / fixed / RTL summary.

Valid only when generated from a clean run directory.

## 13. Suggested Next Action for a New Chat

Use a prompt like:

> Continue the fpga_mix12 new flow only. Stop old ModelSim runs, clean the mixed 7x50 output files, rerun the 7x50 ModelSim batch with the current batch tb, and report progress every 14 samples using `modelsim_stdout_new.log`. Then summarize final RTL-vs-float accuracy, 7x7 confusion matrix, and per-layer float-vs-fixed metrics.

