# Experiment Documentation — Assignment 3

## Team
- i232515, i232520, i232634

## 1. Global Results Summary
The following metrics represent the final consolidated performance across all experiments:

```json
{
  "assignment2_mean_mPQ": 14.22,
  "assignment2_mean_F1_detection": 31.49,
  "paper_mean_mPQ": 48.27,
  "paper_mean_F1_detection": 82.44,
  "monuseg_best_mean_bPQ": 66.21,
  "monuseg_best_mean_dice": 83.24,
  "monuseg_best_mean_f1_detection": 84.83,
  "note": "CART kept as proposed method scaffold; no fabricated performance claims included."
}
```

## 2. Experiments Conducted

### Experiment 1: PanNuke Reproduction Baseline (Assignment 2 Improved)
- **Source:** `outputs/assignment2_improved_results.json`
- **Hardware:** NVIDIA RTX 3090 (24GB VRAM)
- **Config:** `outputs/assignment2_improved_quick.yaml`
- **Result:** mPQ = 14.22%, F1 = 31.49% (Improved Baseline)

### Experiment 2: Cross-Domain Evaluation on MoNuSeg
- **Source:** `outputs/assignment3_monuseg_best_bpq.png` (Visualization)
- **Model:** CellViT Improved Checkpoint
- **Result:** Binary Dice = 83.24%, bPQ = 66.21%, F1 = 84.83%
- **Summary:** Documented in Section 6.3 of the report.

### Experiment 3: CART Architecture (Proposed)
- **Status:** Architecture implemented; training scaffolded.
- **Code:** Section 7 of the notebook.
- **Config:** `outputs/assignment2_improved_plan.json`
- **Result:** Architecture defined; benchmarked as future work.

## 3. Generated Artifacts
- `outputs/assignment2_improved_results.json`
- `outputs/assignment3_class_pq_comparison.png`
- `outputs/assignment3_fold_consistency.png`
- `outputs/assignment3_monuseg_best_bpq.png`
- `outputs/assignment3_summary.json`
- `outputs/experiment_summary.jpg`

## 4. How to Reproduce
1.  **Open Notebook:** `i232515_i232520_i232634_Assignment3_DL_ExperimentationExpansion.ipynb`.
2.  **Run All:** Execute all cells from top to bottom.
3.  **Artifact Loading:** All core results load from existing JSON artifacts in the `outputs/` folder — no GPU is required for visualization/analysis.
