# Assignment 2 CellViT Reproduction Kickoff

Date: 2026-04-02
Target paper: CellViT (Horst et al.)
Target repo: https://github.com/TIO-IKIM/CellViT

## Current Status

- [x] Official CellViT repository cloned to `external/CellViT`
- [x] Python 3.9 installed locally
- [x] Virtual environment created at `.venv39`
- [ ] Full dependency installation completed in `.venv39`
- [ ] GPU visibility check completed
- [ ] PanNuke fold structure verified
- [ ] Smoke run completed

## Phase 1: Paper Deep-Dive Checklist

Fill this before running long training:

| Item | Value |
|---|---|
| Dataset protocol | PanNuke 3-fold CV |
| Input size | 256x256 |
| Optimizer | AdamW |
| Learning rate | TBD from config/paper |
| Weight decay | TBD |
| Epochs | 130 (repo note) |
| Loss terms | BCE + Dice + CE |
| Reported metrics | mPQ 0.51, F1 0.83 |
| Checkpoint for eval | latest_checkpoint.pth |

## Phase 2: Environment Commands

Run these from the project root:

```powershell
py -3.9 -m venv .venv39
.\.venv39\Scripts\python -m pip install --upgrade pip
.\.venv39\Scripts\python -m pip install -r external/CellViT/requirements.txt
```

If `openslide` issues appear on Windows, prefer a conda/miniforge environment for that package.

## Phase 3: First Smoke Test Goal

After dependencies are installed:

```powershell
.\.venv39\Scripts\python external/CellViT/cell_segmentation/run_cellvit.py --help
```

Success criteria:
- CLI help prints without import errors.
- Config path can be parsed.

## Next Immediate Tasks

1. Complete dependency installation in `.venv39`.
2. Verify torch CUDA access.
3. Prepare PanNuke fold paths according to `external/CellViT/docs/readmes/pannuke.md`.
4. Run one short config-based smoke training/inference job.
