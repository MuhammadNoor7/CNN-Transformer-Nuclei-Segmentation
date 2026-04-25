## Detailed Comparison: Original vs. Improved CellViT Reproduction Notebooks

### **OVERVIEW**
- **Original**: 42 cells | **Improved**: 43 cells
- **Total lines**: Original ~2340 | Improved ~2384
- **Key theme**: Improved notebook enhances the training pipeline, monitoring, and reproducibility while keeping the original notebook as reference

---

### **1. STRUCTURAL DIFFERENCES**

#### **Section Modifications**
| Section | Original | Improved |
|---------|----------|----------|
| Title | "Assignment 2: Reproducibility Study — CellViT" | "Assignment 2: Reproducibility Study -- CellViT (**Improved Copy**)" |
| Objective | Single focus on reproduction | Clarifies "original constrained" vs "improved reproduction" tracks |
| Table of Contents | Same 11 sections | Same structure but refined descriptions |

#### **New/Added Content in Improved Version**
- Additional markdown cell explaining "Reproduction Tracks" (original constrained vs improved with specifics)
- Enhanced Section 3 markdown: mentions **cached preprocessing** and **class-aware loss weighting**
- New Section 6 markdown explaining improvements (fine-tuning, longer budget, validation tracking)
- More detailed Section 11 markdown with structured failure analysis section
- New utility function: `plot_validation_metrics()` — tracks mPQ and F1_detection per epoch
- New utility function: `plot_classwise_pq()` — visualizes class-wise PQ with error bars
- New configuration tracking: `get_hardware_info()` function added

---

### **2. CODE DIFFERENCES**

#### **A. Imports**
**Original**:
```python
from collections import defaultdict, OrderedDict
from scipy.ndimage import measurements, binary_fill_holes
from skimage import morphology as sk_morph
```

**Improved**:
```python
import time  # Added for execution timing
from collections import defaultdict  # OrderedDict removed (unused)
# Simplified scipy imports
from scipy.ndimage import binary_fill_holes  # measurements, sk_morph removed
```

**Reason**: Cleaner dependencies, removed unused imports

---

#### **B. Seed-Setting Output**
**Original**: Silent seed setting
**Improved**: 
```python
set_seed(42)
print("Seed fixed at 42 for Python, NumPy, and PyTorch.")  # Added explicit output
```

---

#### **C. Configuration Class (Config)**

**Original** (~70 lines):
```python
class Config:
    """Central configuration for the entire pipeline."""
    # Basic paths, model, training settings
    BATCH_SIZE = 10  # Reduced for 12GB GPU
    NUM_EPOCHS = 10
    FREEZE_ENCODER = True  # [COLAB-OPT] Train decoder only
```

**Improved** (~110 lines):
```python
class Config:
    '''Central configuration for the improved Assignment 2 pipeline.'''
    RUN_NAME = "assignment2_improved"  # NEW
    BATCH_SIZE = 8  # Changed from 10
    NUM_EPOCHS = 40  # Changed from 10 (4x longer!)
    FREEZE_ENCODER = False  # Changed from True (fine-tune encoder)
    CACHE_PREPROCESS = True  # NEW
    CLASS_WEIGHTED_CE = True  # NEW
    GRAD_CLIP_NORM = 1.0  # NEW
    VAL_RATIO = 0.15  # NEW
    
    # Baseline reference for comparison
    BASELINE_RESULTS = {  # NEW
        "mPQ": 0.0732,
        "F1_detection": 0.2286,
    }
    
    # Output file specifications
    RESULTS_JSON = OUTPUT_DIR / "assignment2_improved_results.json"  # NEW
    FOLD_RESULTS_CSV = OUTPUT_DIR / "assignment2_improved_fold_results.csv"  # NEW
    # ... more output paths
    
    @classmethod
    def as_dict(cls):  # NEW
        return {...}
```

**Key Differences**:
- 4x more training epochs (10 → 40)
- Unfrozen encoder for end-to-end fine-tuning
- Class-weighted cross-entropy enabled
- Explicit output file naming
- Configuration serialization method added

---

#### **D. Hardware Detection**
**Original**: No explicit hardware logging
**Improved**: 
```python
def get_hardware_info():  # NEW FUNCTION
    info = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
    }
    return info

hardware_info = get_hardware_info()
print("Improved Assignment 2 configuration loaded.")
print(json.dumps({k: str(v) for k, v in cfg.as_dict().items()}, indent=2))  # Pretty-print config
print(json.dumps(hardware_info, indent=2))
```

---

#### **E. Data Loading & Fold Management**
**Original**:
```python
print(f"Successfully mapped {len(fold_data)} folds into 'fold_data'.")
```

**Improved**:
```python
fold_summaries = []  # NEW: Track dataset statistics
# ... populate fold_summaries ...
fold_summary_df = pd.DataFrame(fold_summaries)  # NEW: DataFrame summary
print(f"Successfully mapped {len(fold_data)} folds into 'fold_data'.")
if not fold_summary_df.empty:
    print("Dataset fold summary:")
    display(fold_summary_df)  # Display in notebook
```

---

#### **F. Visualization Output Paths**
**Original**:
```python
plt.savefig(cfg.OUTPUT_DIR / 'sample_data.png', ...)
```

**Improved**:
```python
plt.savefig(cfg.OUTPUT_DIR / 'assignment2_improved_sample_data.png', ...)  # More descriptive naming
```

---

#### **G. Loss Functions**

**Original** (Class-agnostic):
```python
class CellViTLoss(nn.Module):
    def __init__(self, lambda_seg=1.0, lambda_hover=1.0, lambda_cls=1.0, num_classes=6):
        self.cls_loss = nn.CrossEntropyLoss()  # Uniform weighting
```

**Improved** (Class-aware):
```python
class CellViTLoss(nn.Module):
    def __init__(self, lambda_seg=1.0, lambda_hover=1.0, lambda_cls=1.0, class_weights=None):
        self.cls_loss = nn.CrossEntropyLoss(weight=class_weights)  # NEW: optional class weights
```

**Impact**: Enables class-weighted loss for handling imbalanced rare classes (Inflammatory, Dead)

---

#### **H. Training Scheduler**
**Original**:
```python
def step(self):
    self.current_epoch += 1
    if self.current_epoch <= self.warmup_epochs:
        scale = self.current_epoch / self.warmup_epochs
    else:
        progress = (self.current_epoch - self.warmup_epochs) / (
            self.total_epochs - self.warmup_epochs)
```

**Improved** (Robustness):
```python
def step(self):
    self.current_epoch += 1
    if self.current_epoch <= self.warmup_epochs:
        scale = self.current_epoch / max(self.warmup_epochs, 1)  # Safe division
    else:
        progress = (self.current_epoch - self.warmup_epochs) / max(self.total_epochs - self.warmup_epochs, 1)  # Safe division
```

---

#### **I. Model Architecture**
**Original** (Simpler output):
```python
print(f"Detected Encoder Channels: {encoder_channels}")
# ...
if getattr(cfg, 'FREEZE_ENCODER', True):
    print(f"Encoder frozen. Trainable params: {trainable:,} / {total:,}")
```

**Improved** (More diagnostic info):
```python
print(f"Detected encoder channels: {encoder_channels}")
# ...
if getattr(cfg, "FREEZE_ENCODER", False):  # Changed default to False
# ...
print(f"Device: {device}")
print(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.1f}%)")
print("\nOutput shapes:")
for key, val in outputs.items():
    print(f"  {key}: {tuple(val.shape)}")  # Wrapped in tuple() for clarity
```

---

#### **J. Evaluation Pipeline**

**Original** (~130 lines of on-the-fly evaluation with manual accumulation)

**Improved** (~180 lines with more sophisticated tracking):
```python
# Same on-the-fly approach but with enhancements:
# - Better variable naming and organization
# - Per-class tracking with validation flags (pq_valid)
# - Clearer F1 calculation logic
```

---

#### **K. Training Loop Enhancements**

**Original** (`train_cellvit` function):
- Basic 3-fold CV loop
- Early stopping on validation loss
- History tracking: train/val loss per component

**Improved** (`train_cellvit` function):
```python
# Same structure but NEW tracking:
split_summaries = []  # NEW: Log train/val/test splits per fold
split_df = pd.DataFrame(...)  # NEW: Formatted split summary

# Configuration now includes:
class_weights  # NEW: From dataloader
split_summaries  # NEW: Logging per-fold data distribution

# Return expanded tuple:
# OLD: (all_results, all_histories)
# NEW: (all_results, all_histories, split_summaries)
```

---

#### **L. Results Reporting**

**Original** `print_results_comparison()`:
```python
# Compares reproduced results to paper baseline (mPQ=0.510, F1=0.830)
# Prints per-class PQ, per-fold breakdown
```

**Improved** `print_results_comparison()`:
```python
# Compares reproduced to BASELINE_RESULTS from config
baseline_val = cfg.BASELINE_RESULTS.get(metric, 0.0)  # From config

# NEW: Failure analysis section
weak_classes = []
for cls_name in ["Inflammatory", "Dead"]:  # Explicitly analyzes rare classes
    # ...
print("\nShort failure analysis:")
for cls_name, score in weak_classes:
    print(f"  {cls_name} remains challenging (PQ={score:.4f}); ...")

# Returns expanded tuple:
# OLD: (avg_results, std_results)
# NEW: (avg_results, std_results, fold_results_df)

# Saves CSV with explicit path from config
df.to_csv(cfg.FOLD_RESULTS_CSV, index=False)
```

---

#### **M. Plotting Functions**

**NEW in Improved version**:
```python
def plot_training_curves(histories: List[dict], save_path: Path = None):
    '''Multi-panel training curve visualization for all folds.'''
    # Plots train/val loss for: seg, hover, classification

def plot_validation_metrics(histories: List[dict], save_path: Path = None):
    '''Tracks validation mPQ and F1_detection across epochs.'''

def plot_classwise_pq(all_results: List[dict], save_path: Path = None):
    '''Bar chart with error bars per-class PQ.'''
```

**Original**: Only static plotting of pre-computed values

---

#### **N. Experiment Logging**

**Original** (`export_experiment_log`):
```python
log = {
    'experiment': 'CellViT Reproduction on PanNuke',
    'configuration': { ... },
    'hardware': { ... },
    'dataset': { ... }
}
# Minimal structure
```

**Improved** (`export_experiment_log`):
```python
log = {
    "experiment": "CellViT Improved Reproduction on PanNuke",  # Updated name
    # ... all original fields ...
    "baseline_reference": cfg.BASELINE_RESULTS,  # NEW
    # NEW tracking:
    "history_epochs": [len(hist["train_loss"]) for hist in all_histories]
}
# Enhanced metadata for reuse in Assignment 3
```

---

### **3. CONTENT & DOCUMENTATION DIFFERENCES**

#### **Objective/Motivation**
**Original**:
> "Reproduce the results of CellViT (Hörst et al., 2024) on the PanNuke dataset..."

**Improved**:
> "Reproduce the results of CellViT... This improved copy keeps the original Assignment 2 notebook untouched while upgrading the training pipeline into a **stronger end-to-end baseline for better hardware**."
> 
> "**Reproduction Tracks**:
> - Original constrained reproduction: frozen encoder, short training budget, minimal logging
> - **Improved reproduction**: full fine-tuning, class-aware loss weighting, cached preprocessing, stronger validation tracking, and richer result reporting"

---

#### **Section 3 Markdown Enhancement**
**Original**: Generic description
**Improved**: Specifies improvements in detail:
> "Following CellViT, this improved reproduction:
> 1. Applies **Macenko stain normalization**...
> 2. Generates **HoVer maps**...
> 3. Uses **cached preprocessing** so expensive targets are computed once per sample
> 4. Applies controlled histology-safe augmentation..."

---

#### **Section 6 Markdown Enhancement**
**Original**: Simple "Training Pipeline" header

**Improved**:
> "## 6. Training Pipeline
> Improvements in this copied notebook:
> - end-to-end fine-tuning from epoch 0
> - mixed precision training and evaluation
> - per-epoch validation loss, mPQ, and F1 tracking
> - best-checkpoint selection
> - structured fold histories for downstream comparison"

---

#### **Section 11 Reproducibility Discussion**

**Original Table**:
| Factor | Original Paper | Our Setup | Impact |
| GPU | A100 80GB | [Your GPU] | Batch size constraints |
| Backbone weights | SAM ViT-L | timm ViT-L pretrained | Feature quality |
| ... | ... | ... | ... |

**Improved Table** (More specific):
| Factor | Original Paper | **Improved Notebook Copy** | Impact |
| GPU | A100 80GB | **RTX 3080-class setup** | Smaller batch, shorter wall-clock |
| Backbone | SAM-scale | **timm ViT-Base pretrained** | Different init quality |
| **Batch size** | Large | **Practical GPU budget** | Affects gradient stability |
| **Training duration** | Long paper-scale | **40 epochs with early stopping** | Better than constrained, below paper |

**Improved Analysis**: Adds explicit "Why Results May Still Differ from the Paper" section discussing:
- Simplified local reproduction path
- Post-processing sensitivity
- **Rare class challenges** (Inflammatory, Dead)

---

#### **Failure Analysis (NEW in Improved)**
> "### Possible Reasons for Result Differences
> - Pretrained Weights
> - Batch Normalization Statistics
> - **Post-Processing Sensitivity**: Small changes in watershed parameters can shift PQ by 1-3%
> - Data Augmentation"

And explicit rare-class analysis in results:
> "Rare-class failure analysis: Inflammatory and Dead remain challenging (PQ=...; likely affected by rarity, overlap ambiguity, and class confusion."

---

### **4. HYPERPARAMETER DIFFERENCES**

| Parameter | Original | Improved | Rationale |
|-----------|----------|----------|-----------|
| **BATCH_SIZE** | 10 | 8 | Better stability |
| **NUM_EPOCHS** | 10 | 40 | Longer training budget |
| **FREEZE_ENCODER** | True | False | End-to-end fine-tuning |
| **LEARNING_RATE** | 1e-4 | 2e-4 | Higher LR with longer budget |
| **WARMUP_EPOCHS** | 5 | 5 | Same |
| **PATIENCE** | 5 | 8 | More tolerance |
| **NUM_WORKERS** | 0 | 4 | Better hardware utilization |
| **CACHE_PREPROCESS** | N/A | True | Memory efficiency |
| **CLASS_WEIGHTED_CE** | N/A | True | Rare class handling |
| **GRAD_CLIP_NORM** | N/A | 1.0 | Training stability |

---

### **5. OUTPUT ARTIFACTS**

**Original outputs**:
```
outputs/sample_data.png
outputs/training_curves.png
outputs/fold_results.csv
outputs/experiment_log.json
```

**Improved outputs** (with explicit naming):
```
outputs/assignment2_improved_sample_data.png
outputs/assignment2_improved_training_curves.png
outputs/assignment2_improved_class_pq.png        # NEW
outputs/assignment2_improved_sample_predictions.png  # NEW  
outputs/assignment2_improved_fold_results.csv
outputs/assignment2_improved_results.json        # NEW
outputs/assignment2_improved_experiment_log.json
checkpoints/assignment2_improved_best_model_fold{1,2,3}.pth
```

---

### **6. KEY IMPROVEMENTS SUMMARY**

| Aspect | Original | Improved |
|--------|----------|----------|
| **Training Budget** | 10 epochs | 40 epochs (4x) |
| **Encoder Training** | Frozen | Fine-tuned end-to-end |
| **Loss Function** | Uniform weighting | Class-weighted CE for rare classes |
| **Preprocessing** | Computed on-the-fly | Cached for efficiency |
| **Validation Tracking** | Loss only | Loss + mPQ + F1_detection per epoch |
| **Hardware Info** | Not tracked | Fully logged with GPU memory, CUDA version |
| **Results Reporting** | Basic comparison | Detailed with failure analysis |
| **Configuration** | Hardcoded | Serializable config class with versioning |
| **Data Split Tracking** | Implicit | Explicit with fold summaries |
| **Visualization** | Static pre-computed plots | Dynamic plotting functions with validation metrics |
| **Documentation** | Standard | Enhanced with "Reproduction Tracks" concept |

---

### **7. FORWARD COMPATIBILITY**

The improved notebook is designed to:
1. ✅ **Keep original notebook unchanged** (separate artifact)
2. ✅ **Produce reusable fold artifacts** for Assignment 3
3. ✅ **Export structured logs** (JSON, CSV, PNG) for downstream analysis
4. ✅ **Serialize full config** for reproducibility across runs
5. ✅ **Track split statistics** for cross-validation integrity

---

This summary is suitable for inclusion in a `differences.md` file documenting the evolution from the original constrained reproduction to the improved, production-ready baseline.