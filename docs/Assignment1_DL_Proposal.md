# Supervised Deep Learning with CNN–Transformer Hybrids for Multi-Tissue Nuclei Instance Segmentation and Classification in Digital Pathology

**Research Proposal**

**Domain:** Medical Imaging
**Paradigm:** Supervised Deep Learning

---

**Group Members:**

- **Member 1:** Data & Preprocessing
- **Member 2:** Model Architecture & Training
- **Member 3:** Evaluation & Statistical Analysis

---

**Department of Data Science**
**Deep Learning Course**
*March 1, 2026*

---

## Contents

1. [Problem Definition](#1-problem-definition)
   - 1.1 [Task Description](#11-task-description)
   - 1.2 [Motivation and Relevance](#12-motivation-and-relevance)
   - 1.3 [The Necessity of Deep Learning](#13-the-necessity-of-deep-learning)
   - 1.4 [Research Questions](#14-research-questions)
2. [Literature Review](#2-literature-review)
   - 2.1 [Overview of Architectural Evolution](#21-overview-of-architectural-evolution)
   - 2.2 [Detailed Review of Baseline Papers (2024+)](#22-detailed-review-of-baseline-papers-2024)
   - 2.3 [Summaries of Additional Papers](#23-summaries-of-additional-papers)
3. [Proposed Deep Learning Approach](#3-proposed-deep-learning-approach)
   - 3.1 [Architectural Strategy](#31-architectural-strategy)
   - 3.2 [Mathematical Formulation](#32-mathematical-formulation)
   - 3.3 [Why this Approach is Suitable](#33-why-this-approach-is-suitable)
4. [Dataset Description: PanNuke](#4-dataset-description-pannuke)
5. [Evaluation Plan](#5-evaluation-plan)
   - 5.1 [Metrics](#51-metrics)
   - 5.2 [Experimental Setup](#52-experimental-setup)
6. [Expected Outcomes and Limitations](#6-expected-outcomes-and-limitations)
   - 6.1 [Performance Expectations](#61-performance-expectations)
   - 6.2 [Potential Limitations](#62-potential-limitations)

---

## 1 Problem Definition

### 1.1 Task Description

The primary objective of this research is to solve the dual problem of **nuclei instance segmentation** and **cell classification** within digitized histopathology images. Specifically, given an input RGB image patch **x ∈ ℝ^(H×W×3)** derived from hematoxylin and eosin (H&E) stained Whole Slide Images (WSIs), the model must output:

1. A set of instance masks **{m₁, m₂, …, mₙ}**, where each **mᵢ** corresponds to the precise pixel-wise boundary of an individual nucleus.
2. A set of corresponding categorical labels **{y₁, y₂, …, yₙ}**, assigning each nucleus to a specific clinical type (e.g., Neoplastic, Inflammatory, Epithelial).

Following the definitions in Goodfellow et al. [1], this is a **supervised learning** task. The model learns a function **f(x; θ)** that maps high-dimensional input pixels to structured outputs by minimizing a loss function **L** over a labeled training set **D = {(x^(i), y^(i))}^N_(i=1)**.

### 1.2 Motivation and Relevance

The analysis of nuclear morphology is the gold standard for cancer diagnosis, grading, and prognosis. However, manual examination by pathologists is subjective, prone to inter-observer variability, and increasingly bottlenecked by the sheer volume of digital data [2]. Automated analysis facilitates:

- **Clinical Precision:** Quantitative morphometry allows for the discovery of sub-visual biomarkers.
- **Reproducibility:** Deep learning models provide consistent, objective measurements across different laboratories.
- **Workload Reduction:** AI assistants can pre-screen slides, highlighting regions of interest for human verification.

### 1.3 The Necessity of Deep Learning

Standard image processing algorithms fail in the "clinical wild" due to nuclear overlap, staining variations, and complex tissue architectures. As articulated by Goodfellow [1], deep neural networks are fundamentally necessary here for three reasons:

1. **Representation Learning:** Unlike handcrafted features, deep models automatically learn a hierarchy of features directly from raw pixels.
2. **Curse of Dimensionality:** Histopathology patches reside in an extremely high-dimensional space. Deep models utilize inductive biases to learn the underlying manifold of nuclear morphology effectively.
3. **Depth and Complexity:** The variability in nuclear shapes and overlapping boundaries requires the composition of many non-linear functions (depth) to approximate the complex mapping required for accurate instance separation.

### 1.4 Research Questions

This project seeks to answer the following:

- **RQ1:** Can a CNN–Transformer hybrid architecture outperform 2024 convolutional baselines in Panoptic Quality (PQ) by leveraging global context?
- **RQ2:** To what extent does multi-task learning (joint segmentation and classification) improve the feature representation for rare cell types compared to single-task models?
- **RQ3:** Does the application of Macenko stain normalization significantly improve model generalization across the 19 diverse tissue types found in the PanNuke dataset?

---

## 2 Literature Review

### 2.1 Overview of Architectural Evolution

The field has transitioned from traditional watershed algorithms to Fully Convolutional Networks (FCNs), and subsequently to multi-branch architectures like HoVer-Net [5]. Recently, the trend has shifted toward Vision Transformers (ViT) to capture long-range spatial dependencies and Foundation Models that leverage massive pre-training on pathology data [2].

### 2.2 Detailed Review of Baseline Papers (2024+)

#### Baseline 1 (2024+): CellViT [2]

| Attribute | Detail |
|-----------|--------|
| **Problem** | High-precision segmentation across heterogeneous pan-cancer data. |
| **Model** | A U-Net-shaped hybrid utilizing a ViT encoder. Training employs Binary Cross-Entropy (BCE) and Dice loss for the mask branches, and Cross-Entropy for classification, optimized via the AdamW optimizer [2]. |
| **Metrics** | Achieved **0.51 mPQ** and **0.83 F1-detection** on PanNuke. |
| **Strengths** | Leverages large-scale representation learning for robustness. |
| **Weakness** | High computational demand and lack of convolutional inductive bias in standard transformers. |

#### Baseline 2 (2024+): LKCell [3]

| Attribute | Detail |
|-----------|--------|
| **Problem** | Balancing receptive field size with computational efficiency. |
| **Model** | A network using "large convolution kernels" (e.g., 31 × 31). Trained using the AdamW optimizer for 100 epochs specifically without early stopping to ensure the largest possible training database is utilized [3]. |
| **Metrics** | **0.5080 mPQ** on PanNuke with only **21.6% of the FLOPs** of leading Transformer models. |
| **Strengths** | Extremely efficient; shows that large receptive fields are key for context. |
| **Weakness** | Can be sensitive to local noise if the large kernel captures artifacts. |

#### Baseline 3 (2024+): RepSNet [4]

| Attribute | Detail |
|-----------|--------|
| **Problem** | Separating dense nuclei in blurred boundary regions. Although RepSNet reports results on the Lizard dataset, we will adapt it to PanNuke for fair comparison. |
| **Model** | A re-parameterizable network. Training involves a weighted combination of BCE, Dice loss, Smooth L1 for distance regression, and a novel Boundary Isoheight loss to penalize boundary deviations [4]. |
| **Metrics** | **0.5633 mPQ** on the Lizard dataset. |
| **Strengths** | High inference speed (10+ FPS) and robust boundary voting mechanism. |
| **Weakness** | Relies on regression maps which may struggle with extreme morphological outliers. |

### 2.3 Summaries of Additional Papers

1. **HoVer-Net (2019) [5]:** Established the standard multi-branch approach using HoVer maps. <https://doi.org/10.1016/j.media.2019.101563>
2. **StarDist (2020) [6]:** Introduced star-convex polygon representations for nuclei. <https://doi.org/10.1101/2020.04.22.055376>
3. **TSFD-Net (2022) [8]:** Proposed tissue-specific feature distillation to handle variations across organs. <https://doi.org/10.1016/j.neunet.2022.03.011>
4. **PanNuke Baselines (2020) [7]:** Introduced the first pan-cancer dataset with semi-automatic verification. <https://arxiv.org/abs/2003.10778>
5. **NuCLS (2022) [9]:** A large-scale crowdsourced dataset for breast cancer nuclei. <https://doi.org/10.1093/gigascience/giac037>
6. **CelloType (2024) [10]:** A unified Transformer model for spatial omics and histology. <https://doi.org/10.1038/s41592-024-02513-1>
7. **PointFormer (2025) [11]:** A keypoint-guided Transformer using a tri-decoder structure. <https://doi.org/10.1109/TIP.2025.3565184>

---

## 3 Proposed Deep Learning Approach

### 3.1 Architectural Strategy

We propose the **Context-Aware Re-parameterizable Transformer (CART)**. The model joints a CNN's local efficiency with a Transformer's global context.

- **Hybrid Backbone:** A shared encoder using re-parameterizable convolutions followed by a global attention bottleneck layer to capture the spatial relationship between distant nuclei.
- **Multi-Task Heads:** Three task-specific decoders for mask prediction, distance regression, and classification.

### 3.2 Mathematical Formulation

The model is optimized using a weighted multi-task loss function:

$$\mathcal{L}_{\text{total}} = \lambda_{\text{seg}} \mathcal{L}_{\text{Dice}} + \lambda_{\text{dist}} \mathcal{L}_{L1} + \lambda_{\text{cls}} \mathcal{L}_{\text{CE}} \tag{1}$$

Following Goodfellow [1], we utilize **parameter sharing** in the backbone to improve statistical efficiency.

### 3.3 Why this Approach is Suitable

As noted by Goodfellow [1], depth allows the model to learn complex hierarchical functions. By combining CNN layers (local patterns) with Transformers (global context), we address the high-dimensional nature of histopathology.

---

## 4 Dataset Description: PanNuke

**Source:** <https://warwick.ac.uk/fac/sci/dcs/research/tia/data/pannuke>

PanNuke is the most diverse dataset in computational pathology [7]:

| Property | Detail |
|----------|--------|
| **Volume** | 189,744 segmented nuclei across 7,904 image patches |
| **Diversity** | 19 different tissue types (Lung, Colon, Breast, etc.) |
| **Classes** | 5 clinical types (Neoplastic, Inflammatory, Epithelial, Dead, Connective) |

---

## 5 Evaluation Plan

### 5.1 Metrics

- **Panoptic Quality (PQ):** Primary metric, accounting for detection and segmentation quality.
- **F1-Score:** To evaluate classification performance per category.

### 5.2 Experimental Setup

We will utilize an **80/10/10 train/val/test split**. Training will involve **AdamW** optimization. To prevent overfitting, **Early Stopping** will be implemented, halting training if the validation loss does not decrease for 10 consecutive epochs.

We will perform **Hyperparameter Tuning** via grid search, specifically testing:

- **Learning rates:** `[1e-5, 1e-4, 5e-4]`
- **Batch sizes:** `[1, 2, 3]` (to account for memory constraints)

We will reproduce the three baseline methods using the same setup for a rigorous comparison.

---

## 6 Expected Outcomes and Limitations

### 6.1 Performance Expectations

We anticipate our CART model will achieve a **PanNuke mPQ > 0.52**, surpassing the 2024 CellViT benchmark while maintaining a lower parameter count during inference through re-parameterization.

### 6.2 Potential Limitations

- **Computational Demand:** Hybrid architectures are resource-intensive during training, potentially requiring high-end GPUs.
- **GPU Memory:** Large patch sizes (1024 × 1024) may limit batch size, affecting gradient stability.
- **Domain Shift:** Despite PanNuke's diversity, performance may drop on unseen tissue types or artifacts not present in the training set.
- **Baseline Approximation:** Reproducing baselines may involve approximations where specific pre-training weights or hardware-specific optimizations are unavailable.

---

## References

[1] Goodfellow, I., Bengio, Y., and Courville, A. (2016). *Deep Learning*. MIT Press.

[2] Horst, F., et al. (2024). CellViT: Vision Transformers for Precise Cell Segmentation and Classification. *Medical Image Analysis*. <https://doi.org/10.1016/j.media.2024.103143>

[3] Cui, Z., et al. (2024). LKCell: Efficient Cell Nuclei Instance Segmentation with Large Convolution Kernels. *arXiv:2407.18054*. <https://arxiv.org/abs/2407.18054>

[4] Xiong, S., et al. (2025). RepSNet: A Nucleus Instance Segmentation Model Based on Boundary Regression and Structural Re-Parameterization. *IJCV*. <https://doi.org/10.1007/s11263-024-02332-z>

[5] Graham, S., et al. (2019). HoVer-Net: Simultaneous Segmentation and Classification of Nuclei. *Medical Image Analysis*. <https://doi.org/10.1016/j.media.2019.101563>

[6] Weigert, M., and Schmidt, U. (2020). Nuclei Instance Segmentation and Classification with StarDist. *ISBI 2022*. <https://doi.org/10.1101/2020.04.22.055376>

[7] Gamper, J., et al. (2020). PanNuke Dataset Extension, Insights and Baselines. *arXiv:2003.10778*. <https://arxiv.org/abs/2003.10778>

[8] Ilyas, T., et al. (2022). TSFD-Net: Tissue Specific Feature Distillation Network. *Neural Networks*. <https://doi.org/10.1016/j.neunet.2022.03.011>

[9] Amgad, M., et al. (2022). NuCLS: A Scalable Crowdsourcing Approach for Nucleus Classification. *GigaScience*. <https://doi.org/10.1093/gigascience/giac037>

[10] Pang, M., et al. (2024). CelloType: A Unified Model for Segmentation and Classification. *Nature Methods*. <https://doi.org/10.1038/s41592-024-02513-1>

[11] Xu, J., et al. (2025). PointFormer: Keypoint-Guided Transformer for Simultaneous Nuclei NSC. *IEEE TIP*. <https://doi.org/10.1109/TIP.2025.3565184>
