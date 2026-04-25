# CNN–Transformer Nuclei Segmentation in Digital Pathology

```text
  _____     _ _  __      __  _____  _____ 
 / ____|   | | | \  \   /  / |   | |_   _|
| |     ___| | |  \  \_/  /   | |    | |  
| |    / _ \ | |   \     /    | |    | |  
| |___|  __/ | |    \   /     | |    | |  
 \_____\___|_|_|     \_/    |_____|  |_|  
                                    
  NUCLEI SEGMENTATION & CLASSIFICATION
```

---

## 📑 Table of Contents
1. [Overview](#overview)
2. [Literature Review & Proposal (Assignment 1)](#literature-review--proposal-assignment-1)
    - [Project Motivation](#project-motivation)
    - [The Hybrid Approach](#the-hybrid-approach)
3. [Reproducibility Study (Assignment 2)](#reproducibility-study-assignment-2)
    - [Constrained Track Details](#constrained-track-details)
    - [Improved Track Enhancements](#improved-track-enhancements)
    - [Performance Leap Analysis](#performance-leap-analysis)
4. [Research Expansion & Innovation (Assignment 3)](#research-expansion--innovation-assignment-3)
    - [MoNuSeg Cross-Domain Transfer](#monuseg-cross-domain-transfer)
    - [CART Architecture Proposal](#cart-architecture-proposal)
5. [Deep Dive: Evaluation Metrics](#deep-dive-evaluation-metrics)
    - [Panoptic Quality (PQ)](#panoptic-quality-pq)
    - [Dice Coefficient](#dice-coefficient)
    - [F1-Detection](#f1-detection)
6. [Dataset Glossary](#dataset-glossary)
    - [Nucleus Classes](#nucleus-classes)
    - [Tissue Types](#tissue-types)
7. [Installation & Environment Setup](#installation--environment-setup)
    - [Windows Setup](#windows-setup)
    - [Linux/Ubuntu Setup](#linuxubuntu-setup)
8. [Usage Guide](#usage-guide)
    - [Running the Notebook](#running-the-notebook)
    - [Training from Scratch](#training-from-scratch)
9. [Folder Structure Details](#folder-structure-details)
10. [Frequently Asked Questions (FAQ)](#frequently-asked-questions-faq)
11. [Troubleshooting](#troubleshooting)
12. [Future Roadmap](#future-roadmap)
13. [Authors & Acknowledgments](#authors--acknowledgments)
14. [License](#license)

---

## 1. Overview
This project explores supervised deep learning techniques for nuclei instance segmentation and classification in histopathology images. It proposes a hybrid CNN–Transformer architecture that combines the local feature extraction capabilities of convolutional networks with the global contextual modeling of transformers.

Histopathology images are high-resolution, dense, and contain complex spatial relationships. Accurate segmentation is vital for:
- **Oncology:** Grading tumors and assessing prognosis.
- **Biomedical Research:** Understanding cellular response to drugs.
- **Pathology Automation:** Reducing the manual workload of pathologists.

The model is evaluated on the PanNuke dataset, a large-scale benchmark containing nearly 190k labeled nuclei across 19 tissue types.

---

## 2. Literature Review & Proposal (Assignment 1)

### Project Motivation
This repository presents a **supervised deep learning project** focused on **nuclei instance segmentation and classification in histopathology images** using a **hybrid CNN–Transformer architecture**. The project is designed for **digital pathology research**, where accurate nuclei detection and classification are essential for understanding tissue structure, identifying disease patterns, and supporting biomedical analysis.

Histopathology images are highly complex and often contain nuclei that overlap, vary in size and shape, and appear differently across tissue types. These challenges make nuclei segmentation and classification difficult for traditional approaches. 

### The Hybrid Approach
The project explores a hybrid model that combines the strengths of **convolutional neural networks (CNNs)** and **Transformers**:
1.  **CNNs (Local Context):** Efficient at extracting edges, contours, and fine texture details. They are spatially invariant and excellent for local morphology.
2.  **Transformers (Global Context):** Better at capturing long-range dependencies and global contextual information via the self-attention mechanism.

Together, they provide a more robust framework for analyzing microscopy images where the identity of a cell is often determined by its relationship to its neighbors.

---

## 3. Reproducibility Study (Assignment 2)

The second phase of this project focused on a rigorous reproduction of the **CellViT** architecture.

### Constrained Track Details
In the constrained track, we simulated a student environment with limited resources:
- **Backbone:** Frozen ViT-256 (Pretrained on ImageNet-21k).
- **Epochs:** 10.
- **Batch Size:** 8.
- **GPU:** Tesla T4 (16GB).
- **Outcome:** Mean Panoptic Quality (mPQ) of **7.32%**. The model struggled with rare classes like 'Dead' nuclei due to lack of backbone adaptation.

### Improved Track Enhancements
The improved track optimized the pipeline for professional performance:
- **Backbone Fine-tuning:** Encoder was unfrozen to allow the ViT to learn histopathology-specific features.
- **Class-Weighted Loss:** Introduced to penalize errors on rare classes, forcing the model to learn 'Dead' and 'Inflammatory' cells.
- **AMP (Automatic Mixed Precision):** Enabled 16-bit training, reducing VRAM usage and accelerating throughput.
- **Budget:** 30 Epochs with Early Stopping.

### Performance Leap Analysis
The improved track reached an mPQ of **14.22%**, representing a **94% relative improvement** over the constrained run. This jump confirms that for high-parameter architectures like ViT, backbone fine-tuning is non-negotiable for medical imaging tasks.

---

## 4. Research Expansion & Innovation (Assignment 3)

### MoNuSeg Cross-Domain Transfer
To test the robustness of our improved model, we performed a cross-domain evaluation on the **MoNuSeg** dataset.
- **Challenge:** MoNuSeg contains images from 7 different organs, most of which have different staining characteristics than PanNuke.
- **Results:** The model achieved a **Binary Dice score of 83.24%**. This result is significant because it proves that the model has learned a "universal" nuclear representation that transcends tissue-specific domains.

### CART Architecture Proposal
We propose the **Context-Aware Re-parameterizable Transformer (CART)**.
- **Inspiration:** MHSA is computationally expensive for whole-slide images.
- **Mechanism:** replaces MHSA with a dual-branch convolutional block during training. 
- **Re-parameterization:** During inference, the branches are folded into a single $3\times3$ convolution, maintaining accuracy while dramatically increasing speed.

---

## 5. Deep Dive: Evaluation Metrics

### Panoptic Quality (PQ)
Panoptic Quality is the primary metric for instance segmentation. It is calculated as:
$$PQ = \frac{\sum_{(p, g) \in TP} IoU(p, g)}{|TP| + \frac{1}{2}|FP| + \frac{1}{2}|FN|}$$

Where:
- **TP (True Positives):** Predicted instances that have an IoU > 0.5 with a ground truth instance.
- **FP (False Positives):** Predicted instances with no matching ground truth.
- **FN (False Negatives):** Ground truth instances with no matching prediction.

PQ is the product of:
1.  **Segmentation Quality (SQ):** Average IoU of true positives.
2.  **Detection Quality (DQ):** F1-score of the detected instances.

### Dice Coefficient
Used for pixel-level overlap accuracy:
$$Dice = \frac{2 \times |P \cap G|}{|P| + |G|}$$

---

## 6. Dataset Glossary

### Nucleus Classes
1.  **Neoplastic:** Cancerous cells with irregular boundaries.
2.  **Inflammatory:** Immune cells like lymphocytes.
3.  **Connective:** Fibroblasts and structural cells.
4.  **Dead:** Apoptotic or necrotic cells (rare and small).
5.  **Epithelial:** Cells lining the organs.

### Tissue Types (PanNuke)
PanNuke includes 19 tissues, including:
- Adrenal Gland
- Bile Duct
- Bladder
- Breast
- Cervix
- Colon
- Esophagus
- Head and Neck
- Kidney
- Liver
- Lung
- Ovarian
- Pancreatic
- Prostate
- Skin
- Stomach
- Testis
- Thyroid
- Uterus

---

## 7. Installation & Environment Setup

To ensure a clean environment, it is recommended to use a virtual environment (venv) or Conda:

1.  **Create and Activate Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    ```
2.  **Install PyTorch (CUDA 12.1):**
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ```
3.  **Install Remaining Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## 8. Usage Guide

### Running the Notebook
The full research analysis, including the CART architecture implementation and MoNuSeg evaluation, is contained in:
`notebook/i232515_i232520_i232634_Assignment3_DL_ExperimentationExpansion.ipynb`

- **Step 1:** Place your PanNuke data in the `data/` folder.
- **Step 2:** Ensure the best model weights are located in the `checkpoints/` directory.
- **Step 3:** Open the notebook and click **"Run All Cells"**.

---

## 9. Folder Structure Details

```bash
├── checkpoints/              # Best model weights (.pth)
│   ├── assignment2_improved_best_model_fold1.pth
│   ├── assignment2_improved_best_model_fold2.pth
│   └── assignment2_improved_best_model_fold3.pth
├── data/                     # Raw and prepared data
│   └── pannuke/              # PanNuke images/masks
│       ├── fold1
│       ├── fold2
│       └── fold3
├── Dataset/                  # Raw PanNuke dataset
├── docs/                     # Research and progress reports
│   ├── Assignment1_DL_Proposal.md
│   ├── Assignment2_CellViT_Kickoff.md
│   ├── Assignment3_DL_Experiment_Expansion.md
│   ├── CNNTransformer_NucleiSegmentation_Literature.md
│   ├── CNNTransformer_NucleiSegmentation_Proposal.md
│   ├── i232515_i232520_i232634_Assignment2_DL_CellViT_Reproduction.pdf
│   └── i232515_i232520_i232634_Assignment3_DL_Experimentation_Expansion.pdf
├── external/                 # External dependencies
├── logs/                     # Training and validation logs
├── notebook/                 # Primary implementation files
│   ├── i232515_i232520_i232634_Assignment2_DL_CellViT_Constrained_Reproduction.ipynb
│   ├── i232515_i232520_i232634_Assignment2_DL_CellViT_Improved_Reproduction.ipynb
│   └── i232515_i232520_i232634_Assignment3_DL_ExperimentationExpansion.ipynb
├── outputs/                  # Final results and visualizations
├── differences.md            # Notebook comparison log
├── experiment_doc.md         # Technical experiment log
├── LICENSE                   # Licensing info
├── README.md                 # This document
└── requirements.txt          # Python dependencies
```


---

## 12. Future Roadmap

1.  **Full CART Training:** Benchmarking the efficiency gains on a cluster of A100 GPUs.
2.  **Self-Supervised Pretraining:** Using Masked Autoencoders (MAE) on 1 million pathology patches before fine-tuning.
3.  **Active Learning:** Developing a loop where the model asks for annotations on its most uncertain predictions.
4.  **Mobile Deployment:** Optimizing CART for real-time inference on mobile pathology scanners.

---

## 13. Authors & Acknowledgments

### Team Members
- **Muhammad Noor**
- **Shahoud Shahid**
- **Saif Shehzad**

### Acknowledgments
We would like to thank our instructors for their guidance:
- **Dr. Qurat Ul Ain**
- **Dr. Zohair Ahmed**
- **Mr. Ubaid Ur Rehman**

Special thanks to the authors of **CellViT** and the **PanNuke** consortium for their open-access datasets and codebases.

---

## 14. License

This project is licensed under the **MIT License**.

Copyright (c) 2026 Shahoud Shahid, Muhammad Noor, Saif Shehzad

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---
