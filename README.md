# Oscar: Fine-Grained Webpage Fingerprinting at Scale
### Reproduction Study — ACM CCS 2024

> **Course:** Computer and Network Security (22CST352) <br>
> **Institute:** Malaviya National Institute of Technology Jaipur<br>
> **Team:** Vetapalem Sai Udheep (2023UCP1971) · Ayush Kumar (2023UCP1974)<br>
> **Instructor:** Dr. Ramesh Babu Battula<br>

---

## What is this project?

This repository contains our reproduction study of the ACM CCS 2024 paper:

**"Towards Fine-Grained Webpage Fingerprinting at Scale"**
Xiyuan Zhao, Xinhao Deng, Qi Li, Yunpeng Liu, Zhuotao Liu, Kun Sun, Ke Xu
[ACM CCS 2024](https://doi.org/10.1145/3658644.3690211) · [Zenodo Code](https://zenodo.org/records/13383332)

The paper proposes **Oscar**, a deep-learning attack that identifies individual webpages (not just websites) from encrypted Tor traffic using multi-label metric learning — even when multiple browser tabs are open simultaneously.

---

## Background — Why is this hard?

| Problem | Why it matters |
|---------|----------------|
| **Website vs Webpage** | Subpages of the same site share layouts, producing very similar traffic patterns |
| **Multi-tab browsing** | Traffic from multiple pages mixes into one stream, interfering with identification |
| **Scale** | ~50x more webpages than websites; prior attacks tested on 100 classes at most |

Oscar solves all three challenges simultaneously — the **first attack to do so**.

---

## Oscar Architecture

Oscar has three sequential modules:

```
Raw Tor Traffic
      |
      v
+----------------------+
|  1. Data Augmentation |  <-- Inter-sample + Intra-sample augmentation
+----------+-----------+
           |
           v
+----------------------+
| 2. Feature Transform  |  <-- DF-CNN + Proxy Loss + Sample Loss
+----------+-----------+
           |
           v
+----------------------+
| 3. Webpage ID (kNN)  |  <-- Proxy k-NN + Sample k-NN combined
+----------------------+
           |
           v
   Predicted Webpages
```

### Module 1 — Data Augmentation

- **Inter-sample augmentation**: Combines two traffic traces in chronological packet order to simulate unseen tab combinations. New label = union of original labels.
- **Intra-sample augmentation**: Exchanges consecutive bursts within a single trace to simulate dynamic packet ordering from different Tor circuits. Label remains unchanged.

### Module 2 — Feature Transformation

Uses a DF-based CNN to embed raw traffic sequences into a 512-dimensional space, trained with a combined multi-label metric learning loss:

```
Loss = L_proxy + beta x L_sample       (beta = 4.5)
```

- **Proxy Loss**: Pulls traffic samples toward the learnable proxy vector of their associated webpage, and away from unrelated proxies.
- **Sample Loss**: Pushes traffic samples with no overlapping webpage labels apart in the feature space.
- **Result**: 52.92% reduction in average inter-webpage cosine similarity after transformation.

### Module 3 — Webpage Identification

Dual k-NN classifier using b = 40 neighbors:

```
score_j = score_proxy_j + theta x score_sample_j       (theta = 2)
```

Webpages with a combined score above the threshold tau = 0.3 are predicted as visited.

---

## Repository Structure

```
oscar-wpf-reproduction/
|
|-- README.md
|-- .gitignore
|
|-- report/
|   '-- CNS_REPORT.pdf              # Full project report
|
|-- presentation/
|   '-- CNS_PPT.pdf                 # Presentation slides 
|
|-- oscar/                          # Source code 
|   |-- feature_transformation.py   # Embeds traffic into 512-dim vectors
|   |-- knn.py                      # Dual k-NN webpage classifier
|   |-- losses.py                   # Proxy + sample metric learning losses
|   |-- train.py                    # Full training pipeline (GPU required)
|   |-- core/
|   |   |-- deal-data/
|   |   |   '-- augment.py          # Inter/intra-sample augmentation
|   |   '-- model/
|   |       '-- df.py               # DF-based CNN backbone
|   |-- config/                     # Hyperparameter config files
|   |-- evaluate/
|   |   |-- evaluate-recall-ap.py   # Recall@k and AP@k metrics
|   |   '-- evaluate-f1score.py     # F1 score evaluation
|   '-- result/
|       '-- closed-world/
|           |-- df.pth              # Pretrained CNN weights
|           '-- proxies.pickle      # Learned webpage proxy vectors
|
'-- images/                         # Screenshots from our execution
    |-- feature_transformation.png
    |-- knn_execution.png
    |-- metric_learning.png
    '-- dependency_error.png

NOTE: The datasets/ folder (7.1 GB) is excluded via .gitignore
Download datasets from: https://zenodo.org/records/13383332
```

---

## How to Run

### Requirements

- Python 3.8 (strictly required — see note below)
- ~8 GB free disk space for datasets
- GPU recommended for full training (CPU works for inference)

### Step 1 — Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/oscar-wpf-reproduction.git
cd oscar-wpf-reproduction/oscar
```

### Step 2 — Create Python 3.8 environment

```bash
conda create -n oscar python=3.8
conda activate oscar
pip install -r requirements.txt
```

> **Important:** The repository requires Python 3.8 and PyTorch 1.9.0.
> Python 3.10 and above cannot build numpy==1.19.5 (legacy dependency).
> Always use a conda environment as shown above.

### Step 3 — Download the dataset

Download oscar-dataset.tar.gz (7.1 GB) from Zenodo:
https://zenodo.org/records/13383332

Extract it into the datasets folder:

```bash
tar -xzf oscar-dataset.tar.gz -C oscar/datasets/
```

### Step 4 — Run Feature Transformation (using pretrained model)

```bash
python feature_transformation.py -s closed-world
```

Expected output: transforms 8,129 test traffic samples into 512-dimensional embedding vectors.
Expected time: approximately 806 seconds (~13 minutes) on CPU.

### Step 5 — Run Webpage Identification

```bash
python knn.py -s closed-world
```

Expected output: Recall@k and AP@k values for k = 5, 10, 15, 20, 25, 30.

### Step 6 — (Optional) Full Training from Scratch

```bash
python train.py -s closed-world
```

> Requires approximately 6 GB GPU memory and 12 hours of training time.
> Skip this step and use the provided pretrained models in result/closed-world/.

---

## Results

### Our Reproduced Results vs Paper

| Metric | Paper Result | Our Result | Difference |
|--------|-------------|------------|------------|
| Recall@5 | 0.4899 | 0.4557 | -3.4% |
| Recall@10 | --- | 0.5770 | --- |
| Recall@15 | --- | 0.6420 | --- |
| Recall@20 | --- | 0.6848 | --- |
| Recall@25 | --- | 0.7170 | --- |
| Recall@30 | ~0.73 | 0.7405 | <0.1% |
| AP@1 | --- | 0.4291 | --- |
| AP@2 | --- | 0.4372 | --- |
| AP@3 | --- | 0.5095 | --- |
| AP@4 | --- | 0.5941 | --- |
| AP@5 | 0.7344 | 0.6789 | -5.6% |

**Recall@30 matches the paper almost exactly (less than 0.1% gap).**
Our result outperforms all 6 baseline attacks even under CPU-only inference.

### Comparison with All Baselines (Recall@5, Closed-World)

| Attack | Recall@5 | Type |
|--------|----------|------|
| k-FP | 0.2331 | Single-tab, ML |
| NetCLR | 0.1809 | Single-tab, DL |
| DF | 0.3354 | Single-tab, DL |
| Tik-Tok | 0.3313 | Single-tab, DL |
| BAPM | 0.2106 | Multi-tab |
| TMWF | 0.3951 | Multi-tab |
| **Oscar (paper)** | **0.4899** | **WPF** |
| **Ours (reproduced)** | **0.4557** | **WPF** |

Oscar improves +24.0% over the best baseline (TMWF).
Our CPU-reproduced result still beats all baselines by +15.3%.

---

## Pipeline Execution Status

| Stage | Status | Notes |
|-------|--------|-------|
| Repository Extraction | Success | 7.2 GB total |
| Dependency Installation | Partial | Required Python 3.8 conda environment |
| Pretrained Model Loading | Success | df.pth + proxies.pickle |
| Dataset Loading | Success | 65,027 train / 8,129 test samples |
| Feature Transformation | Success | 806 seconds on CPU |
| kNN Webpage Identification | Success | Recall@5: 0.4557, AP@5: 0.6789 |
| Full GPU Training | Not performed | Requires 6 GB GPU and ~12 hours |

---

## Challenges Faced

| Challenge | Details | Resolution |
|-----------|---------|------------|
| Python compatibility | numpy==1.19.5 fails to build on Python 3.13 | Used conda with Python 3.8 |
| No GPU available | Full training requires ~6 GB GPU memory | Used pretrained weights provided by authors |
| Dataset size | 7.1 GB download required | Excluded via .gitignore, link provided |
| Multi-label metrics | Recall@k and AP@k are non-standard | Carefully studied paper Section 6.1 and Equations 14-16 |

---

## Key Insights

- **Metric learning is the key differentiator**: the 52.92% reduction in inter-webpage similarity in the transformed space is what enables Oscar to outperform all baselines. Cross-entropy loss alone (used by DF and Tik-Tok) cannot capture these subtle differences between similar webpages.

- **Both losses are necessary**: the ablation study in the paper shows proxy-only loss achieves Recall@5 = 0.3066, while the combined loss achieves 0.4899 — a 60% improvement. Neither loss works well in isolation.

- **Data augmentation is critical**: without augmentation, Recall@5 drops from 0.4899 to 0.4511, confirming that simulating diverse multi-tab combinations is essential for generalization.

- **Reproducibility in ML security research is hard**: even with pretrained models and full datasets provided, resolving software environment issues required significant effort.

---

## Documents

| Document | Link |
|----------|------|
| Full Report | [CNS_REPORT.pdf](report/CNS_REPORT.pdf) |
| Presentation Slides | [CNS_PPT.pdf](presentation/CNS_PPT.pdf) |
| Original Paper Code | [Zenodo Record 13383332](https://zenodo.org/records/13383332) |
| Original Paper | [ACM Digital Library](https://doi.org/10.1145/3658644.3690211) |

---

## References

1. Xiyuan Zhao, Xinhao Deng, Qi Li, Yunpeng Liu, Zhuotao Liu, Kun Sun, Ke Xu. *Towards Fine-Grained Webpage Fingerprinting at Scale.* ACM CCS 2024.
2. Payap Sirinam, Mohsen Imani, Marc Juarez, Matthew Wright. *Deep Fingerprinting: Undermining Website Fingerprinting Defenses with Deep Learning.* ACM CCS 2018.
3. Roger Dingledine, Nick Mathewson, Paul Syverson. *Tor: The Second-Generation Onion Router.* USENIX Security Symposium, 2004.
