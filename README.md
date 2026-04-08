<h1 align="center">Federated Learning With Stage-Wise Trajectory Matching<br>for AAV Networks on Non-IID Data</h1>

<p align="center">
  <em>IEEE Transactions on Cognitive Communications and Networking (TCCN), 2026</em>
</p>

<p align="center">
  <a href="paper/Supplementary%20Materials.pdf"><img src="https://img.shields.io/badge/Supplementary%20Materials-PDF-green?logo=adobeacrobatreader" /></a>
  <a href="paper/Federated%20Learning%20with%20Stage-wise%20Trajectory%20Matching%20for%20UAV%20Networks%20on%20Non-IID%20Data.pdf"><img src="https://img.shields.io/badge/Paper-PDF-red?logo=adobeacrobatreader" /></a>
  <a href="https://ieeexplore.ieee.org/document/11421445"><img src="https://img.shields.io/badge/IEEE%20Xplore-11421445-blue?logo=ieee" /></a>
  <img src="https://visitor-badge.laobi.icu/badge?page_id=zhangweikk.FedSTM&left_color=gray&right_color=orange" alt="Visitors" />
  <img src="https://img.shields.io/github/stars/zhangweikk/FedSTM?style=social" alt="GitHub Stars" />
</p>

<p align="center">
  <a href="paper/Supplementary%20Materials.pdf">📎 Supplementary Materials (PDF)</a>&nbsp;&nbsp;|&nbsp;&nbsp;<a href="paper/Federated%20Learning%20with%20Stage-wise%20Trajectory%20Matching%20for%20UAV%20Networks%20on%20Non-IID%20Data.pdf">📄 Paper (PDF)</a>&nbsp;&nbsp;|&nbsp;&nbsp;<a href="https://ieeexplore.ieee.org/document/11421445">🌐 IEEE Xplore</a>
</p>

> [!NOTE]
> The supplementary materials are also available via **Supplemental Items** on the [IEEE Xplore](https://ieeexplore.ieee.org/document/11421445) page.

---

## ✍️ Authors

<p align="center">
  <span style="color: #0969da; font-size: 1.05em;">Xiufang Shi</span><sup>1</sup>&nbsp;&nbsp;
  <span style="color: #0969da; font-size: 1.05em;">Wei Zhang</span><sup>1</sup>&nbsp;&nbsp;
  <span style="color: #0969da; font-size: 1.05em;">Yuheng Li</span><sup>1</sup>&nbsp;&nbsp;
  <span style="color: #0969da; font-size: 1.05em;">Mincheng Wu</span><sup>1</sup>&nbsp;&nbsp;
  <span style="color: #0969da; font-size: 1.05em;">Rushi Li</span><sup>1</sup>&nbsp;&nbsp;
  <span style="color: #0969da; font-size: 1.05em;">Shibo He</span><sup>2</sup>
</p>

<p align="center">
  <sup>1</sup>&nbsp;Zhejiang University of Technology &nbsp;&nbsp;|&nbsp;&nbsp;
  <sup>2</sup>&nbsp;Zhejiang University
</p>

---

## 🔭 Overview

We propose **FedSTM**, a federated learning framework with **stage-wise trajectory matching** for autonomous aerial vehicle (AAV) networks under Non-IID data. FedSTM leverages dataset distillation with two-stage trajectory matching to effectively address data heterogeneity across clients.

<p align="center">
  <img src="figure/overview.png" width="90%" alt="FedSTM Overview" />
</p>

---

## 🚀 Getting Started

### 🛠 1. Environment Setup

```bash
conda env create --name FedSTM -f environment.yml
conda activate FedSTM
```

### ⚙️ 2. Configure and Run

Before running experiments, modify the following in `main.py`:

- **`hyperparameters01`**: Set `dataset`, `eta`, and `distill_lr` according to the dataset.
- **`--runs_name`**: Set your custom experiment name in the `parser` section.
- **`--ipc` / `--ipc2`**: Set images per class for early-stage and late-stage synthesis.

Then simply run:

```bash
python main.py
```

---

### 📂 Dataset Configurations

#### CIFAR-10

CIFAR-10 will be **downloaded automatically**. Set the following parameters:

```python
# In hyperparameters01
"dataset": ["cifar10"],  "eta": [0.4],  "distill_lr": [1e-4]
```

```python
# In parser
parser.add_argument('--ipc',  type=int, default=15, help='image(s) per class')
parser.add_argument('--ipc2', type=int, default=50, help='image(s) per class')
```

---

#### CINIC-10

Download the [CINIC-10 dataset](https://github.com/BayesWatch/cinic-10), extract it and place under `data/cinic10/`. The expected directory structure:

```
data/
└── cinic10/
    ├── README.md
    ├── imagenet-contributors.csv
    ├── synsets-to-cifar-10-classes.txt
    ├── test/
    ├── train/
    └── valid/
```

Set the following parameters:

```python
# In hyperparameters01
"dataset": ["cinic10"],  "eta": [0.8],  "distill_lr": [0.00025]
```

```python
# In parser
parser.add_argument('--ipc',  type=int, default=15, help='image(s) per class')
parser.add_argument('--ipc2', type=int, default=50, help='image(s) per class')
```

---

#### NWPU-RESISC45

Download the NWPU-RESISC45 dataset from [here](https://gcheng-nwpu.github.io/#Datasets) and place it under `data/NWPU_RESISC45/`. The expected directory structure:

```
data/
└── NWPU_RESISC45/
    ├── airplane/
    ├── bridge/
    ├── commercial_area/
    ├── golf_course/
    ├── island/
    ├── ...
    ├── runway/
    ├── stadium/
    └── wetland/
```

Set the following parameters:

```python
# In hyperparameters01
"dataset": ["NWPU_RESISC45"],  "eta": [0.4],  "distill_lr": [1e-4]
```

```python
# In parser
parser.add_argument('--ipc',  type=int, default=10, help='image(s) per class')
parser.add_argument('--ipc2', type=int, default=15, help='image(s) per class')
```

---

### 💾 GPU Memory Requirements

| Dataset | GPU | VRAM Usage |
|:---|:---|:---|
| CIFAR-10 | NVIDIA RTX 4090 | ~10.4 GB (10664 MB) |
| CINIC-10 | NVIDIA RTX 4090 | ~10.4 GB (10666 MB) |
| NWPU-RESISC45 | NVIDIA A800 | ~28 GB (28678 MB) |

> [!TIP]
> You can adjust `--batch_syn` and `--batch_syn2` to control GPU memory usage based on your hardware.

---

### 🎯 Adjusting Stage-wise Trajectory Matching

FedSTM uses two trajectory matching stages controlled by three key parameters (**t1**, **t2**, **t3** in the paper):

| Parameter | Location | Description |
|:---|:---|:---|
| `minimum_trajectory_length[0]` | `hyperparameters01` | **t1** — Starting round for early-stage trajectory matching |
| `--min_start_epoch2` | `parser` argument | **t2** — Starting point for late-stage expert trajectory |
| `minimum_trajectory_length[1]` | `hyperparameters01` | **t3** — Starting round for late-stage trajectory matching |

For example, the default configuration is:

```python
"minimum_trajectory_length": [[25, 150]]   # t1=25, t3=150
parser.add_argument('--min_start_epoch2', type=int, default=100)  # t2=100
```

> [!NOTE]
> For detailed analysis on the impact of different starting and ending rounds, please refer to the **Supplementary Materials**.

---

## 📊 Experimental Results

### Test Accuracy under Different Non-IID Settings

| | α = 0.01 | α = 0.02 | α = 0.04 | α = 0.1 |
|:---:|:---:|:---:|:---:|:---:|
| **CIFAR-10** | <img src="figure/cifar10_0.01.png" width="200"/> | <img src="figure/cifar10_0.02.png" width="200"/> | <img src="figure/cifar10_0.04.png" width="200"/> | <img src="figure/cifar10_0.1.png" width="200"/> |
| **CINIC-10** | <img src="figure/CINIC10_0.01.png" width="200"/> | <img src="figure/CINIC10_0.02.png" width="200"/> | <img src="figure/CINIC10_0.04.png" width="200"/> | <img src="figure/CINIC10_0.1.png" width="200"/> |
| **NWPU** | <img src="figure/NWPU_0.01.png" width="200"/> | <img src="figure/NWPU_0.02.png" width="200"/> | <img src="figure/NWPU_0.04.png" width="200"/> | <img src="figure/NWPU_0.1.png" width="200"/> |

### Impact of Different Starting and Ending Rounds

| t1 | t2 | t3 |
|:---:|:---:|:---:|
| <img src="figure/var_t1.png" width="280"/> | <img src="figure/var_t2.png" width="280"/> | <img src="figure/var_t3.png" width="280"/> |

---

## 🙏 Acknowledgments

This project builds upon the following open-source works:

- [**DynaFed**](https://github.com/pipilurj/DynaFed) — Tackling Client Data Heterogeneity with Global Dynamics
- [**MTT-Distillation**](https://github.com/GeorgeCazenavette/mtt-distillation) — Dataset Distillation by Matching Training Trajectories

---

## 📝 Citation

If you find this work useful, please cite our paper:

```bibtex
@ARTICLE{FedSTM,
  author={Shi, Xiufang and Zhang, Wei and Li, Yuheng and Wu, Mincheng and Li, Rushi and He, Shibo},
  journal={IEEE Transactions on Cognitive Communications and Networking}, 
  title={Federated Learning With Stage-Wise Trajectory Matching for AAV Networks on Non-IID Data}, 
  year={2026},
  volume={12},
  number={},
  pages={6698-6709},
  keywords={Data models;Training;Trajectory;Autonomous aerial vehicles;Servers;Federated learning;Data privacy;Privacy;Convergence;Computational modeling;Federated learning;autonomous aerial vehicles;dataset distillation;data heterogeneity},
  doi={10.1109/TCCN.2026.3670272}
}
```

---

## 📧 Contact

For questions or suggestions, please feel free to reach out:

**Wei Zhang** — [221122030290@zjut.edu.cn](mailto:221122030290@zjut.edu.cn) or [1923868444@qq.com](mailto:1923868444@qq.com)

---

<p align="center">
  If you find this project helpful, please consider giving it a ⭐
</p>
