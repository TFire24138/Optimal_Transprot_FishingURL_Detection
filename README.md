# 🚀 OTPhish

[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE) [![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/) [![Dataset](https://img.shields.io/badge/dataset-Phishing_URL-orange.svg)](https://huggingface.co/datasets/TFIRE24138/Phishing_URL) [![Status](https://img.shields.io/badge/status-Experimental-yellow.svg)]()

**OTPhish — Optimal Transport powered Semi-Supervised Phishing Detection**

---

## 🌐 Project Overview

**OTPhish** introduces an **Optimal Transport (OT)** based semi-supervised training framework for phishing website detection. The goal is to leverage OT to align labeled and unlabeled sample distributions while reducing the influence of noisy pseudo-labels.

✨ **Highlights:**

* Improve utilization of **unlabeled data** in phishing detection.
* Enhance robustness against **noisy pseudo-labels**.
* Provide a **reproducible and easy-to-run** research codebase.

---

## 🧠 Key Features

* ⚙️ OT-based semi-supervised loss aligning labeled and unlabeled distributions.
* 🛡️ Noise-robust pseudo-labeling with adaptive filtering.
* 🧩 Config-driven experiments for easy tuning and ablation.
* 📂 Modular design for training, evaluation, and data preprocessing.

---

## ⚡ Quick Start

### 1️⃣ Clone the repository

```bash
git clone https://github.com/TFire24138/Optimal_Transprot_FishingURL_Detection.git
cd OTPhish
```

### 2️⃣ Create the environment

```bash
conda env create -f environment.yml
conda activate otphish
```

> `environment.yml` includes PyTorch, scikit-learn, and Optimal Transport libraries (e.g., POT). Ensure your PyTorch build matches your CUDA version for GPU support.

### 3️⃣ Download the dataset

This project uses the **Phishing_URL** dataset, **collected by our team** and publicly released on Hugging Face:

* 🧾 Dataset page: [https://huggingface.co/datasets/TFIRE24138/Phishing_URL](https://huggingface.co/datasets/TFIRE24138/Phishing_URL)

Download and place the dataset under `data/phishing_url` (or update the path in configuration):

```bash
# Example using Hugging Face datasets library
python scripts/download_hf_dataset.py --dst data/phishing_url
```

### 4️⃣ Train

Training scripts are located in `Optimal_Transport/`.

🚀 **Train:**

```bash
python Optimal_Transport/train.py
```


## 🧾 Citation

If you use **OTPhish** in your research, please cite this repository (citation entry to be added upon publication).

---

## 📜 License

This project is released under the **MIT License**. See `LICENSE` for details.

---

## 💬 Contact

If you encounter issues or have questions, feel free to open an issue or contact the maintainers.

---

⭐ *Empowering phishing detection with Optimal Transport and semi-supervised learning — OTPhish.*
