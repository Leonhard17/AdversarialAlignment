# Adversarial Alignment

Adversarial Alignment is a research project that investigates how to estimate the uncertainty of large language models (LLMs) by analyzing their internal attention activations during inference. The overarching aim is to enable LLMs to reason dynamically, introspect on their own uncertainty, and support scalable oversight—key steps toward safer and more reliable AI systems.

This project was entirely developed by me, Leonhard Waibl.

---

## 🚀 Project Overview

This project explores whether the internal attention patterns of transformer-based LLMs can be used to predict the model’s own uncertainty about its outputs. By encoding attention maps as graphs and processing them with Graph Neural Networks (GNNs) and Transformer encoders, the system learns to regress a scalar uncertainty score for each model output.

**Key objectives:**
- Enable LLMs to introspect and estimate their own uncertainty.
- Support dynamic reasoning depth (e.g., more computation for harder problems).
- Lay groundwork for scalable, automated oversight of LLMs.

---

## ✨ Features

- **Uncertainty Prediction:** Learns to predict model uncertainty from internal attention activations.
- **Graph Neural Network Encoding:** Converts attention matrices into graph structures for GNN processing.
- **Transformer Aggregation:** Aggregates GNN outputs across layers and reasoning steps using Transformer encoders.
- **Stable Regression Loss:** Uses log loss for robust uncertainty regression.
- **Math Reasoning Prototype:** Initial experiments focus on mathematical problem-solving with fine-tuned GPT-2.

---

## 🏗️ Project Structure

```
AdversarialAlignment/
│
├── AttentionDataset/
│   ├── AttentionDataset.py         # Dataset and DataLoader logic for attention/reward data
│   ├── data/                      # Precomputed attention datasets (pickled)
│   └── ...                        # Utilities for dataset generation and processing
│
├── Model/
│   ├── full_model.py              # Full GNN + Transformer model definition
│   └── ...                        # GNN, aggregation, and reward modules
│
├── Trainer/
│   ├── train.py                   # Training script with W&B logging and checkpointing
│   └── sweep.py                   # (Optional) Hyperparameter sweep logic
│
├── test_model.py                  # Script for running a forward pass/test on a batch
├── config.py                      # Model and training configuration
├── requirements.txt               # Python dependencies
├── environment.yaml               # Conda environment specification
└── README.md                      # Project documentation
```

---

## 🛠️ Technologies Used

- **Python 3.10+**
- **PyTorch** (deep learning framework)
- **PyTorch Geometric** (graph neural networks)
- **Hugging Face Transformers** (LLM fine-tuning)
- **NetworkX** (graph construction)
- **Weights & Biases (wandb)** (experiment tracking)
- **NumPy, tqdm** (utilities and math)

---

## 🧑‍💻 Getting Started

### 1. Clone the repository

```bash
git clone <repo-url>
cd AdversarialAlignment
```

### 2. Set up the environment

```bash
conda env create -f environment.yaml
conda activate adverserialAlignment
```

### 3. Prepare the data

- Ensure you have a pickled attention dataset at `AttentionDataset/data/attention_dataset.pkl`.
- The dataset should be a tuple: `(attentions, rewards)`.

### 4. Train the model

```bash
python Trainer/train.py --data-pkl AttentionDataset/data/attention_dataset.pkl --batch-size 32 --epochs 10 --lr 1e-4 --project adversarial-alignment
```

- Training logs and checkpoints will be saved in the `Trainer/alignment_models/` directory.
- Training progress is tracked with Weights & Biases (wandb).

### 5. Test the model

```bash
python test_model.py
```

---

## 📊 Current Status

- [x] LLM fine-tuning on math data (GPT-2 small)
- [x] Attention extraction and dataset pipeline
- [x] GNN + Transformer uncertainty model implemented
- [x] Training and evaluation scripts functional
- [ ] Scaling to larger models and datasets (pending compute resources)

---

## 🔬 Research Focus

- **Introspective LLMs:** Can models estimate their own uncertainty from internal activations?
- **Dynamic Reasoning:** Adjusting computation depth based on predicted uncertainty.
- **Scalable Oversight:** Building tools for automated, scalable model monitoring and alignment.

---

## 🤝 Contributing & Contact

This project is developed independently as part of a broader research journey in AI alignment.  
Feedback, ideas, and collaboration are welcome!

- Email: leonhardwaibl@gmail.com

---

## 📄 License

This project is licensed under the MIT License.  
See the [LICENSE](LICENSE) file for details.

---

*Adversarial Alignment is a step toward more transparent, introspective, and reliable AI systems. Your feedback and suggestions are welcome!