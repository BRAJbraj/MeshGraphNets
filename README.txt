# PressNet: Mesh-Based Simulation with Graph Neural Networks

![Python](https://img.shields.io/badge/Python-3.7-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-1.15-orange.svg)
![Status](https://img.shields.io/badge/Status-Proof_of_Concept-yellow.svg)

> **🚧 Work in Progress:** This repository is currently in the **validation phase**. The model has been successfully trained on a single data group (`group_0`) to verify its ability to capture nonlinear deformation dynamics. Scaling to larger trajectory sets is the next step.

## 📌 Overview
**PressNet** is an implementation of **MeshGraphNets** (based on the paper by DeepMind) tailored for simulating complex physical deformations, specifically focusing on **industrial metal pressing and nonlinear material deformation**.

The goal of this project is to learn mesh-based simulations using Graph Neural Networks (GNNs). The current implementation handles the interaction between "Tool" (kinematic) and "Metal" (dynamic) meshes using a hybrid velocity approach.

---

## 🎥 Preliminary Results (Proof of Concept)
*Current progress: The model was trained on `group_0` data to test if the GNN architecture could successfully capture the nonlinearity of the deformation.*

The GIFs below demonstrate that the model **successfully learned the deformation mechanics** for this specific group, validating the core architecture.

| View 1: Deformation Dynamics | View 2: Stress/Strain Visualization |
| :---: | :---: |
| | View 1: Deformation Dynamics | View 2: Stress/Strain Visualization |
| :---: | :---: |
| <img src="./images/rollout_pressnet_rollout_group0_50k_radius_25_result.gif" width="100%"> | *Coming Soon* |

*(Note: These results represent the model over-fitting to `group_0` as a sanity check before scaling up to the full dataset.)*

---

## 🚀 Key Features
* **Encoder-Processor-Decoder Architecture:** Implements the core GNN structure to pass messages between mesh nodes.
* **Hybrid Velocity Handling:** Differentiates between:
    * **Tool Nodes:** Driven by kinematic target positions.
    * **Metal Nodes:** Driven by physical forces/previous velocity.
* **Nonlinearity Capture:** Validated capability to learn complex, non-linear mesh deformations.
* **History Windowing:** Utilizes `t-1`, `t`, and `t+1` states to capture temporal dynamics.
* **Custom Early Stopping:** Implements a "Best Model" saver to ensure the optimal weights are preserved during training.

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/pressnet-meshgraphnets.git](https://github.com/your-username/pressnet-meshgraphnets.git)
    cd pressnet-meshgraphnets
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Download the dataset:**
    * *Data is currently private/local. Place `meta.json`, `train.tfrecord`, and `valid.tfrecord` files in your dataset directory (e.g., `./data/`).*
    * **Dataset Details:** For a detailed breakdown of the data structure, variable normalization, and mesh properties, please refer to the [PressNet Dataset Specifications](https://github.com/AnK-Accelerated-Komputing/PressNet/tree/main/datasets#details-of-pressnet-dataset).

## 💻 Usage

### Training
To train the model on the dataset:

```bash
python pressnet_run_model.py \
  --mode=train \
  --model=pressnet \
  --dataset_dir=./data \
  --checkpoint_dir=./checkpoints \
  --batch_size=2
Evaluation
To run a rollout and generate a trajectory file from the saved best_model:

Bash

python pressnet_run_model.py \
  --mode=eval \
  --model=pressnet \
  --dataset_dir=./data \
  --checkpoint_dir=./checkpoints \
  --rollout_path=output/rollout.pkl
📉 Development Roadmap
I am currently working on the following improvements:

[x] Core Architecture: Implement Encoder, Processor (Message Passing), and Decoder.

[x] Proof of Concept: Verify model can capture nonlinearity on single-group data (group_0).

[x] Data Loading: Efficient TFRecord parsing with history buffering.

[x] Training Logic: Integrated Early Stopping and "Best Model" saving.

[ ] Scaling Up: Train on higher number of trajectories (multi-group) to generalize the physics.

[ ] Main Dataset Integration: Final training on the complete industrial dataset.

[ ] Long-term Stability: Reducing error accumulation over long rollouts (>500 steps).

📚 Acknowledgements & References
This project is built upon the foundational research in graph-based physical simulations.

Original Paper: "Learning Mesh-Based Simulation with Graph Networks" by Pfaff et al. (ICML 2021). Read here.

DeepMind Implementation: This code is heavily inspired by the official DeepMind research repository. View Repository.

PressNet Dataset: Specifications and data handling protocols. View Details.
