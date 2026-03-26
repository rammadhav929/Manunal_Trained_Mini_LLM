# Mini GPT Language Model Training (Educational Project) 

## Overview
This project demonstrates training a **small GPT-style causal language model from scratch** using the Hugging Face Transformers Trainer API.

The objective of this repository is to understand the **end-to-end LLM training pipeline**, including tokenization, custom model configuration, training workflow, and checkpoint management.

> ⚠️ This model is intentionally small and trained on limited data. It is built for learning purposes and is not intended for production usage.

---

## Learning Objectives

- Understand causal language modeling (next-token prediction)
- Configure a custom GPT architecture
- Use Hugging Face Trainer API for training
- Handle tokenization, padding, and batching
- Manage training checkpoints and model saving
- Observe hardware impact (CPU vs GPU training)

---

## Model Architecture

- Architecture: GPT-style Transformer (trained from scratch)
- Context Window: **256 tokens**
- Embedding Dimension: **384**
- Transformer Layers: **6**
- Attention Heads: **6**
- Training Objective: **Causal Language Modeling**

---

## Dataset

- **WikiText-2 Raw**
- Public benchmark dataset commonly used for language modeling experiments  
- Small dataset suitable for educational training runs

---

## Training Configuration

- Framework: **PyTorch + Hugging Face Transformers**
- Trainer API used for training loop abstraction
- Epochs: **2**
- Batch Size: **8**
- Mixed Precision: Enabled when CUDA is available
- Periodic checkpoint saving enabled

---

## Hardware Notes

- Designed to run on **CPU or low-VRAM GPUs (e.g., GTX 1650)**
- Training on CPU is significantly slower
- Model size and context window were kept small due to hardware limitations

---

## Repository Structure
