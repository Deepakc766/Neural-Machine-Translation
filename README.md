# 🌏 Neural Machine Translation for Indian Languages  
**English → Hindi / Bengali**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)]()
[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg)]()

> **Author:** Deepak Chaurasia  
> **Roll No:** 220330
> deepakc22@iitk.ac.in
> **Institute:** Indian Institute of Technology Kanpur  
> **Course:** CS779 – Lifelong Learning with CIFAR-10 (Project: Machine Translation System for India)  
>  
> This repository implements and analyzes Neural Machine Translation (NMT) systems for **English → Hindi** and **English → Bengali**.  
> It explores **Seq2Seq**, **BiLSTM with Attention**, and **Transformer** architectures with various preprocessing, tokenization, and optimization strategies.

---

## 🧭 Table of Contents
1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Setup Instructions](#setup-instructions)
4. [Dataset Description](#dataset-description)
5. [Data Preprocessing](#data-preprocessing)
6. [Model Architectures](#model-architectures)
7. [Training Configuration](#training-configuration)
8. [Evaluation Metrics](#evaluation-metrics)
9. [Experimental Results](#experimental-results)
10. [Error Analysis](#error-analysis)
11. [Reproduction Commands](#reproduction-commands)
12. [Scripts & Notebooks](#scripts--notebooks)
13. [Troubleshooting](#troubleshooting)
14. [License](#license)
15. [Acknowledgements](#acknowledgements)

---

## 📘 Project Overview

This project focuses on **Neural Machine Translation (NMT)** for **low-resource Indic languages**, specifically translating from **English to Hindi** and **English to Bengali**.

The models were trained using multiple architectures — **Seq2Seq**, **BiLSTM + Attention**, and **Transformer** — with different tokenization methods (**BPE**, **SentencePiece**).  
The primary goal was to achieve the **highest chrF++** and **BLEU** scores while maintaining generalization across domains.

---

## 📁 Repository Structure
```bash
.
├── README.md
├── data/
│   ├── raw/                # Original datasets (train/val/test)
│   └── processed/          # Tokenized and cleaned data
├── src/
│   ├── preprocess.py       # Text cleaning & subword tokenization
│   ├── dataset.py          # PyTorch dataset utilities
│   ├── models/
│   │   ├── seq2seq.py
│   │   ├── bilstm_attention.py
│   │   └── transformer.py
│   ├── train.py            # Main training loop
│   ├── evaluate.py         # Evaluation (chrF, BLEU, ROUGE)
│   └── infer.py            # Generate translations from checkpoints
├── experiments/
│   ├── configs/            # Model config YAMLs
│   └── run_*.sh            # Example run scripts
├── notebooks/              # Jupyter notebooks for analysis
├── requirements.txt
└── LICENSE



| Dataset | Language Pair   | Train  | Validation | Test   |
| ------- | --------------- | ------ | ---------- | ------ |
| IndicMT | English–Bengali | 68,849 | 9,836      | 19,672 |
| IndicMT | English–Hindi   | 80,797 | 11,543     | 23,085 |


Validation and test targets were withheld for leaderboard scoring.

Average sentence length after tokenization: 30–50 tokens.

Augmentation: +100k parallel sentences from publicly available Indic corpora.

🧹 Data Preprocessing
Steps Performed

Whitespace normalization → collapse multiple spaces.

Unicode normalization → NFC/NFKC for Indic scripts.

Script filtering → retain only valid Devanagari or Bengali characters.

Lowercasing → applied to English text only.

Subword tokenization → BPE / SentencePiece (vocab size = 40k).

| Language | Range           | Description       |
| -------- | --------------- | ----------------- |
| Hindi    | `\u0900–\u097F` | Devanagari script |
| Bengali  | `\u0980–\u09FF` | Bengali script    |


| Week      | Submissions |
| --------- | ----------- |
| Week 1    | 0           |
| Week 2    | 2           |
| Week 3    | 6           |
| Week 4    | 4           |
| Week 5    | 4           |
| **Total** | **16**      |


| Model                  | Encoder     | Decoder     | Attention  | Hidden Size / d_model | Dropout | Layers | Notes              |
| ---------------------- | ----------- | ----------- | ---------- | --------------------- | ------- | ------ | ------------------ |
| **Seq2Seq (GRU/LSTM)** | LSTM        | LSTM        | None       | 256                   | 0.2     | 2      | Baseline           |
| **BiLSTM + Attention** | BiLSTM      | LSTM        | Luong      | 512                   | 0.1     | 2      | Strong performance |
| **Transformer (Best)** | Transformer | Transformer | Multi-head | 256                   | 0.1     | 4      | Best results       |


| Parameter             | Value                     |
| --------------------- | ------------------------- |
| Optimizer             | Adam                      |
| Learning Rate         | 5e-4                      |
| Scheduler             | Warmup + Decay            |
| Batch Size            | 64–128 (tokens-based)     |
| Epochs                | 20                        |
| Gradient Clipping     | 1.0                       |
| Teacher Forcing Ratio | 0.6–0.85                  |
| Loss Function         | CrossEntropy (ignore PAD) |

Experimental Results

| Metric     | Description              | Purpose                                         |
| ---------- | ------------------------ | ----------------------------------------------- |
| **chrF++** | Character n-gram F-score | Primary leaderboard metric                      |
| **BLEU**   | Word n-gram precision    | Secondary quality metric                        |
| **ROUGE**  | Recall-oriented          | Measures overlap for summarization-like scoring |

| Model                        | Validation chrF++ | Test chrF++ | BLEU  | Rank |
| ---------------------------- | ----------------- | ----------- | ----- | ---- |
| Transformer (4-blocks + Aug) | **0.32**          | **0.29**    | 0.073 | 78   |
| BiLSTM + Attention           | 0.28              | 0.25        | 0.060 | 100+ |
| Seq2Seq (Baseline)           | 0.22              | 0.20        | 0.041 | —    |

Error Analysis

Key Observations:

Overfitting on smaller datasets for deeper models (solved via augmentation).

Beam search sometimes caused repetition — greedy decoding often better.

SentencePiece gave slightly better morphology handling than BPE.

Transformer 4-layer achieved the best validation and test scores.

chrF++ was more reliable than BLEU for Indic scripts.

ransformer Config

name: transformer_4block
model:
  type: transformer
  d_model: 256
  nhead: 8
  num_encoder_layers: 4
  num_decoder_layers: 4
  dim_feedforward: 1024
  dropout: 0.1
training:
  optimizer: adam
  lr: 5e-4
  scheduler: warmup_decay
  warmup_steps: 4000
  batch_size: 4096
  epochs: 20
data:
  tokenizer: bpe
  vocab_size: 40000
  max_seq_len: 50
