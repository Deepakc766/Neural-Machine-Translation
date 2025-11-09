
Rank in Training phase : 67/150

Rank in testing phase :78/150

final score=0.32

# Neural Machine Translation (English to Indian Languages)

This repository contains the implementation of a Neural Machine Translation (NMT) system designed to translate English sentences into Indian languages, specifically Hindi and Bengali. The project is developed using the sequence-to-sequence (encoder-decoder) paradigm and follows a systematic progression from basic recurrent models to attention-based architectures. Work on transformer-based models is currently in progress. All models are trained from scratch using PyTorch, without relying on pre-trained transformer language models, in accordance with competition rules.

---

## 1. Problem Statement

The objective of this project is to build an NMT system capable of generating accurate translations from English to Indian languages. The task follows a supervised learning setup where the model is trained on parallel sentence pairs. The system is evaluated using established MT metrics such as BLEU, chrF++, and ROUGE. The broader goal is to study the impact of different encoder-decoder architectures on translation quality while keeping the training setup and data constant.

---

## 2. Model Experiments and Results

Multiple architectures were implemented and evaluated. The performance of each model on the development set is summarized below:

| Model | Architecture Summary | BLEU Score |
|--------|----------------------|------------|
| Seq2Seq (RNN) | Vanilla encoder-decoder with recurrent units | 0.20 |
| Seq2Seq (LSTM) | LSTM encoder-decoder to address vanishing gradients | 0.21 |
| Bi-LSTM | Bidirectional encoder to capture richer context | 0.23 |
| LSTM with Attention | Bahdanau additive attention for better alignment | 0.24 |
| Transformer | Encoder-decoder self-attention architecture | 0.32 |

Observations:

1. Introduction of LSTMs improved learning stability and produced slightly higher BLEU scores compared to vanilla RNNs.
2. Bidirectional encoders captured longer dependencies, resulting in noticeable improvements.
3. Attention mechanisms further improved translation quality by learning source-target alignment.
4. Transformer implementation is expected to outperform RNN-based models once fully trained and optimized.

---

## 3. Key Features

- End-to-end encoder-decoder translation models implemented in PyTorch.
- Static word embeddings (Word2Vec / GloVe) used where applicable.
- Training performed on Google Colab GPUs.
- Tokenization and vocabulary generation through standard preprocessing routines.
- Decoding methods include greedy decoding (beam search and no-repeat-ngram planned).
- Evaluation supported through BLEU, ROUGE, and chrF++ metrics.
- Modular codebase to enable easy switching between model types.

---

## 4. Repository Structure



---

## 6. Future Work

1. Complete transformer model implementation and tuning.
2. Add beam search with configurable beam size (default planned: 5).
3. Add no-repeat n-gram decoding penalties for repetition control.
4. Explore embedding sharing between encoder and decoder.
5. Improve regularization to reduce exposure bias during decoding.

---

## 7. Dataset and Restrictions

The dataset is provided as part of the Machine Translation Competition focused on English to Indian languages. External datasets, pre-trained LLMs, and pre-trained transformer models are not allowed. Models must be trained from scratch. :contentReference[oaicite:0]{index=0}

---

## 8. Author

Deepak Chaurasia 
Neural Machine Translation Research and Development

