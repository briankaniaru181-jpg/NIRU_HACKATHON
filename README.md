# Sauti AI

A Reproducible, Parameter-Efficient Framework for African Language Conversational AI

A two-stage training methodology for culturally grounded Swahili and Kenyan languages

## Executive Summary

Sauti AI is a production-oriented conversational AI framework designed for African languages, beginning with Swahili and extending systematically to Kikuyu and other Kenyan languages. The project introduces a validated two-stage training methodology—continued pretraining followed by conversational fine-tuning—implemented using parameter-efficient techniques (LoRA + 4-bit quantization) on accessible hardware.

An initial Swahili implementation of this methodology achieved a validation perplexity of 3.97, with end-to-end training completed in approximately 9 hours on a single P100 GPU. During the NIRU AI Hackathon 2026, this repository serves both as:

1) A record of validated results, and

2) A reproducible framework, which will be re-executed publicly during the competition period.

Because training outcomes depend on stochastic initialization, hardware variance, and data ordering, future runs may produce slightly different metrics. Accordingly, this repository emphasizes methodological reproducibility and engineering rigor, not one-off numerical claims.

### What This Repository Represents

✅ A proven training architecture for low-resource African languages

✅ A production-quality Swahili MVP, already trained and deployed

🔄 A live reproducibility exercise, to be re-run during the competition

🔜 A foundation for Kikuyu expansion, using the same pipeline

This distinction is intentional and central to the project’s scientific and engineering integrity.

### Core Methodology: Two-Stage Training
### Stage 1 — Continued Pretraining

#### Linguistic & Cultural Foundation

##### Objective
To strengthen the base model’s internal representation of Swahili grammar, narrative structure, metaphor, and cultural context before any conversational specialization.

Validated Swahili configuration:

- Base model: CraneAILabs/swahili-gemma-1b

- Corpus size: ~18 million tokens

- Epochs: 2

- Duration: ~6 hours

- Hardware: Single P100 GPU (16GB VRAM)

- Training method: LoRA + 4-bit NF4 quantization

##### Corpus composition

- Classical and modern Swahili literature

- Folktales and Zanzibar narratives

- Swahili Bible and canonical prose

- Expert-validated Swahili translations of public-domain world literature

- News, encyclopedic, and formal registers

This stage establishes deep linguistic grounding, reducing overfitting and instability during dialogue fine-tuning.

### Stage 2 — Conversational Fine-Tuning

#### Dialogue Specialization

##### Objective
To teach the model how to engage in natural, culturally appropriate conversation, without erasing the linguistic depth acquired during pretraining.

Validated Swahili configuration

- Dataset size: 3,175 curated dialogue examples

- Train/validation split: 2,857 / 318

- Epochs: 7

- Duration: ~2.5 hours

- Formatting: Gemma-style conversational tokens

Observed outcome (one validated run)

- Training loss: ~1.12

- Validation loss: ~1.387

- Validation perplexity: 3.97

Important: These values are documented results. Re-executions may vary modestly while remaining within the same performance.

#### Parameter-Efficient Training Strategy

Sauti AI is designed to be computationally accessible, not infrastructure-heavy.

##### LoRA Configuration

- Rank: 8

- Alpha: 16

- Target modules: q_proj, k_proj, v_proj, o_proj

- Dropout: 0.05

- Trainable parameters: ~0.5% of the base model

##### Quantization

- 4-bit NF4 with double quantization

- FP16 compute

- ~4× memory reduction

This enables single-GPU training without measurable degradation in conversational quality.

#### 🚀 Quick Start (Minimal & Reproducible)

This section provides a minimal, example-based entry point for running the Sauti AI training pipeline.

Note: Exact losses and perplexity values may vary across runs due to stochastic training dynamics, hardware differences, and data ordering.

#### Prerequisites

Python 3.9+

CUDA-capable GPU (16GB VRAM recommended)

Git LFS (for large model and dataset files)

Installation
### Clone the repository
git clone https://github.com/briankaniaru181-jpg/NIRU_HACKATHON.git
cd NIRU_HACKATHON

### Install dependencies
pip install -r requirements.txt

### Initialize Git LFS
git lfs install
git lfs pull

Stage 1 — Continued Pretraining (Example)
python src/training_pipeline.py \
  --stage pretraining \
  --config configs/pretraining.yaml \
  --output_dir models/pretrained


Typical runtime (validated Swahili configuration): ~6 hours on a single P100 GPU.

Stage 2 — Conversational Fine-Tuning (Example)
python src/training_pipeline.py \
  --stage finetuning \
  --base_model models/pretrained \
  --config configs/finetuning.yaml \
  --output_dir models/final


Typical runtime (validated Swahili configuration): ~2–3 hours on a single GPU.

Inference (Optional)
python src/inference.py \
  --model_path models/final \
  --prompt "Habari, unaweza kunisaidia?"

## Reproducibility Commitment

During the NIRU AI Hackathon 2026, I will:

- Re-execute Stage 1 (continued pretraining) from scratch

- Re-execute Stage 2 (conversational fine-tuning)

- Publish training logs, checkpoints, and metrics

- Document deviations from prior runs

The objective is not to reproduce a single number, but to demonstrate that the methodology reliably produces high-quality African-language conversational models under constrained resources.

## Dual-Interface System (Implemented)

A single trained model serves three production interfaces:

### 1. Kiswahili Learning Assistant (RAG)

- Curriculum-aligned knowledge base (Forms 1–4)

- Semantic retrieval

- Confidence scoring for educational reliability

### 2. General Conversation Assistant

- Open-domain dialogue

- Creative and casual interaction

- Prompt-guided generation

### 3. Speech-to-Text Integration (Planned & In Progress)

Using Meta’s Omnilingual ASR, Sauti AI supports transcription in:

Swahili • Kikuyu • Kamba • Luhya • Luo • Kalenjin • Maasai

This enables:

- Voice-based interaction

- Subtitle generation for creators

- Preservation of oral traditions for low-resource languages

## Extension to Kikuyu

The same two-stage framework is being applied to Kikuyu:

- Stage 1 target: 4–6M tokens

- Stage 2 target: ~1,500 conversations

Data sources include oral traditions transcribed via ASR, web-scraped content,  public-domain literature, and native-speaker-validated text

Swahili serves as the proof of methodology.


## Current limitations

- Conversational dataset size remains modest

- Literary translations may retain subtle cultural transfer artifacts

- Evaluation relies primarily on perplexity and expert review

## Future work

- Expansion to additional Kenyan and African languages

- Larger conversational datasets

- Domain-specific assistants (education, health, agriculture)

- Broader human evaluation studies

- Multimodal extensions beyond speech-to-text