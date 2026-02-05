# Sauti AI: Production-Quality Swahili Conversational AI 🎯

**A novel two-stage methodology achieving 3.97 perplexity with 9-hour training on single GPU**

[![GitHub Stars](https://img.shields.io/github/stars/briankanaru181-jpg/NIRU_HACKATHON?style=social)](https://github.com/briankanaru181-jpg/NIRU_HACKATHON)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hackathon: NIRU 2026](https://img.shields.io/badge/Hackathon-NIRU_2026-blue)](https://niru.ai)

##  Executive Summary

**Sauti AI** delivers production-ready Swahili conversational AI through an innovative two-stage training methodology. Our system achieves **exceptional performance (3.97 validation perplexity)** with **remarkable efficiency (9 hours total training)** using only **0.5% trainable parameters** via LoRA + 4-bit quantization.

###  Key Results
| Metric | Value | Improvement |
|--------|-------|-------------|
| **Validation Perplexity** | **3.97** | 32.3% reduction |
| **Total Training Time** | **9 hours** | Single P100 GPU |
| **Trainable Parameters** | **0.5%** | LoRA + 4-bit quantization |
| **Swahili Corpus** | **12M tokens** | Literary translations included |
| **Conversational Examples** | **3,175** | Expert-validated quality |

##  Innovative Architecture

###  Two-Stage Methodology
```mermaid
graph LR
    A[12M Token Corpus] --> B[Stage 1: Pretraining];
    C[swahili-gemma-1b] --> B;
    B --> D[Linguistic Foundation];
    E[3,175 Dialogues] --> F[Stage 2: Fine-tuning];
    D --> F;
    F --> G[Production Model];
    G --> H[Kiswahili Learning Assistant];
    G --> I[General Conversation];
   
    Stage 1: Continued Pretraining (6 hours)

    12M token curated corpus with diverse Swahili sources

    Literary translations of classics (Chekhov, Tolstoy, Stevenson)

    Cultural narrative integration from traditional folktales

    Training loss: 1.080

Stage 2: Conversational Fine-tuning (2.5 hours)

    3,175 high-quality dialogue examples

    7 epochs to convergence with smooth training curves

    Validation loss: 1.387

    Final perplexity: 3.97

    Quick Start
Prerequisites

    Python 3.9+

    CUDA-capable GPU (16GB VRAM recommended)

    Git LFS for large files

   INSTALLATION
    # Clone repository
git clone https://github.com/briankanaru181-jpg/NIRU_HACKATHON.git
cd NIRU_HACKATHON

# Install dependencies
pip install -r requirements.txt

# Setup Git LFS for large files
git lfs install
git lfs pull

Reproduce Full Training
# Stage 1: Pretraining
python src/training_pipeline.py --stage=pretraining \
    --config=configs/pretraining.yaml \
    --output_dir=models/pretrained

# Stage 2: Fine-tuning
python src/training_pipeline.py --stage=finetuning \
    --base_model=models/pretrained \
    --config=configs/finetuning.yaml \
    --output_dir=models/final

Launch Dual Interface
# Educational RAG Assistant
python deployment/kiswahili_assistant.py --mode=education

# General Conversation Assistant
python deployment/general_assistant.py

Live Results & Demos
    📈 Training Dashboard: Weights & Biases Logs

    🎥 Video Demonstration: YouTube Walkthrough

    🤖 Live Chat Interface: Gradio Demo

    📚 Educational RAG System: Swahili Learning Portal

Repository Structure
NIRU_HACKATHON/
├── 📁 Corpus/                           # Training datasets (12M tokens)
│   ├── literary_translations/          # Chekhov, Tolstoy, Stevenson
│   ├── swahili_literature/             # Traditional & modern works
│   ├── conversational_data/            # 3,175 dialogue examples
│   └── processed/                      # Tokenized datasets
├── 📁 src/                             # Training pipeline
│   ├── training_pipeline.py            # Main training script
│   ├── data_processing.py              # Corpus preprocessing
│   ├── model_utils.py                  # LoRA + quantization setup
│   └── evaluation.py                   # Metrics calculation
├── 📁 notebooks/                       # Analysis & visualization
│   ├── token_analysis.ipynb            # Corpus token statistics
│   ├── training_curves.ipynb           # Loss/perplexity visualization
│   └── qualitative_evaluation.ipynb    # Human evaluation results
├── 📁 deployment/                      # Production interfaces
│   ├── dual_interface/                 # Educational + General chat
│   ├── rag_pipeline/                   # Kiswahili learning assistant
│   └── asr_integration/                # Omnilingual ASR for 7 languages
├── 📁 configs/                         # Training configurations
│   ├── pretraining.yaml                # Stage 1 parameters
│   ├── finetuning.yaml                 # Stage 2 parameters
│   └── inference.yaml                  # Deployment settings
├── 📁 results/                         # Validation metrics
│   ├── training_logs/                  # Complete training history
│   ├── model_checkpoints/              # Best model versions
│   └── evaluation_metrics/             # Quantitative results
├── 📁 docs/                            # Documentation
│   ├── EVALUATION_GUIDE.md             # For hackathon judges
│   ├── METHODOLOGY.md                  # Technical details
│   └── API_REFERENCE.md                # Deployment API docs
├── requirements.txt                    # Python dependencies
├── .gitattributes                      # Git LFS configuration
└── README.md                           # This file

Technical Innovation
Parameter-Efficient Training
# LoRA Configuration (0.5% trainable parameters)
lora_config = LoraConfig(
    r=8,                    # Rank
    lora_alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 4-bit Quantization (4x memory reduction)
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

Literary Translation Pipeline

Our novel approach includes first-ever Swahili translations of:

    Anton Chekhov's "The Duel" - Psychological dialogue patterns

    Leo Tolstoy's "The Death of Ivan Ilyich" - Philosophical discourse

    Robert Louis Stevenson's "Dr. Jekyll and Mr. Hyde" - Narrative complexity

Omnilingual ASR Integration

Supporting 7 Kenyan languages for speech-to-text:

    Swahili, Kikuyu, Kamba, Luhya, Luo, Kalenjin, Maasai

Performance Validation
Quantitative Metrics
Epoch	Training Loss	Validation Loss	Perplexity
1	2.15	2.31	10.07
2	1.87	1.98	7.25
3	1.64	1.72	5.58
4	1.43	1.55	4.71
5	1.28	1.43	4.18
6	1.18	1.39	4.01
7	1.12	1.387	3.97
Qualitative Excellence

    Cultural Authenticity: Native speaker validation throughout

    Linguistic Sophistication: Literary translations enhance depth

    Dialogue Quality: Multi-turn conversation coherence

    Educational Value: Curriculum-aligned RAG system

 License

    Code: MIT License - See LICENSE file

    WAXAL Dataset: CC-BY-SA-4.0

    Gemma Model: Google Gemma Terms of Use

    Literary Translations: Public Domain + Creative Commons

Acknowledgments

    NIRU AI for the hackathon platform and opportunity

    Crane AI Labs for the swahili-gemma-1b base model

    Native Swahili speakers and linguists for cultural validation and translations

    Open-source community for invaluable tools and libraries

    Meta AI for Omnilingual ASR technology

References & Citations

If you use this work, please cite:
@software{sauti_ai_2026,
  author = {Kanaru, Brian},
  title = {Sauti AI: Production-Quality Swahili Conversational AI},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/briankanaru181-jpg/NIRU_HACKATHON}}
}
