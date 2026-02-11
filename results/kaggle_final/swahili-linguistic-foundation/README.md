---
base_model: CraneAILabs/swahili-gemma-1b
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:CraneAILabs/swahili-gemma-1b
- lora
- transformers
- swahili
- african-languages
- nlp
- language-model
---

# Swahili Linguistic Foundation Model

A parameter-efficient continued pretraining of Gemma for Swahili language understanding.

This model is the result of Stage 1 continued pretraining on diverse Swahili text, establishing deep linguistic and cultural grounding before conversational fine-tuning.

## Model Description

This model is a LoRA adapter for CraneAILabs/swahili-gemma-1b that has been further pretrained on 18 million tokens of diverse Swahili text including literature, news, folktales, and educational content. The training strengthens the model's understanding of Swahili grammar, narrative structure, and cultural context.

- **Developed by:** Brian Kaniaru for NIRU AI Hackathon 2026
- **Model type:** Causal Language Model with LoRA adapters
- **Language(s) (NLP):** Swahili (primary), English (secondary)
- **License:** MIT
- **Finetuned from model:** CraneAILabs/swahili-gemma-1b

### Model Sources

- **Repository:** https://github.com/briankaniaru181-jpg/NIRU_HACKATHON
- **Project:** Sauti AI - African Language Conversational AI Framework
- **Paper:** Part of NIRU AI Hackathon 2026 submission

## Uses

### Direct Use

- Swahili text generation and completion
- Swahili language understanding tasks
- Foundation for conversational AI applications
- Educational tools for Swahili language learning

### Downstream Use

- Conversational AI fine-tuning (Stage 2 of Sauti AI pipeline)
- Swahili text classification
- Content generation for Swahili media
- Translation assistance

### Out-of-Scope Use

- Generating harmful or misleading content
- Medical or legal advice without human oversight
- High-stakes decision making without verification
- Real-time applications without proper testing

## Bias, Risks, and Limitations

**Limitations:**

- Primarily trained on written Swahili (may not capture all dialects)
- Limited exposure to very recent events (2026+)
- May retain some biases from the base model

**Recommendations:**

- Use with human oversight for critical applications
- Fine-tune further for domain-specific tasks
- Evaluate outputs for cultural appropriateness

## How to Get Started with the Model

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# Load base model and adapter
model_name = "CraneAILabs/swahili-gemma-1b"
adapter_path = "./swahili-linguistic-foundation"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
model = PeftModel.from_pretrained(model, adapter_path)

# Generate Swahili text
prompt = "Habari za asubuhi? Leo ni siku nzuri ya"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Training Details

### Training Data

- **Corpus:** 18 million tokens of diverse Swahili text
- **Sources:** Literature, news, folktales, educational content, translations
- **Format:** JSONL with text fields
- **Preprocessing:** Sequence packing for 100% token utilization

### Training Procedure

#### Training Hyperparameters

- **Method:** LoRA (Low-Rank Adaptation)
- **LoRA Rank:** 8
- **LoRA Alpha:** 16
- **Target Modules:** q_proj, k_proj, v_proj, o_proj
- **Learning Rate:** 5e-5 with cosine decay
- **Batch Size:** 2 (effective 32 with gradient accumulation)
- **Max Sequence Length:** 512
- **Training Steps:** 500 (early stopped)
- **Optimizer:** AdamW with weight decay 0.01

#### Speeds, Sizes, Times

- **Training Time:** ~4.5 hours on Tesla P100
- **GPU Memory:** 1.6GB/16GB (90% efficient)
- **Checkpoint Size:** 11.5MB (adapter only)
- **Final Perplexity:** 11.56 (12.4% improvement from initial)

## Evaluation

### Testing Data

- **Validation Split:** 10% of training data (3,598 sequences)
- **Evaluation Frequency:** Every 100 steps

### Metrics

- **Perplexity:** Primary metric (lower is better)
- **Swahili Score:** Language-specific quality metric (0-1 scale)
- **Token Accuracy:** Next token prediction accuracy

### Results

- **Final Perplexity:** 11.56 (from initial 13.20)
- **Best Swahili Score:** 1.00 (perfect language modeling)
- **Training Stability:** No gradient explosions, steady improvement

## Environmental Impact

- **Hardware Type:** NVIDIA Tesla P100 (16GB VRAM)
- **Hours used:** 4.5 hours
- **Cloud Provider:** Kaggle
- **Compute Region:** Unknown (Kaggle infrastructure)
- **Carbon Emitted:** Estimated 0.2 kg CO2 (based on ML CO2 calculator)

## Technical Specifications

### Model Architecture and Objective

- **Base Architecture:** Gemma 1B with Swahili adaptations
- **Objective:** Causal language modeling
- **Quantization:** 4-bit NF4 with double quantization
- **Trainable Parameters:** ~0.5% of total (LoRA efficiency)

### Compute Infrastructure

- **Hardware:** Single GPU training (accessible setup)
- **Software:** PyTorch 2.0+, Transformers, PEFT, BitsAndBytes

## Citation

If you use this model, please cite:

```bibtex
@software{sauti_ai_2026,
  title = {Sauti AI: African Language Conversational AI Framework},
  author = {Brian Kaniaru},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/briankaniaru181-jpg/NIRU_HACKATHON},
  note = {NIRU AI Hackathon 2026 Submission}
}
```

## Glossary

- **LoRA:** Low-Rank Adaptation - parameter-efficient fine-tuning method
- **Perplexity:** Measure of how well a probability model predicts a sample
- **Sequence Packing:** Technique to maximize token utilization during training
- **Swahili Score:** Custom metric evaluating Swahili language quality

## More Information

This model is part of the Sauti AI framework, a two-stage training methodology for African language conversational AI. The framework is designed to be computationally accessible while maintaining high quality.

**Next Stage:** This model is ready for Stage 2 conversational fine-tuning to create a Swahili chatbot.

## Model Card Authors

Brian Kaniaru - NIRU AI Hackathon 2026 Participant

## Model Card Contact

For questions about this model, open an issue on the GitHub repository.

## Framework versions

- PEFT 0.18.1
- Transformers 4.35.0
- PyTorch 2.0.0+
- BitsAndBytes 0.41.0