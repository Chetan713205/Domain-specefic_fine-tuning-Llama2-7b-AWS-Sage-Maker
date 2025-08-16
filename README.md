# 🇮🇳 **Indian Constitutional Law — Domain‑Specific Finetuning (Llama‑2‑7B‑Chat)**

> 🧠⚖️ High‑quality legal assistant fine‑tuned on Indian Constitutional Law to deliver precise, citation‑aware, and safety‑conscious answers.

[![Model](https://img.shields.io/badge/HF%20Model-Available-ffcc00?logo=huggingface)](https://huggingface.co/chetantiwari/Llama-2-7b-chat-finetune)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](#license)
[![Transformers](https://img.shields.io/badge/Transformers-%F0%9F%A6%84-0?logo=python\&labelColor=black\&color=green)](https://github.com/huggingface/transformers)

---

## ✨ Overview

This repository hosts a fine‑tuned **Llama‑2‑7B‑Chat** model specialized in **Indian Constitutional Law**. The model has been adapted on a curated corpus of constitutional provisions, case law summaries, doctrine explanations, and notable amendments to assist with:

* 📚 **Doctrinal explanations** (e.g., Basic Structure, Separation of Powers, Fundamental Rights)
* 🔎 **Provision lookup & synthesis** (Articles, Parts, Schedules, Amendments)
* 🏛️ **Case‑aware reasoning** (e.g., *Kesavananda Bharati*, *Maneka Gandhi*, *SR Bommai*)
* 🧩 **Plain‑language summaries** for students & practitioners
* 🛡️ **Safety‑aligned outputs** with disclaimers & refusal policies for legal advice

> 🚀 **Deployed model**: **Hugging Face** → **[`chetantiwari/Llama-2-7b-chat-finetune`](https://huggingface.co/chetantiwari/Llama-2-7b-chat-finetune)**

---

## 🏗️ Project Goals

* Create a **domain‑expert assistant** for Indian Constitutional Law.
* Encourage **transparent, citation‑ready outputs**.
* Provide a **practical baseline** for further legal‑domain finetunes (e.g., Criminal/Contract/Administrative Law).

---

## 🗂️ Dataset (High‑Level)

> *Note: Only public, educational, or permissible‑use texts were included. Proprietary or copyrighted sources were **excluded**.*

* 🔖 **Statutory content**: Constitution of India (Articles, Schedules, Amendments) — normalized & section‑indexed.
* 📜 **Landmark case briefs**: concise holdings, issues, ratios, and principles.
* 🧭 **Doctrines & tests**: Reasonable Classification, Proportionality, Wednesbury, Basic Structure, etc.
* 🧹 **Pre‑processing**: de‑duplication, regex cleanup, section tagging, article↔topic mapping.
* 🪄 **Instruction format**: (system, user, assistant) chat triples + chain‑of‑thought **not** included in supervised targets.

> If you want to extend/replace the dataset, see **`data/`** format spec below.

---

## 🔧 Training Pipeline

* **Base model**: `meta-llama/Llama-2-7b-chat`
* **Method**: Parameter‑efficient finetuning (e.g., **LoRA/QLoRA**) for compute efficiency
* **Precision**: 4‑bit or 8‑bit weights for memory‑constrained training
* **Loss**: Supervised cross‑entropy on instruction‑tuned chat format
* **Eval**: Held‑out prompts covering Articles, rights limitations, and doctrine applications
* **Safety**: Refusal templates for legal advice & enforcement of disclaimers

**Example training config (pseudo)**

```yaml
model_name: meta-llama/Llama-2-7b-chat
method: qlora
bnb_4bit: true
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
max_seq_len: 4096
per_device_train_batch_size: 8
gradient_accumulation_steps: 4
lr: 2e-4
epochs: 3
warmup_ratio: 0.03
save_strategy: steps
logging_steps: 25
```

---

## 🧪 Evaluation (Illustrative)

* ✅ **Article retrieval fidelity** (does the response cite correct Articles/Parts?)
* ✅ **Doctrinal accuracy** (correct test/application summaries)
* ✅ **Case consistency** (names, years, holdings not hallucinated)
* ⚠️ **Safety**: Refuses to provide formal legal advice; adds disclaimers

**Sample rubric**

* 0–2: Incorrect / hallucinated
* 3–4: Partially correct with gaps
* 5: Accurate, sourced, concise

---

## 🚀 Quick Start

### 1) Install

```bash
pip install transformers accelerate bitsandbytes einops sentencepiece
```

### 2) Load from Hugging Face

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_ID = "chetantiwari/Llama-2-7b-chat-finetune"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto"
)

def chat(prompt, system="You are a helpful assistant on Indian Constitutional Law."):
    # Simple chat template
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=0.2, top_p=0.9)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

print(chat("Explain the Basic Structure doctrine."))
```

### 3) Hugging Face Inference API (cURL)

```bash
curl -X POST \
  -H "Authorization: Bearer $HF_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"inputs": "Explain Article 21 and its expansion via Maneka Gandhi."}' \
  https://api-inference.huggingface.co/models/chetantiwari/Llama-2-7b-chat-finetune
```

> 💡 Tip: Keep **temperature low (0.1–0.3)** for fact‑heavy answers.

---

## 🗣️ Prompting Guide

* ✅ Ask **focused questions**: *“What is the scope of Article 19(1)(a) free speech, and what are the reasonable restrictions?”*
* ✅ Request **short citations**: *“Cite relevant Articles and landmark cases.”*
* ✅ Prefer **stepwise reasoning** for complex issues: *“List tests → apply to facts → conclude.”*
* ❌ Avoid asking for **legal advice** or strategies for ongoing litigation.

**Example prompts**

```text
Summarize the evolution of Article 21 post-Maneka Gandhi in 5 bullet points.
```

```text
Compare basic structure review vs. ordinary judicial review with 3 case examples.
```

---

## 📦 Repository Structure

```
.
├── data/
│   ├── constitution_articles.jsonl      # {id, article_no, text, topics}
│   ├── cases_summaries.jsonl            # {case_name, year, issue, holding}
│   └── doctrines.jsonl                  # {doctrine, definition, key_cases}
├── training/
│   ├── config.yaml                      # sample training config
│   ├── prepare_dataset.py               # cleaning & chat-formatting
│   ├── train_qlora.py                   # PEFT training loop
│   └── eval.py                          # lightweight eval metrics
├── notebooks/
│   └── main.ipynb                       # end-to-end demo (preprocess → train → eval)
├── scripts/
│   └── merge_lora_weights.py            # optional: merge + push to HF
├── app/
│   └── demo.py                          # (optional) Gradio/Streamlit app
├── LICENSE
└── README.md
```

---

## 🛡️ Safety, Use & Limitations

* ⚠️ **Not legal advice.** Outputs are for **education & research** only.
* 🧪 May hallucinate case names or dates — **verify before use**.
* 🌐 Jurisdiction‑specific nuances: model is **India‑centric**; not tuned for other jurisdictions.

**Recommended system prompt**

```text
You are a domain-specific assistant for Indian Constitutional Law. Provide concise, neutral answers with relevant Articles, Parts, Schedules, and 1–3 landmark cases. Add a disclaimer: “Educational use only, not legal advice.”
```

---

## 🧰 Reproducibility Notes

* Set `TORCH_DISABLE_MPS_FALLBACK=1` (macOS) if needed.
* Use `--gradient_checkpointing` for memory savings on long contexts.
* Determinism: `torch.manual_seed(42)` + controlled sampling parameters.

---

## 🗺️ Roadmap

* [ ] Expand dataset: recent Supreme Court constitutional benches
* [ ] RAG connector with SCC/Manupatra‑style citations (public alternatives only)
* [ ] Multi‑task finetune with QA + summarization + classification heads
* [ ] Robust eval set with **article‑span grounding**
* [ ] Distilled 3B/1.3B variants for on‑device use

---

## 🤝 Contributing

Contributions are welcome! Please open an **Issue** or **Pull Request** with:

* Clear description of changes
* Source of any added legal content (must be public/permitted)
* Minimal examples or unit tests where applicable

---

## 📜 License

This project is released under the **Apache 2.0 License**. See `LICENSE` for more details. Dataset components must respect their original licenses.

---

## 🙏 Acknowledgements

* Meta for Llama‑2 base model
* Hugging Face ecosystem (Transformers, Datasets, PEFT)
* Indian legal scholarship & public resources used for education

---

## 🔖 Citation

```bibtex
@software{tiwari2025indianconstllm,
  author  = {Chetan Tiwari},
  title   = {Indian Constitutional Law — Domain-Specific Finetuning (Llama-2-7B-Chat)},
  year    = {2025},
  url     = {https://huggingface.co/chetantiwari/Llama-2-7b-chat-finetune}
}
```

---

## 🌟 Star & Share

If this helped you, please ⭐ the repo and share with fellow learners & practitioners! 🙌
