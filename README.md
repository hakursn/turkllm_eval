# TurkLLM Eval

A minimal evaluation library for Turkish language LLMs. Supports standard tasks:

- ✅ Semantic Textual Similarity (STS)
- ✅ Classification
- ✅ Question Answering (QA)
- ✅ Translation

---

## 🔧 Installation
```bash
pip install -r requirements.txt
```

Or simply clone and import locally:
```bash
git clone https://github.com/hakursn/turkllm_eval.git
cd turkllm_eval
```

Then in your Python code:
```python
from turkllm_eval import eval_runner, config

results = eval_runner.evaluate_sts(config.DEFAULT_MODEL)
print(results)
```

---

## 🚀 Usage
### Command-line
```bash
python -m turkllm_eval.cli --task sts
python -m turkllm_eval.cli --task classification --model lmezenturk/bert-base-turkish-movie-sentiment
python -m turkllm_eval.cli --task qa
python -m turkllm_eval.cli --task translation
```

### Python Notebook
```python
from turkllm_eval import eval_runner, config

# STS
results_sts = eval_runner.evaluate_sts(config.DEFAULT_MODEL)
print(results_sts)

# Classification
from transformers import AutoTokenizer, AutoModelForSequenceClassification
model_cls = AutoModelForSequenceClassification.from_pretrained(config.DEFAULT_CLS_MODEL)
tokenizer_cls = AutoTokenizer.from_pretrained(config.DEFAULT_CLS_MODEL)
results_cls = eval_runner.evaluate_classification(model_cls, tokenizer_cls)
```

---

## 🧪 Available Datasets
| Task         | Dataset Name                      |
|--------------|------------------------------------|
| STS          | `figenfikri/stsb_tr`              |
| Classification | `tekir`, `turkish_movie_sentiment` |
| QA           | `turquad`, `mquad`                |
| Translation  | `wmt14`, `opus`                   |

---

## 📦 Project Structure
```
turkllm_eval/
├── cli.py
├── config.py
├── datasets/
│   ├── classification.py
│   ├── nli.py
│   ├── qa.py
│   ├── sts.py
│   └── translation.py
├── eval_runner.py
├── metrics.py
├── __init__.py
```

---

## 🎯 Roadmap
- [x] STS, Classification, QA, Translation evaluation
- [x] Turkish STS fallback using `figenfikri/stsb_tr`
- [x] Evaluation progress logging with tqdm
- [ ] Batch + CUDA acceleration across all tasks
- [ ] XNLI Natural Language Inference eval (Turkish)
- [ ] LoRA/PEFT support & detection
- [ ] CSV/JSON output & leaderboard mode
- [ ] HF Spaces integration
- [ ] Matplotlib/Seaborn based visualizations
- [ ] Embedding model evaluation & ranking

---

## 🤝 Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

### Development setup
```bash
git clone https://github.com/yourname/turkllm_eval.git
cd turkllm_eval
pip install -r requirements.txt
```

---

## 📄 License
MIT License

---

## ✨ Coming Soon
- LoRA evaluation hooks
- HuggingFace Hub integration
- Model card generation & export tools