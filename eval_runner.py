from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm  # 🔥 Add this line

from turkllm_eval.datasets import (
    load_sts_dataset,
    load_classification_dataset,
    load_qa_dataset,
    load_nli_dataset,
    load_translation_dataset,
)
from turkllm_eval.metrics import (
    compute_sts_metrics,
    compute_classification_metrics,
    compute_qa_metrics,
    compute_bleu,
)

def embed(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0]


def evaluate_sts(model_name):
    print("🔍 Loading STS dataset...")
    data = load_sts_dataset()
    print(f"✅ Loaded {len(data)} examples.")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    pred_scores = []
    true_scores = []

    print("🧠 Computing similarity scores...")
    for example in tqdm(data, desc="Evaluating STS"):
        v1 = embed(model, tokenizer, example["sentence1"])
        v2 = embed(model, tokenizer, example["sentence2"])
        score = torch.nn.functional.cosine_similarity(v1, v2).item()
        pred_scores.append(score)
        true_scores.append(example["score"])

    print("📊 Computing STS metric...")
    return compute_sts_metrics(pred_scores, true_scores)


def evaluate_classification(model, tokenizer, dataset_name="tekir"):
    print("🔍 Loading classification dataset...")
    data = load_classification_dataset(name=dataset_name)
    print(f"✅ Loaded {len(data)} examples.")

    preds, labels = [], []
    print("🧠 Running classification model...")
    for example in tqdm(data, desc="Evaluating Classification"):
        inputs = tokenizer(example["text"], return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=-1).item()
        preds.append(pred)
        labels.append(example["label"])

    print("📊 Computing classification metrics...")
    return compute_classification_metrics(preds, labels)


def evaluate_qa(model, tokenizer, dataset_name="turquad"):
    print("🔍 Loading QA dataset...")
    data = load_qa_dataset(name=dataset_name)
    print(f"✅ Loaded {len(data)} examples.")

    preds, answers = [], []
    print("🧠 Generating answers...")
    for example in tqdm(data, desc="Evaluating QA"):
        prompt = f"Soru: {example['question']}\nParça: {example['context']}\nCevap:"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=32)
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        preds.append(decoded.split("Cevap:")[-1].strip())
        answers.append(example["answers"][0])

    print("📊 Computing QA metrics...")
    return compute_qa_metrics(preds, answers)


def evaluate_translation(model, tokenizer):
    print("🔍 Loading translation dataset...")
    data = load_translation_dataset()
    print(f"✅ Loaded {len(data)} examples.")

    preds, targets = [], []
    print("🧠 Translating...")
    for example in tqdm(data, desc="Evaluating Translation"):
        inputs = tokenizer(example["source"], return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=64)
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        preds.append(decoded)
        targets.append(example["target"])

    print("📊 Computing BLEU score...")
    return compute_bleu(preds, targets)
