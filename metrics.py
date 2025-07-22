from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import spearmanr
from evaluate import load as load_metric


def compute_sts_metrics(pred_scores, true_scores):
    corr = spearmanr(pred_scores, true_scores).correlation
    return {"spearman": corr}


def compute_classification_metrics(preds, labels):
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1": f1}


def compute_qa_metrics(preds, references):
    squad = load_metric("squad")
    formatted_preds = [
        {"id": str(i), "prediction_text": p} for i, p in enumerate(preds)
    ]
    formatted_refs = [
        {"id": str(i), "answers": {"text": [a], "answer_start": [0]}} for i, a in enumerate(references)
    ]
    results = squad.compute(predictions=formatted_preds, references=formatted_refs)
    return results


def compute_bleu(preds, references):
    bleu = load_metric("bleu")
    return bleu.compute(predictions=[[p.split()] for p in preds], references=[[[r.split()[0]]] for r in references])
