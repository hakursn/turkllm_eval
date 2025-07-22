import argparse
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, AutoModelForQuestionAnswering

from turkllm_eval import config, eval_runner


def main():
    parser = argparse.ArgumentParser(description="Evaluate Turkish LLM on a selected task.")
    parser.add_argument("--task", choices=config.EVAL_TASKS, required=True, help="Evaluation task name")
    parser.add_argument("--model", default=None, help="HuggingFace model name or path")
    args = parser.parse_args()

    task = args.task
    model_name = args.model or {
        "sts": config.DEFAULT_MODEL,
        "classification": config.DEFAULT_CLS_MODEL,
        "qa": config.DEFAULT_QA_MODEL,
        "translation": config.DEFAULT_TRANSLATION_MODEL,
    }.get(task, config.DEFAULT_MODEL)

    if task == "sts":
        result = eval_runner.evaluate_sts(model_name)
    elif task == "classification":
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        result = eval_runner.evaluate_classification(model, tokenizer)
    elif task == "qa":
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        result = eval_runner.evaluate_qa(model, tokenizer)
    elif task == "translation":
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        result = eval_runner.evaluate_translation(model, tokenizer)
    else:
        raise ValueError("Unsupported task: " + task)

    print("\n=== Evaluation Result ===")
    for key, val in result.items():
        print(f"{key}: {val:.4f}")


if __name__ == "__main__":
    main()
