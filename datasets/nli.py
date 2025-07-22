from datasets import load_dataset

def load_nli_dataset(split="validation"):
    ds = load_dataset("xnli", split=split)
    return [
        {
            "premise": example["premise"],
            "hypothesis": example["hypothesis"],
            "label": example["label"]
        }
        for example in ds if example["language"] == "turkish"
    ]
