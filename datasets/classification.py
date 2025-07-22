from datasets import load_dataset

def load_classification_dataset(name="tekir", split="test"):
    ds = load_dataset(name, split=split)
    return [
        {
            "text": example["text"],
            "label": example["label"]
        }
        for example in ds
    ]
