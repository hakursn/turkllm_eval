from datasets import load_dataset

def load_qa_dataset(name="turquad", split="validation"):
    ds = load_dataset(name, split=split)
    return [
        {
            "context": example["context"],
            "question": example["question"],
            "answers": example["answers"]["text"]
        }
        for example in ds
    ]
