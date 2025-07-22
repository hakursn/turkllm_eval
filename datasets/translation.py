from datasets import load_dataset

def load_translation_dataset(name="wmt14", split="test", lang_pair=("en", "tr")):
    ds = load_dataset(name, f"{lang_pair[0]}-{lang_pair[1]}", split=split)
    return [
        {
            "source": example["translation"][lang_pair[0]],
            "target": example["translation"][lang_pair[1]],
        }
        for example in ds
    ]
