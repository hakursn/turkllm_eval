from datasets import load_dataset

def load_sts_dataset(split='test'):
    ds = load_dataset("figenfikri/stsb_tr", split=split)
    clean_data = []

    for example in ds:
        try:
            # Fix: Handle Turkish comma decimal format ("4,0" → "4.0")
            raw_score = str(example["score"]).replace(",", ".")
            score = float(raw_score) / 5.0  # normalize
        except Exception as e:
            continue  # skip problematic row

        clean_data.append({
            "sentence1": example["sentence1"],
            "sentence2": example["sentence2"],
            "score": score
        })

    return clean_data
