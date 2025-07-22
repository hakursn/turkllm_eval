DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_CLS_MODEL = "lmezenturk/bert-base-turkish-movie-sentiment"
DEFAULT_QA_MODEL = "mrm8488/bert-multi-cased-finetuned-xquadv1"
DEFAULT_TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-en-tr"

EVAL_TASKS = [
    "sts",
    "classification",
    "qa",
    "translation",
    "nli",  # optional, experimental
]