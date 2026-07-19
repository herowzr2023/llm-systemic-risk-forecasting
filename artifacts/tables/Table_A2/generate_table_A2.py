from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
MODEL_NAMES = {
    "fin_chinese_sen_llama3": "Llama-3-Chinese-8B-Instruct-v3-emotion",
    "hfl_llama-3-chinese-8b-instruct-v2": "Llama-3-Chinese-8B-Instruct-v2",
    "hfl_llama-3-chinese-8b-instruct-v3": "Llama-3-Chinese-8B-Instruct-v3",
    "hfl_llama-3-chinese-8b-instruct": "Llama-3-Chinese-8B-Instruct",
    "hfl_llama-3-chinese-8b": "Llama-3-Chinese-8B",
    "meta-llama_Meta-Llama-3-8B-Instruct": "Llama-3-8B-Instruct",
    "meta-llama_Meta-Llama-3-8B": "Llama-3-8B",
}
DATASET_NAMES = {"train": "Training set", "val": "Held-out test set"}


def main() -> None:
    source = pd.read_csv(ROOT / "data" / "combined_evaluation.csv")
    required = {"dataset", "model_name", "accuracy", "precision", "recall", "f1"}
    missing = required.difference(source.columns)
    if missing:
        raise ValueError(f"combined_evaluation.csv is missing columns: {sorted(missing)}")
    rows = [["Data set", "Model", "Accuracy", "Precision", "Recall", "F1-score"]]
    for record in source.itertuples(index=False):
        rows.append([
            DATASET_NAMES.get(record.dataset, record.dataset),
            MODEL_NAMES.get(record.model_name, record.model_name),
            f"{record.accuracy:.3f}",
            f"{record.precision:.3f}",
            f"{record.recall:.3f}",
            f"{record.f1:.3f}",
        ])
    output = ROOT / "output" / "Table_A2.csv"
    output.parent.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False, header=False, encoding="utf-8-sig")
    print(output)


if __name__ == "__main__":
    main()
