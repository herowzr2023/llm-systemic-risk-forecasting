"""Train the BERT-family comparison models used as sentiment baselines.

This script uses the Transformers API in-process, so it can be debugged without
starting a child Python process. Full training is optional and GPU-intensive.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
LABELS = {
    -1: 0, 0: 1, 1: 2,
    "-1": 0, "0": 1, "1": 2,
    "负面": 0, "中性": 1, "正面": 2,
    "negative": 0, "neutral": 1, "positive": 2,
}


def metrics(prediction) -> dict[str, float]:
    predicted = np.argmax(prediction.predictions, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        prediction.label_ids, predicted, average="weighted", zero_division=0
    )
    return {"accuracy": accuracy_score(prediction.label_ids, predicted),
            "precision": precision, "recall": recall, "f1": f1}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "bert_models.json"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.dry_run:
        print(json.dumps(config, ensure_ascii=False, indent=2))
        return

    from datasets import Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              Trainer, TrainingArguments, set_seed)

    set_seed(config["random_seed"])
    source = (ROOT / config["input_csv"]).resolve()
    frame = pd.read_csv(source)
    frame["label"] = frame[config["label_column"]].map(LABELS)
    if frame["label"].isna().any():
        raise ValueError("Unrecognized sentiment labels")
    train, test = train_test_split(frame, test_size=.1, random_state=config["random_seed"],
                                   stratify=frame["label"])
    records = []
    for model_name in config["models"]:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
        def tokenize(batch):
            return tokenizer(batch[config["text_column"]], truncation=True, max_length=256)
        train_set = Dataset.from_pandas(train[[config["text_column"], "label"]], preserve_index=False).map(tokenize, batched=True)
        test_set = Dataset.from_pandas(test[[config["text_column"], "label"]], preserve_index=False).map(tokenize, batched=True)
        slug = model_name.replace("/", "_")
        training_args = TrainingArguments(
            output_dir=str(ROOT / config["output_dir"] / slug), num_train_epochs=config["epochs"],
            per_device_train_batch_size=config["batch_size"],
            per_device_eval_batch_size=config["batch_size"], learning_rate=config["learning_rate"],
            eval_strategy="epoch", save_strategy="epoch", report_to=[], seed=config["random_seed"])
        trainer = Trainer(model=model, args=training_args, train_dataset=train_set,
                          eval_dataset=test_set, processing_class=tokenizer, compute_metrics=metrics)
        trainer.train()
        for sample, dataset in (("train", train_set), ("test", test_set)):
            result = trainer.evaluate(dataset)
            records.append({"dataset": sample, "model_name": slug,
                            "accuracy": result["eval_accuracy"], "precision": result["eval_precision"],
                            "recall": result["eval_recall"], "f1": result["eval_f1"]})
    pd.DataFrame(records).to_csv(ROOT / "bert_evaluation_results_tidy.csv", index=False)


if __name__ == "__main__":
    main()
