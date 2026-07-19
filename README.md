# LLM-Extracted Sentiment for Systemic-Risk Forecasting

This repository contains the research code, processed data, archived predictions, and table/figure generators for a bank-level systemic-risk forecasting study. The workflow covers sentiment-label preparation, LoRA fine-tuning, batch sentiment inference, daily sentiment-indicator construction, forecasting-data integration, Informer estimation, benchmark comparison, forecast evaluation, and Diebold-Mariano testing.

The forecasting target is the transformed Delta-CoVaR-based systemic-risk proxy

```text
SR = -100 * Delta-CoVaR
```

where a larger value indicates higher bank-level systemic risk. Forecasts are produced at horizons of one and five trading days using information available at the forecast origin.

## Repository Structure

| Path | Purpose |
|---|---|
| `code/` | Eight executable entry points for the end-to-end empirical workflow. |
| `configs/` | JSON configuration files used by the corresponding entry points. |
| `src/` | Reusable data-processing, forecasting, evaluation, and utility modules. |
| `libs/` | Locally archived model implementations and training support modules. |
| `data/` | Public labeled data, raw market and macro-financial inputs, and intermediate daily datasets. |
| `results/` | Archived predictions, error metrics, DM-test results, and inputs used by the reported tables. |
| `artifacts/` | Independent generators, data, and outputs for reproducible tables and figures. |
| `outputs/` | Default destination for newly generated pipeline outputs. |

## Environment

Python 3.10 is recommended. For data processing, evaluation, and table/figure reproduction:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements_core.txt
```

GPU training and inference require a CUDA-compatible PyTorch installation. Install PyTorch first, followed by the full training dependencies:

```powershell
python -m pip install -r requirements_torch_cuda124.txt
python -m pip install -r requirements_full_training.txt
```

The CUDA requirement file targets CUDA 12.4. Select the PyTorch build appropriate for the local CUDA runtime when using a different system configuration.

## End-to-End Workflow

Run commands from the repository root. All supplied configuration files use relative paths, and each stage is designed to consume the preceding stage's output.

### 1. Split the labeled sentiment sample

```powershell
python code/01_split_labeled_data.py --config configs/01_label_split.json
```

This stage performs a stratified train/validation split and creates the CSV and instruction-format JSON files required for supervised fine-tuning.

### 2. Fine-tune the sentiment classifier with LoRA

Inspect the resolved training configuration without starting the computationally intensive run:

```powershell
python code/02_train_sentiment_lora.py --config configs/02_lora_sft.json --dry-run
```

Remove `--dry-run` to start GPU training.

### 3. Run batch sentiment inference

```powershell
python code/03_predict_sentiment.py --config configs/03_inference.json --dry-run
```

Remove `--dry-run` after configuring the fine-tuned model and the source forum-post files. Raw forum text and model weights are not included because of platform-use, privacy, and storage constraints.

### 4. Merge predictions and construct daily sentiment indicators

```powershell
python code/04_merge_predictions_and_build_daily_sentiment.py --config configs/04_daily_sentiment.json --step all
```

This stage audits and merges post-level predictions, repairs invalid labels, and constructs the four institution-day indicators used in the empirical analysis: sentiment direction, attention-based intensity, exposure-based intensity, and engagement-based intensity.

The archived `data/intermediate/daily_sentiment_indicators.csv` allows reproduction to begin at Stage 5 without rerunning restricted text inference.

### 5. Build bank-level and aggregate forecasting datasets

```powershell
python code/05_build_forecasting_datasets.py --config configs/05_forecast_data.json
```

This stage combines the SR proxy, DY connectedness measures, bank characteristics, macro-financial predictors, and sentiment indicators. It produces institution-level and aggregate datasets with and without sentiment variables. Missing values are not filled by forward or backward propagation.

### 6. Run Informer experiments

Single experiment:

```powershell
python code/06_run_informer.py --config configs/06_informer_single.json --dry-run
```

Batch experiment discovery and execution:

```powershell
python code/06_batch_train_informer.py --config configs/06_informer_batch.json --job list
python code/06_batch_train_informer.py --config configs/06_informer_batch.json --job aggregate_1d_with_sentiment --dry-run
```

Remove `--dry-run`, or use the batch runner's explicit execution option, only when the GPU environment and output locations have been reviewed.

### 7. Run comparison models

```powershell
python code/07_run_model_comparison.py --config configs/07_model_comparison.json --job list
python code/07_run_model_comparison.py --config configs/07_model_comparison.json --job one_step --dry-run
```

The comparison design includes persistence, AR(1), exponential smoothing, gradient boosting regression, and PatchTST. Their predictions are aligned with the archived Informer predictions by forecast origin, target date, and horizon.

### 8. Evaluate predictive accuracy

```powershell
python code/08_analyze_model_comparison.py --config configs/08_dm_analysis.json --job all
```

This stage calculates MSE, RMSE, MAE, and R-squared and performs two-sided Diebold-Mariano tests under absolute-error and squared-error loss. The loss differential is defined as Informer loss minus comparator loss, so a negative statistic favors Informer.

## Reproducing Tables and Figures

Every generated artifact has an independent directory containing its script, required input data, and output. Reproduce all lightweight artifacts with:

```powershell
python artifacts/run_all.py
```

The included generators cover Tables 3-5, Tables A2-A4, Tables A6-A7, Figure 2, and Figures A1-A2. The manuscript reports only Informer, persistence, AR(1), exponential smoothing, gradient boosting regression, and PatchTST in the principal model-comparison artifacts, while the archived result files retain the complete experiment record.

## Archived Results

The `results/` directory includes:

- one- and five-day target-date prediction files;
- model-comparison accuracy metrics;
- Diebold-Mariano test outputs;
- classifier-evaluation results;
- institution-level early-warning summaries; and
- the empirical datasets used by the reproducible table and figure scripts.

These files permit verification of the reported evaluation and artifacts without rerunning LoRA, deep-learning inference, or Informer training.

## Reproducibility Notes

- Configuration files use repository-relative paths and contain no machine-specific absolute paths.
- Random seeds and chronological data splits are specified in the relevant configurations.
- Standardization parameters are estimated from the training sample and then applied to validation and test samples.
- Forecast predictors are aligned to the forecast origin; sentiment view and comment counts refer to information recorded for the post's publication day.
- Deep-learning results may vary slightly across GPU hardware, CUDA, cuDNN, and PyTorch versions despite fixed seeds.

## Citation and Use

This private repository is intended for editorial review and controlled research replication. Before public release, confirm the redistribution conditions of all included market, macro-financial, and platform-derived data and add the final article citation and an explicit software/data license.
