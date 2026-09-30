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

## Raw Data Overview

| File | Data |
|---|---|
| [`bank_code_mapping.csv`](data/raw/bank_code_mapping.csv) | Mapping between bank codes and bank names. |
| [`dy_connectedness_from_daily.csv`](data/raw/dy_connectedness_from_daily.csv) | Daily directional connectedness received by each institution from the rest of the system. |
| [`dy_connectedness_net_daily.csv`](data/raw/dy_connectedness_net_daily.csv) | Daily net directional connectedness for each institution. |
| [`dy_connectedness_to_daily.csv`](data/raw/dy_connectedness_to_daily.csv) | Daily directional connectedness transmitted by each institution to the rest of the system. |
| [`institution_covar_daily.csv`](data/raw/institution_covar_daily.csv) | Daily institution-level CoVaR estimates. |
| [`institution_market_data_daily.csv`](data/raw/institution_market_data_daily.csv) | Daily market data for the sampled financial institutions. |
| [`interbank_rate_3m_daily.csv`](data/raw/interbank_rate_3m_daily.csv) | Daily three-month interbank rate. |
| [`macro_yields_daily.csv`](data/raw/macro_yields_daily.csv) | Daily government- and corporate-bond yields at selected maturities. |
| [`market_volatility_daily.csv`](data/raw/market_volatility_daily.csv) | Daily market close, log return, and volatility series. |
| [`sentiment_labels_public.csv`](data/raw/sentiment_labels_public.csv) | Public forum-text sample and its sentiment labels. |

For the forum data, both the comment count and the view count for each post are measured on the post's publication date.

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

Remove `--dry-run` after configuring the fine-tuned model and the source forum-post files. Raw forum text, the complete post-level Eastmoney records, and model weights are not included. The complete Eastmoney records are not publicly redistributed because the source material and platform fields are subject to copyright and platform-use restrictions; privacy and storage constraints also apply to the text and model artifacts.

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

The active configurations use **scalar direct targets** at `lead=1` and `lead=5`, with `data=direct` and `pred_len=1` in both cases. For origin session \(t\), the five-day job learns only \(SR_{t+5}\); it does not learn a joint vector of days 1–5 or average those days. Encoder and decoder-prefix observations end at \(t\). Only the known target-date calendar features are supplied for the future step; its observed predictor values are not supplied. Each exported row records `origin_date`, `target_date`, `lead`/`horizon`, `forecast_mode`, actual and predicted values.

`data/intermediate/trading_calendar.csv` contains the 3,971 dated market rows from `data/raw/institution_market_data_daily.csv` (2007-12-27 through 2024-04-30), sorted and deduplicated. Leads count Chinese trading sessions, rather than calendar days or rows remaining after a missing-data filter. A sample is omitted when its look-back is not consecutive in this calendar or its exact terminal target is unavailable. Direct Informer splits are determined by target-row boundaries (70% training, 10% validation, 20% test); scalers fit only the training rows. Validation origins must be at or after the final training row, and test origins must be at or after the final validation row; earlier cross-boundary origins are omitted. The existing look-back lengths and architecture/optimizer settings are retained.

Direct outputs and checkpoints use separate paths and lead-specific setting names. The legacy `custom` datasets remain available for historical replication, but no active one-day or five-day configuration selects them. Direct unseen-future `--do_predict` is rejected rather than falling back to the legacy next-row prediction dataset.

### 7. Run comparison models

```powershell
python code/07_run_model_comparison.py --config configs/07_model_comparison.json --job list
python code/07_run_model_comparison.py --config configs/07_model_comparison.json --job one_step --dry-run
```

The comparison design includes persistence, AR(1), exponential smoothing, gradient boosting regression, and PatchTST. GBR and the neural comparator heads learn one terminal target at the requested lead. AR(1) iterates its fitted one-session recurrence to that lead; persistence and ETS produce their terminal forecasts from observations available at the origin. The current PatchTST implementation uses one multivariate input window containing aggregate SR history and all 16 configured non-target predictors. Each channel is patch-encoded with shared Transformer weights, and the scalar prediction head uses all 17 channels to forecast aggregate SR.

Comparators use the configured 70% raw chronological training cutoff, fit scalers and deterministic-model parameters only before that cutoff, and exclude training labels crossing it. Their evaluation predictions are restricted to the exact origin/target/lead keys supplied by the direct Informer test archive. Input archives must declare `forecast_mode=direct` (or `direct_terminal_target`) and contain explicit origin and lead metadata; the calendar and actual values are checked against the forecasting inputs. Copy a newly generated direct Informer `test_results.csv` to `data/intermediate/informer_direct_1d.csv` or `informer_direct_5d.csv` as appropriate. Do not relabel an old joint-output archive as direct. The active comparison and Stage 8 evaluation outputs use `results/direct/`.

This source correction does not run training, regenerate the archived paper results, or change their provenance. Files under `artifacts/` still reproduce the historical submitted tables and figures. The original DM implementation and its documented effective-horizon-one convention are unchanged by the forecast-target correction; updating that evaluation convention is a separate methodological decision.

Run the direct-target contract checks in an environment with the repository dependencies installed:

```powershell
python -m unittest discover -s tests -p "test_direct_*.py" -v
```

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

## Reproducibility Notes

- Random seeds and chronological data splits are explicitly specified in the relevant configuration files.
- Even with fixed random seeds, deep-learning results may vary slightly because of differences in GPU hardware, CUDA, cuDNN, and PyTorch versions.
