# Evaluation Results

This directory contains the evaluation results for LLM predictions on the Recognition Logic (RL) classification task, for both Chinese and English prompts.

## Overview

- Models are assessed using multi-label F1 scores (micro, macro, weighted, samples) and detailed classification reports for each RL category.
- Both Chinese and English model outputs are compared against human-annotated golden standards.
- Key Results are listed in the paper's Table 1-4

## Key Results Summary (Micro/Macro F1)

| Model        | Prompt Language | Zero-shot               | Few-shot (k=7)      | Standard Prompting (Non-CoT) |
|--------------|-----------------|-------------------------|---------------------|------------------------------|
| gemma-3-27b-it | Chinese         | 0.6688 / 0.5682         | 0.7813 / 0.6804     | 0.7321 / 0.6407              |
|              | English         | 0.7107 / 0.6218         | 0.7384 / 0.6529     | 0.6726 / 0.5589              |
| Qwen3-235B-A22B | Chinese         | 0.6993 / 0.6424         | 0.7901 / 0.7164     | 0.7224 / 0.6227              |
|              | English         | 0.7155 / 0.6147         | 0.7567 / 0.6694     | 0.6827 / 0.5870              |
| Qwen3-32B    | Chinese         | 0.7355 / 0.6349         | **0.7915 / 0.7175** | 0.7363 / 0.6081              |
|              | English         | 0.7273 / 0.6256         | 0.7623 / **0.6720** | 0.6216 / 0.5093              |
| GPT-4.1      | Chinese         | **0.7443** / **0.6574** | 0.7566 / 0.6827     | **0.7544** / **0.6707**      |
|              | English         | 0.7130 / 0.6365         | **0.7648** / 0.6600 | 0.7323 / 0.6131              |
| Deepseek-R1-0528  | Chinese         | 0.7286 / **0.6640**     | 0.7295 / 0.6436     | 0.7325 / 0.6659              |
|              | English         | 0.6954 / 0.6310         | 0.7240 / 0.6631     | 0.6872 / 0.6202              |

## How to Reproduce

- See `evaluate_zh.ipynb` and `evaluate_en.ipynb` for full evaluation code and detailed results.
- All metrics are computed using scikit-learn's multi-label classification utilities.

## Notes
- For more details, refer to the full notebooks in this directory.