# Recognition Logic (RL) Classification: Analyzing "Local Person" Identity in Chinese Social Media

A research project analyzing social media texts to understand "local person" identity definition standards using large language models and the Recognition Logic (RL) framework.

## Project Overview

This project investigates how individuals define and evaluate "local person" identity in Chinese social media contexts. Using large language models (LLMs), we classify social media texts according to seven distinct RL categories that represent different standards people use to establish local identity or evaluate local areas.

## Project Structure

```
ubiquitous-octo-happiness/
├── 0_data_collection/          # Social media dataset and collection info
├── 1_annotation/               # Annotation guidelines and agreement analysis
├── 2_run_llms/                 # LLM analysis notebooks and outputs
├── 3_evaluation_results/       # Model evaluation and performance metrics
├── 4_error_analysis/           # Detailed error analysis and insights
└── README.md                   # This file
```

## Recognition Logic (RL) Framework

The analysis uses a 7-category Recognition Logic (RL) framework for analyzing social network texts. Each RL type represents standards that authors use to define "local person" identity or evaluate "local" areas.

| RL | Category | Definition |
|----|----------|------------|
| RL1 | Vernacular Spatial Authority | Shared, habitual, or historically sedimented local perceptions of intra-city spatial categories |
| RL2 | Administrative Legitimacy | Appeals to official jurisdiction, legal status, or administrative designation |
| RL3 | Family Rootedness | Evaluation based on generational depth of family settlement |
| RL4 | Linguistic-Cultural Recognition | Relies on dialect, accent, or cultural linguistic habits as boundary markers |
| RL5 | Functional Livability | Evaluation of urban areas in terms of material infrastructure |
| RL6 | Social Embeddedness | Judging local status based on social integration and local assets |
| RL7 | Occupational Typification | Framing identity by associating districts with professional groups |

For detailed definitions and examples, see `1_annotation/Identity Annotation Manual.pdf`.

## Workflow

1. **Data Collection** (`0_data_collection/`) - social media texts from 11 Chinese cities
2. **Annotation** (`1_annotation/`) - Manual annotation using RL framework
3. **LLM Analysis** (`2_run_llms/`) - Model predictions using Gemma 3, Qwen 3 Small, Qwen 3 Large
4. **Evaluation** (`3_evaluation_results/`) - Performance metrics and comparison
5. **Error Analysis** (`4_error_analysis/`) - Detailed error patterns and insights

## Models Used

- **Gemma 3 (27B)**: `google/gemma-3-27b-it`
- **Qwen 3 Small (32B)**: `Qwen/Qwen3-32B`
- **Qwen 3 Large (235B)**: `Qwen/Qwen3-235B-A22B`

All models tested with three prompting methods:
- **Few-shot**: Models provided with examples and reasoning steps
- **Zero-shot**: Models given task description without examples
- **No CoT (No Chain-of-Thought)**: Models without reasoning capabilities (enable_thinking=False)

## Key Findings

- **Few-shot significantly outperforms zero-shot**: Error rates reduced by 45-50%
- **RL7 (Occupational Typification)**: Most challenging category with 84.4% missed rate
- **Zero-shot models show complete failure**: 96-97% error rates
- **Model size doesn't guarantee better performance**: Smaller models can outperform larger ones

For detailed findings and analysis, see `4_error_analysis/comprehensive_error_analysis.md`.

## Quick Start

### Running the Analysis
1. See `2_run_llms/README.md` for detailed LLM analysis instructions
2. See `3_evaluation_results/README.md` for evaluation methodology
3. Review error analysis in `4_error_analysis/` for insights

## Directory Details

### `0_data_collection/`
Contains the main dataset (`dataset.csv`) with social media texts from Red Note platform, covering major Chinese cities. See `0_data_collection/README.md` for detailed dataset documentation.

### `1_annotation/`
Annotation guidelines, manual, and agreement analysis. The `Identity Annotation Manual.pdf` contains complete RL framework definitions with examples.

### `2_run_llms/`
Jupyter notebooks for running LLM analysis. Separate notebooks for each model and language (Chinese/English). See `2_run_llms/README.md` for usage instructions.

### `3_evaluation_results/`
Model evaluation notebooks and performance metrics. See `3_evaluation_results/README.md` for evaluation methodology and results.

### `4_error_analysis/`
Comprehensive error analysis including:
- `comprehensive_error_analysis.md`: Complete error analysis and insights
- `rl7_detailed_analysis.md`: Detailed analysis of RL7 challenges

## Research Applications

- Sociolinguistics: Identity construction in social media
- Computational Social Science: Large-scale analysis of identity markers
- Urban Studies: Local identity in Chinese cities