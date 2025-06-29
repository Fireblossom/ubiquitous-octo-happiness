# LLM Analysis Notebooks

This directory contains Jupyter notebooks for running large language models (LLMs) to analyze social media texts using the Recognition Logic (RL) framework for "local person" identity definition standards.

## Notebook Structure

Each model has two versions - one for Chinese prompting and one for English prompting:

- **Gemma 3 (27B)**: `gemma3.ipynb` (Chinese prompting) | `gemma3-en.ipynb` (English prompting)
- **Qwen 3 Small (32B)**: `qwen3-small.ipynb` (Chinese prompting) | `qwen3-small-en.ipynb` (English prompting)  
- **Qwen 3 Large (235B)**: `qwen3-large.ipynb` (Chinese prompting) | `qwen3-large-en.ipynb` (English prompting)
- **GPT4.1**: `gpt41-api.py` (Chinese prompting) | `gpt41-api-en.py` (English prompting)
- **Deepseek**: `deepseek-api.py` (Chinese prompting) | `deepseek-api-en.py` (English prompting)

## Usage

### Prerequisites

1. Install required dependencies:
   `vllm jupyter pandas transformers torch openai`

2. Ensure sufficient GPU memory:
   - Gemma 3 (27B): ~54GB VRAM
   - Qwen 3 Small (32B): ~64GB VRAM  
   - Qwen 3 Large (235B): ~470GB VRAM

### Running the Notebooks

1. Choose your model and language (Chinese or English)
2. The notebook will load the model and process `../0_data_collection/dataset.csv`
3. Results are saved to `./llm_outputs/` directory
4. Output format: `{model_name}_{method}_{language}.csv`

### Output Format

Each CSV contains:
- `Original_Input_Text`: Input social media text
- `RL_Types`: Predicted RL categories (comma-separated)
- `Raw_Model_Output`: Full model response for debugging