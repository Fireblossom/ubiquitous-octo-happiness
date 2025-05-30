# Us vs. Them

## Repository Structure

- **Notebooks (-en for English prompting)**:
  - `gemma3.ipynb` and `gemma3-en.ipynb`: Notebooks for working with Gemma 3 27B model
  - `qwen3-small.ipynb` and `qwen3-small-en.ipynb`: Notebooks for Qwen 3 32B model
  - `qwen3-large.ipynb` and `qwen3-large-en.ipynb`: Notebooks for Qwen 3 235B-A22B model

- **Tools**:
  - `merge.py`: Script for merging model predictions into a unified format
  - `sample.csv`: Sample social media texts for analysis

- **Directories**:
  - `results/`: Contains classification outputs from different models
  - `agreements/`: Contains annotation agreements data for model evaluation
  - `evaluation/`: Contains evaluation metrics and analysis scripts

## Models Used

1. **Gemma 3**
   - `google/gemma-3-27b-it`: 27B parameter instruction-tuned model

2. **Qwen 3**
   - Small model variant: `Qwen/Qwen3-32B`: 32B parameter model
   - Large model variant: `Qwen/Qwen3-235B-A22B`: 235B parameter model with A22B architecture

## Usage

1. Install required dependencies:
   ```bash
   pip install vllm jupyter
   ```

2. Open the relevant notebooks for your desired model and prompting language (English or Chinese)

3. Follow the notebook instructions to generate predictions for social media texts

4. Merge predictions from different models:
   ```bash
   python merge.py
   ```

5. Evaluate model performance:
   ```bash
   # For English evaluation
   python evaluation/evaluate_en.py
   # For Chinese evaluation
   python evaluation/evaluate_zh.py
   ```

## Taxonomy

The repository implements a Recognition Logic (RL) framework for analyzing social network texts, focusing on:
- Vernacular Spatial Authority (RL1): Local perceptions of spatial categories
- Administrative Legitimacy (RL2): Official jurisdiction and legal status
- Family Rootedness (RL3): Generational depth of family settlement
- Linguistic-Cultural Recognition (RL4): Dialect, accent, and cultural habits
- Functional Livability (RL5): Material infrastructure and environmental quality
- Social Embeddedness (RL6): Integration into local social circles and assets
- Occupational Typification (RL7): Association with dominant professional groups

## Requirements

- vLLM
- Jupyter