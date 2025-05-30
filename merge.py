import pandas as pd
from pathlib import Path
import re


sample = pd.read_csv('sample.csv')

for file in Path('.').glob("results/*.csv"):
    if not file.stem.startswith('sample'):
        df = pd.read_csv(file)
        model = file.stem
        sample[model] = df['RL_Types']

sample.drop(columns=['Anotator F'], inplace=True)
sample.rename(columns={'Anotator T': 'Golden'}, inplace=True)

# Define the cleaning function
def clean_RL_output(text):
    # pd is assumed to be imported as pd in a previous cell and available in the global scope
    if pd.isna(text):
        return text
    if not isinstance(text, str):
        # If not a string (e.g., a number accidentally in the column), return as is.
        # This assumes LLM outputs are primarily strings or NaN.
        return text

    # Preserve placeholder for "no RLs" if it's simply "-"
    if text.strip() == "-":
        return "-"

    # Find all occurrences of RL followed by digits (e.g., "RL1", "RL23")
    RL_patterns = re.findall(r'RL\d+', text)
    
    # Remove duplicates while preserving order of first appearance
    unique_RLs = []
    seen = set() # Use a set for efficient "in" check for uniqueness
    for RL in RL_patterns:
        if RL not in seen:
            unique_RLs.append(RL)
            seen.add(RL)

    # Join the unique RLs with ", "
    # If no RLs were found (and text wasn't "-"), this will return an empty string.
    return ", ".join(unique_RLs)


def clean_zh_output(text):
    # pd is assumed to be imported as pd in a previous cell and available in the global scope
    if pd.isna(text):
        return text
    if not isinstance(text, str):
        # If not a string (e.g., a number accidentally in the column), return as is.
        # This assumes LLM outputs are primarily strings or NaN.
        return text

    # Preserve placeholder for "no RLs" if it's simply "-"
    if text.strip() == "-":
        return "-"

    # Find all occurrences of RL followed by digits (e.g., "RL1", "RL23")
    RL_patterns = re.findall(r'认同逻辑\d+', text)
    
    # Remove duplicates while preserving order of first appearance
    unique_RLs = []
    seen = set() # Use a set for efficient "in" check for uniqueness
    for RL in RL_patterns:
        if RL not in seen:
            unique_RLs.append(RL)
            seen.add(RL)

    # Join the unique RLs with ", "
    # If no RLs were found (and text wasn't "-"), this will return an empty string.
    return ", ".join(unique_RLs)

# Identify model output columns to be cleaned.
# These are columns that were added from the CSV files in the 'results' directory.
# We exclude known non-model columns that are part of the original 'sample.csv' 
# or derived columns like 'Golden'.
# Based on cell 1, 'Unnamed: 0' (if present), 'comment_id', 'comment_type', 
# 'text', and 'Golden' are not model output columns.
known_non_model_cols = ['Unnamed: 0', 'comment_id', 'comment_type', 'text', 'Golden']
model_cols_to_clean = [col for col in sample.columns if col not in known_non_model_cols]

# Apply the cleaning function to each identified model output column
for col_name in model_cols_to_clean:
    # Ensure the column actually exists in the DataFrame before trying to apply the function
    if col_name in sample.columns:
        if col_name.endswith('en'):
            sample[col_name] = sample[col_name].apply(clean_RL_output)
        else:
            sample[col_name] = sample[col_name].apply(clean_zh_output)

# The DataFrame 'sample' is now modified in place with cleaned LLM outputs.
# You can inspect the changes, for example, by printing sample.head() in a new cell:
# sample.head()
sample.to_csv('evaluation/sample_all.csv', index=False)