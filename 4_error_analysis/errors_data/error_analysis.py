import pandas as pd
import numpy as np

# Load the dataset with golden labels
dataset = pd.read_csv('../0_data_collection/dataset.csv')

# Load the model predictions
model_output = pd.read_csv('../2_run_llms/llm_outputs/Qwen3-235B-A22B_few_shot.csv')

# Function to parse RL categories
def parse_rl_categories(rl_string):
    if pd.isna(rl_string) or rl_string == '-' or rl_string == 'N/A':
        return set()
    
    # Handle different formats
    if isinstance(rl_string, str):
        # Remove spaces and split by comma
        categories = rl_string.replace(' ', '').split(',')
        # Extract numbers from "认同逻辑X" format
        rl_numbers = []
        for cat in categories:
            if '认同逻辑' in cat:
                try:
                    num = int(cat.replace('认同逻辑', ''))
                    rl_numbers.append(num)
                except:
                    pass
        return set(rl_numbers)
    return set()

# Function to parse golden labels
def parse_golden(golden_string):
    """Parses the 'Golden' column (e.g., "147" means labels 1, 4, 7)."""
    if pd.isna(golden_string) or str(golden_string).strip() == "-":
        return set()
    # Ensure value is treated as a string of digits
    return set(map(int, list(str(int(golden_string))))) # int(golden_string) handles potential float like 4.0

# Merge datasets
merged = pd.merge(dataset, model_output, left_on='text', right_on='Original_Input_Text', how='inner')

# Parse labels
merged['golden_parsed'] = merged['Golden'].apply(parse_golden)
merged['predicted_parsed'] = merged['RL_Types'].apply(parse_rl_categories)

# Find errors
merged['is_error'] = merged['golden_parsed'] != merged['predicted_parsed']
merged['error_type'] = merged.apply(lambda row: 
    'missed' if row['golden_parsed'] - row['predicted_parsed'] else
    'spurious' if row['predicted_parsed'] - row['golden_parsed'] else
    'both' if (row['golden_parsed'] - row['predicted_parsed']) and (row['predicted_parsed'] - row['golden_parsed']) else
    'none', axis=1)

# Filter error cases
error_cases = merged[merged['is_error'] == True].copy()

print(f"Total samples: {len(merged)}")
print(f"Error cases: {len(error_cases)}")
print(f"Error rate: {len(error_cases)/len(merged)*100:.2f}%")

# Show error types distribution
print("\nError type distribution:")
print(error_cases['error_type'].value_counts())

# Find interesting error cases
print("\n=== TYPICAL ERROR CASES ===")

# Case 1: Missed RL category
missed_cases = error_cases[error_cases['error_type'].isin(['missed', 'both'])]
if len(missed_cases) > 0:
    case1 = missed_cases.iloc[0]
    print(f"\nCase 1 - Missed RL Category:")
    print(f"Text: {case1['text'][:100]}...")
    print(f"Golden: {case1['golden_parsed']}")
    print(f"Predicted: {case1['predicted_parsed']}")
    print(f"Error: Missed {case1['golden_parsed'] - case1['predicted_parsed']}")

# Case 2: Spurious RL category
spurious_cases = error_cases[error_cases['error_type'].isin(['spurious', 'both'])]
if len(spurious_cases) > 0:
    case2 = spurious_cases.iloc[0]
    print(f"\nCase 2 - Spurious RL Category:")
    print(f"Text: {case2['text'][:100]}...")
    print(f"Golden: {case2['golden_parsed']}")
    print(f"Predicted: {case2['predicted_parsed']}")
    print(f"Error: Spurious {case2['predicted_parsed'] - case2['golden_parsed']}")

# Case 3: Complete mismatch
both_cases = error_cases[error_cases['error_type'] == 'both']
if len(both_cases) > 0:
    case3 = both_cases.iloc[0]
    print(f"\nCase 3 - Complete Mismatch:")
    print(f"Text: {case3['text'][:100]}...")
    print(f"Golden: {case3['golden_parsed']}")
    print(f"Predicted: {case3['predicted_parsed']}")
    print(f"Error: Missed {case3['golden_parsed'] - case3['predicted_parsed']}, Spurious {case3['predicted_parsed'] - case3['golden_parsed']}")

# Save error cases to CSV for manual review
error_cases[['text', 'Golden', 'RL_Types', 'golden_parsed', 'predicted_parsed', 'error_type']].to_csv('qwen3_235b_few_shot_errors.csv', index=False)
print(f"\nError cases saved to: qwen3_235b_few_shot_errors.csv") 