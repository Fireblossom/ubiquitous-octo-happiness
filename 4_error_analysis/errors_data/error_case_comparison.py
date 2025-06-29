import pandas as pd
import numpy as np
from pathlib import Path

# Load the dataset with golden labels
dataset = pd.read_csv('../0_data_collection/dataset.csv')

# Function to parse RL categories
def parse_rl_categories(rl_string):
    if pd.isna(rl_string) or rl_string == '-' or rl_string == 'N/A':
        return set()
    
    if isinstance(rl_string, str):
        categories = rl_string.replace(' ', '').split(',')
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
    return set(map(int, list(str(int(golden_string)))))

def load_model_predictions(model_file):
    """Load model predictions and merge with dataset"""
    model_output = pd.read_csv(f'../2_run_llms/llm_outputs/{model_file}')
    merged = pd.merge(dataset, model_output, left_on='text', right_on='Original_Input_Text', how='inner')
    merged['golden_parsed'] = merged['Golden'].apply(parse_golden)
    merged['predicted_parsed'] = merged['RL_Types'].apply(parse_rl_categories)
    merged['is_error'] = merged['golden_parsed'] != merged['predicted_parsed']
    return merged

# Load all models
models = {
    'Qwen3-32B (Few-shot)': 'Qwen3-32B_few_shot.csv',
    'Qwen3-235B-A22B (Few-shot)': 'Qwen3-235B-A22B_few_shot.csv',
    'Gemma-3-27B (Few-shot)': 'gemma-3-27b-it_few_shot.csv',
    'Qwen3-32B (Zero-shot)': 'Qwen3-32B_zero_shot.csv',
    'Qwen3-235B-A22B (Zero-shot)': 'Qwen3-235B-A22B_zero_shot.csv',
    'Gemma-3-27B (Zero-shot)': 'gemma-3-27b-it_zero_shot.csv'
}

# Load all model predictions
all_predictions = {}
for model_name, model_file in models.items():
    try:
        all_predictions[model_name] = load_model_predictions(model_file)
        print(f"Loaded {model_name}")
    except Exception as e:
        print(f"Error loading {model_name}: {e}")

# Function to format RL set for display
def format_rl_set(rl_set):
    if not rl_set:
        return "-"
    return ", ".join([f"RL{r}" for r in sorted(rl_set)])

# Function to get RL descriptions
def get_rl_description(rl_num):
    descriptions = {
        1: "Vernacular Spatial Authority",
        2: "Administrative Legitimacy", 
        3: "Family Rootedness",
        4: "Linguistic-Cultural Recognition",
        5: "Functional Livability",
        6: "Social Embeddedness",
        7: "Occupational Typification"
    }
    return descriptions.get(rl_num, f"RL{rl_num}")

# Select interesting error cases
def select_error_cases():
    """Select interesting error cases for comparison"""
    
    # Use Qwen3-32B few-shot as base to find interesting cases
    base_model = all_predictions['Qwen3-32B (Few-shot)']
    
    # Case 1: Complex case with multiple RLs (RL1, RL4, RL7)
    case1 = base_model[base_model['Golden'] == '147'].iloc[0]
    
    # Case 2: Simple case with RL1 only
    case2 = base_model[base_model['Golden'] == '1'].iloc[0]
    
    # Case 3: Case with RL5 (functional livability)
    case3 = base_model[base_model['Golden'] == '5'].iloc[0]
    
    # Case 4: Case with RL7 (occupational typification)
    case4 = base_model[base_model['Golden'] == '7'].iloc[0]
    
    # Case 5: Complex case with RL1, RL3, RL6
    case5 = base_model[base_model['Golden'] == '136'].iloc[0]
    
    return [case1, case2, case3, case4, case5]

# Get error cases
error_cases = select_error_cases()

print("="*100)
print("ERROR CASE COMPARISON ACROSS MODELS")
print("="*100)

for i, case in enumerate(error_cases, 1):
    print(f"\n{'='*80}")
    print(f"CASE {i}")
    print(f"{'='*80}")
    
    # Display text (truncated)
    text = case['text']
    if len(text) > 200:
        text = text[:200] + "..."
    print(f"Text: {text}")
    
    # Display golden labels
    golden_rls = case['golden_parsed']
    print(f"\nGolden Labels: {format_rl_set(golden_rls)}")
    for rl in sorted(golden_rls):
        print(f"  - {get_rl_description(rl)}")
    
    print(f"\nModel Predictions:")
    print("-" * 60)
    
    # Compare predictions across all models
    for model_name in models.keys():
        if model_name in all_predictions:
            # Find the same text in this model
            model_data = all_predictions[model_name]
            same_text = model_data[model_data['text'] == case['text']]
            
            if len(same_text) > 0:
                pred_rls = same_text.iloc[0]['predicted_parsed']
                is_error = same_text.iloc[0]['is_error']
                
                # Determine error type
                error_type = ""
                if is_error:
                    if golden_rls - pred_rls:
                        error_type += "Missed "
                    if pred_rls - golden_rls:
                        error_type += "Spurious"
                    if not error_type:
                        error_type = "Both"
                else:
                    error_type = "Correct"
                
                print(f"{model_name:30} | {format_rl_set(pred_rls):15} | {error_type}")
                
                # Show detailed RL descriptions for predictions
                for rl in sorted(pred_rls):
                    print(f"  {'':30} |   - {get_rl_description(rl)}")
            else:
                print(f"{model_name:30} | {'Not found':15} | -")

# Create summary table
print(f"\n{'='*100}")
print("SUMMARY TABLE")
print(f"{'='*100}")

# Create comparison table
summary_data = []
for model_name in models.keys():
    if model_name in all_predictions:
        model_data = all_predictions[model_name]
        total_samples = len(model_data)
        error_count = model_data['is_error'].sum()
        error_rate = error_count / total_samples * 100 if total_samples > 0 else 0
        
        # Calculate error types
        error_cases = model_data[model_data['is_error'] == True]
        missed_count = 0
        spurious_count = 0
        
        for _, row in error_cases.iterrows():
            golden = row['golden_parsed']
            predicted = row['predicted_parsed']
            if golden - predicted:  # missed
                missed_count += 1
            if predicted - golden:  # spurious
                spurious_count += 1
        
        summary_data.append({
            'Model': model_name,
            'Total': total_samples,
            'Errors': error_count,
            'Error Rate (%)': f"{error_rate:.1f}",
            'Missed': missed_count,
            'Spurious': spurious_count
        })

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

# Save detailed comparison
print(f"\nDetailed comparison saved to: error_case_comparison.csv")

# Create detailed comparison CSV
detailed_data = []
for i, case in enumerate(error_cases, 1):
    for model_name in models.keys():
        if model_name in all_predictions:
            model_data = all_predictions[model_name]
            same_text = model_data[model_data['text'] == case['text']]
            
            if len(same_text) > 0:
                row = same_text.iloc[0]
                detailed_data.append({
                    'Case': i,
                    'Model': model_name,
                    'Text': case['text'][:100] + "..." if len(case['text']) > 100 else case['text'],
                    'Golden': format_rl_set(case['golden_parsed']),
                    'Predicted': format_rl_set(row['predicted_parsed']),
                    'Is_Error': row['is_error'],
                    'Error_Type': 'Correct' if not row['is_error'] else 'Error'
                })

detailed_df = pd.DataFrame(detailed_data)
detailed_df.to_csv('error_case_comparison.csv', index=False) 