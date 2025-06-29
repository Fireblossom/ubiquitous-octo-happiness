import pandas as pd
import numpy as np
from pathlib import Path
import re

# Load the dataset with golden labels
dataset = pd.read_csv('../0_data_collection/dataset.csv')

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

def analyze_model_errors(model_output_file, model_name):
    """Analyze errors for a specific model"""
    print(f"\n{'='*50}")
    print(f"Analyzing {model_name}")
    print(f"{'='*50}")
    
    # Load the model predictions
    model_output = pd.read_csv(f'../2_run_llms/llm_outputs/{model_output_file}')
    
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
    
    # Calculate statistics
    total_samples = len(merged)
    error_count = len(error_cases)
    error_rate = error_count / total_samples * 100 if total_samples > 0 else 0
    
    # Error type distribution
    error_distribution = error_cases['error_type'].value_counts()
    
    print(f"Total samples: {total_samples}")
    print(f"Error cases: {error_count}")
    print(f"Error rate: {error_rate:.2f}%")
    print(f"\nError type distribution:")
    for error_type, count in error_distribution.items():
        percentage = count / error_count * 100 if error_count > 0 else 0
        print(f"  {error_type}: {count} ({percentage:.1f}%)")
    
    # Save error cases
    output_file = f'{model_name.replace("/", "_").replace("-", "_")}_errors.csv'
    error_cases[['text', 'Golden', 'RL_Types', 'golden_parsed', 'predicted_parsed', 'error_type']].to_csv(output_file, index=False)
    print(f"\nError cases saved to: {output_file}")
    
    return {
        'model_name': model_name,
        'total_samples': total_samples,
        'error_count': error_count,
        'error_rate': error_rate,
        'error_distribution': error_distribution.to_dict(),
        'output_file': output_file
    }

# Define models to analyze
models_to_analyze = [
    ('Qwen3-32B_few_shot.csv', 'Qwen3-32B (Few-shot)'),
    ('Qwen3-32B_zero_shot.csv', 'Qwen3-32B (Zero-shot)'),
    ('Qwen3-32B_no_cot.csv', 'Qwen3-32B (No CoT)'),
    ('gemma-3-27b-it_few_shot.csv', 'Gemma-3-27B (Few-shot)'),
    ('gemma-3-27b-it_zero_shot.csv', 'Gemma-3-27B (Zero-shot)'),
    ('gemma-3-27b-it_no_cot.csv', 'Gemma-3-27B (No CoT)'),
    ('Qwen3-235B-A22B_few_shot.csv', 'Qwen3-235B-A22B (Few-shot)'),
    ('Qwen3-235B-A22B_zero_shot.csv', 'Qwen3-235B-A22B (Zero-shot)'),
    ('Qwen3-235B-A22B_no_cot.csv', 'Qwen3-235B-A22B (No CoT)')
]

# Analyze all models
results = []
for model_file, model_name in models_to_analyze:
    try:
        result = analyze_model_errors(model_file, model_name)
        results.append(result)
    except Exception as e:
        print(f"Error analyzing {model_name}: {e}")

# Create comparison summary
print(f"\n{'='*80}")
print("MODEL COMPARISON SUMMARY")
print(f"{'='*80}")

# Create comparison table
comparison_data = []
for result in results:
    comparison_data.append({
        'Model': result['model_name'],
        'Total Samples': result['total_samples'],
        'Error Count': result['error_count'],
        'Error Rate (%)': f"{result['error_rate']:.2f}",
        'Spurious (%)': f"{result['error_distribution'].get('spurious', 0) / result['error_count'] * 100:.1f}" if result['error_count'] > 0 else "0.0",
        'Missed (%)': f"{result['error_distribution'].get('missed', 0) / result['error_count'] * 100:.1f}" if result['error_count'] > 0 else "0.0",
        'Both (%)': f"{result['error_distribution'].get('both', 0) / result['error_count'] * 100:.1f}" if result['error_count'] > 0 else "0.0"
    })

comparison_df = pd.DataFrame(comparison_data)
print(comparison_df.to_string(index=False))

# Save comparison to CSV
comparison_df.to_csv('model_comparison_summary.csv', index=False)
print(f"\nComparison summary saved to: model_comparison_summary.csv")

# Find best and worst performing models
if results:
    best_model = min(results, key=lambda x: x['error_rate'])
    worst_model = max(results, key=lambda x: x['error_rate'])
    
    print(f"\n{'='*50}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*50}")
    print(f"Best performing model: {best_model['model_name']} (Error rate: {best_model['error_rate']:.2f}%)")
    print(f"Worst performing model: {worst_model['model_name']} (Error rate: {worst_model['error_rate']:.2f}%)")
    
    # Calculate average error rate
    avg_error_rate = sum(r['error_rate'] for r in results) / len(results)
    print(f"Average error rate across all models: {avg_error_rate:.2f}%")

print(f"\nAnalysis complete!") 