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

# Analyze RL category-specific errors
def analyze_rl_category_errors():
    """Analyze errors for each RL category across all models"""
    
    rl_categories = list(range(1, 8))  # RL1 to RL7
    category_stats = {}
    
    for rl in rl_categories:
        missed_count = 0
        spurious_count = 0
        total_golden = 0
        total_predicted = 0
        
        for model_name, model_data in all_predictions.items():
            for _, row in model_data.iterrows():
                golden = row['golden_parsed']
                predicted = row['predicted_parsed']
                
                # Count missed (golden has RL but predicted doesn't)
                if rl in golden and rl not in predicted:
                    missed_count += 1
                
                # Count spurious (predicted has RL but golden doesn't)
                if rl in predicted and rl not in golden:
                    spurious_count += 1
                
                # Count total occurrences
                if rl in golden:
                    total_golden += 1
                if rl in predicted:
                    total_predicted += 1
        
        category_stats[rl] = {
            'missed': missed_count,
            'spurious': spurious_count,
            'total_golden': total_golden,
            'total_predicted': total_predicted,
            'missed_rate': missed_count / total_golden * 100 if total_golden > 0 else 0,
            'spurious_rate': spurious_count / (len(all_predictions) * len(model_data) - total_golden) * 100 if (len(all_predictions) * len(model_data) - total_golden) > 0 else 0
        }
    
    return category_stats

# Find samples that all models missed or spurious
def find_consensus_errors():
    """Find samples where all models made the same type of error"""
    
    all_missed_samples = []
    all_spurious_samples = []
    
    # Get all unique texts
    all_texts = set()
    for model_data in all_predictions.values():
        all_texts.update(model_data['text'].tolist())
    
    for text in all_texts:
        # Check if this text exists in all models
        text_in_all_models = True
        golden_labels = None
        model_predictions = {}
        
        for model_name, model_data in all_predictions.items():
            same_text = model_data[model_data['text'] == text]
            if len(same_text) == 0:
                text_in_all_models = False
                break
            
            row = same_text.iloc[0]
            if golden_labels is None:
                golden_labels = row['golden_parsed']
            model_predictions[model_name] = row['predicted_parsed']
        
        if not text_in_all_models or golden_labels is None:
            continue
        
        # Check if all models missed the same RLs
        all_missed = True
        all_spurious = True
        
        for model_name, predicted in model_predictions.items():
            # Check for missed RLs
            missed_rls = golden_labels - predicted
            if missed_rls:  # If there are missed RLs
                all_missed = False
                break
        
        # Check for spurious RLs
        for model_name, predicted in model_predictions.items():
            spurious_rls = predicted - golden_labels
            if spurious_rls:  # If there are spurious RLs
                all_spurious = False
                break
        
        if all_missed and golden_labels:  # All models missed all golden RLs
            all_missed_samples.append({
                'text': text,
                'golden': golden_labels,
                'predictions': model_predictions
            })
        
        if all_spurious and any(predicted for predicted in model_predictions.values()):  # All models predicted spurious RLs
            all_spurious_samples.append({
                'text': text,
                'golden': golden_labels,
                'predictions': model_predictions
            })
    
    return all_missed_samples, all_spurious_samples

# Run analysis
print("="*80)
print("RL CATEGORY ERROR ANALYSIS")
print("="*80)

category_stats = analyze_rl_category_errors()

print("\nRL Category Error Statistics:")
print("-" * 80)
print(f"{'RL':<3} {'Description':<30} {'Missed':<8} {'Spurious':<8} {'Missed%':<8} {'Spurious%':<8}")
print("-" * 80)

for rl in sorted(category_stats.keys()):
    stats = category_stats[rl]
    description = get_rl_description(rl)
    print(f"{rl:<3} {description:<30} {stats['missed']:<8} {stats['spurious']:<8} {stats['missed_rate']:<8.1f} {stats['spurious_rate']:<8.1f}")

# Find consensus errors
print(f"\n{'='*80}")
print("FINDING CONSENSUS ERRORS")
print(f"{'='*80}")

all_missed_samples, all_spurious_samples = find_consensus_errors()

print(f"\nSamples where ALL models missed golden labels: {len(all_missed_samples)}")
print(f"Samples where ALL models predicted spurious labels: {len(all_spurious_samples)}")

# Analyze consensus missed samples
if all_missed_samples:
    print(f"\n{'='*80}")
    print("SAMPLES WHERE ALL MODELS MISSED GOLDEN LABELS")
    print(f"{'='*80}")
    
    for i, sample in enumerate(all_missed_samples[:10], 1):  # Show first 10
        print(f"\nCase {i}:")
        print(f"Text: {sample['text'][:150]}...")
        print(f"Golden Labels: {sample['golden']}")
        print("Model Predictions:")
        for model_name, predicted in sample['predictions'].items():
            print(f"  {model_name}: {predicted}")
        print("-" * 60)

# Analyze consensus spurious samples
if all_spurious_samples:
    print(f"\n{'='*80}")
    print("SAMPLES WHERE ALL MODELS PREDICTED SPURIOUS LABELS")
    print(f"{'='*80}")
    
    for i, sample in enumerate(all_spurious_samples[:10], 1):  # Show first 10
        print(f"\nCase {i}:")
        print(f"Text: {sample['text'][:150]}...")
        print(f"Golden Labels: {sample['golden']}")
        print("Model Predictions:")
        for model_name, predicted in sample['predictions'].items():
            print(f"  {model_name}: {predicted}")
        print("-" * 60)

# Save detailed results
print(f"\n{'='*80}")
print("SAVING DETAILED RESULTS")
print(f"{'='*80}")

# Save RL category statistics
rl_stats_data = []
for rl in sorted(category_stats.keys()):
    stats = category_stats[rl]
    rl_stats_data.append({
        'RL': rl,
        'Description': get_rl_description(rl),
        'Missed_Count': stats['missed'],
        'Spurious_Count': stats['spurious'],
        'Total_Golden': stats['total_golden'],
        'Total_Predicted': stats['total_predicted'],
        'Missed_Rate': stats['missed_rate'],
        'Spurious_Rate': stats['spurious_rate']
    })

rl_stats_df = pd.DataFrame(rl_stats_data)
rl_stats_df.to_csv('rl_category_error_stats.csv', index=False)
print("RL category error statistics saved to: rl_category_error_stats.csv")

# Save consensus missed samples
if all_missed_samples:
    missed_data = []
    for i, sample in enumerate(all_missed_samples, 1):
        missed_data.append({
            'Case': i,
            'Text': sample['text'],
            'Golden_Labels': str(sample['golden']),
            'Qwen3-32B_Few': str(sample['predictions']['Qwen3-32B (Few-shot)']),
            'Qwen3-235B_Few': str(sample['predictions']['Qwen3-235B-A22B (Few-shot)']),
            'Gemma_Few': str(sample['predictions']['Gemma-3-27B (Few-shot)']),
            'Qwen3-32B_Zero': str(sample['predictions']['Qwen3-32B (Zero-shot)']),
            'Qwen3-235B_Zero': str(sample['predictions']['Qwen3-235B-A22B (Zero-shot)']),
            'Gemma_Zero': str(sample['predictions']['Gemma-3-27B (Zero-shot)'])
        })
    
    missed_df = pd.DataFrame(missed_data)
    missed_df.to_csv('all_models_missed_samples.csv', index=False)
    print(f"All models missed samples saved to: all_models_missed_samples.csv ({len(all_missed_samples)} samples)")

# Save consensus spurious samples
if all_spurious_samples:
    spurious_data = []
    for i, sample in enumerate(all_spurious_samples, 1):
        spurious_data.append({
            'Case': i,
            'Text': sample['text'],
            'Golden_Labels': str(sample['golden']),
            'Qwen3-32B_Few': str(sample['predictions']['Qwen3-32B (Few-shot)']),
            'Qwen3-235B_Few': str(sample['predictions']['Qwen3-235B-A22B (Few-shot)']),
            'Gemma_Few': str(sample['predictions']['Gemma-3-27B (Few-shot)']),
            'Qwen3-32B_Zero': str(sample['predictions']['Qwen3-32B (Zero-shot)']),
            'Qwen3-235B_Zero': str(sample['predictions']['Qwen3-235B-A22B (Zero-shot)']),
            'Gemma_Zero': str(sample['predictions']['Gemma-3-27B (Zero-shot)'])
        })
    
    spurious_df = pd.DataFrame(spurious_data)
    spurious_df.to_csv('all_models_spurious_samples.csv', index=False)
    print(f"All models spurious samples saved to: all_models_spurious_samples.csv ({len(all_spurious_samples)} samples)")

print(f"\nAnalysis complete!") 