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

# Find RL7 missed samples
def find_rl7_missed_samples():
    """Find all samples where RL7 was in golden but missed by models"""
    
    rl7_missed_samples = []
    
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
        
        # Check if RL7 is in golden labels
        if 7 in golden_labels:
            # Check which models missed RL7
            models_missed_rl7 = []
            models_correct_rl7 = []
            
            for model_name, predicted in model_predictions.items():
                if 7 in predicted:
                    models_correct_rl7.append(model_name)
                else:
                    models_missed_rl7.append(model_name)
            
            # If any model missed RL7, add to our analysis
            if models_missed_rl7:
                rl7_missed_samples.append({
                    'text': text,
                    'golden': golden_labels,
                    'models_missed_rl7': models_missed_rl7,
                    'models_correct_rl7': models_correct_rl7,
                    'all_predictions': model_predictions
                })
    
    return rl7_missed_samples

# Analyze RL7 missed samples
print("="*80)
print("RL7 (OCCUPATIONAL TYPIFICATION) MISSED ANALYSIS")
print("="*80)

rl7_missed_samples = find_rl7_missed_samples()

print(f"\nTotal RL7 missed samples: {len(rl7_missed_samples)}")

# Analyze patterns
all_missed_count = 0
partial_missed_count = 0
few_shot_missed_count = 0
zero_shot_missed_count = 0

for sample in rl7_missed_samples:
    # Count how many models missed RL7
    missed_count = len(sample['models_missed_rl7'])
    total_models = len(sample['all_predictions'])
    
    if missed_count == total_models:
        all_missed_count += 1
    else:
        partial_missed_count += 1
    
    # Check few-shot vs zero-shot
    few_shot_missed = any('Few-shot' in model for model in sample['models_missed_rl7'])
    zero_shot_missed = any('Zero-shot' in model for model in sample['models_missed_rl7'])
    
    if few_shot_missed:
        few_shot_missed_count += 1
    if zero_shot_missed:
        zero_shot_missed_count += 1

print(f"\nRL7 Missed Patterns:")
print(f"- All models missed RL7: {all_missed_count} samples")
print(f"- Some models missed RL7: {partial_missed_count} samples")
print(f"- Few-shot models missed RL7: {few_shot_missed_count} samples")
print(f"- Zero-shot models missed RL7: {zero_shot_missed_count} samples")

# Show detailed examples
print(f"\n{'='*80}")
print("DETAILED RL7 MISSED EXAMPLES")
print(f"{'='*80}")

for i, sample in enumerate(rl7_missed_samples[:20], 1):  # Show first 20
    print(f"\nCase {i}:")
    print(f"Text: {sample['text'][:200]}...")
    print(f"Golden Labels: {sample['golden']}")
    print(f"Models that MISSED RL7: {sample['models_missed_rl7']}")
    print(f"Models that GOT RL7: {sample['models_correct_rl7']}")
    print("All Model Predictions:")
    for model_name, predicted in sample['all_predictions'].items():
        status = "✓ RL7" if 7 in predicted else "✗ No RL7"
        print(f"  {model_name}: {predicted} ({status})")
    print("-" * 60)

# Categorize RL7 missed samples by content type
def categorize_rl7_samples(samples):
    """Categorize RL7 samples by their content characteristics"""
    
    categories = {
        'wholesale_market': [],
        'professional_association': [],
        'occupational_stereotype': [],
        'class_based': [],
        'industry_related': [],
        'other': []
    }
    
    keywords = {
        'wholesale_market': ['批发', '市场', '档口', '商户', '生意', '买卖'],
        'professional_association': ['协会', '工会', '行业', '专业', '职业'],
        'occupational_stereotype': ['打工', '上班', '工作', '职业', '行业'],
        'class_based': ['阶层', '阶级', '富人', '穷人', '有钱', '没钱'],
        'industry_related': ['工厂', '企业', '公司', '单位', '部门']
    }
    
    for sample in samples:
        text = sample['text'].lower()
        categorized = False
        
        for category, words in keywords.items():
            if any(word in text for word in words):
                categories[category].append(sample)
                categorized = True
                break
        
        if not categorized:
            categories['other'].append(sample)
    
    return categories

# Categorize samples
categories = categorize_rl7_samples(rl7_missed_samples)

print(f"\n{'='*80}")
print("RL7 MISSED SAMPLES BY CATEGORY")
print(f"{'='*80}")

for category, samples in categories.items():
    if samples:
        print(f"\n{category.upper()} ({len(samples)} samples):")
        for i, sample in enumerate(samples[:3], 1):  # Show first 3 of each category
            print(f"  {i}. {sample['text'][:100]}...")
            print(f"     Golden: {sample['golden']}")
            print(f"     Missed by: {len(sample['models_missed_rl7'])} models")

# Save detailed results
print(f"\n{'='*80}")
print("SAVING DETAILED RESULTS")
print(f"{'='*80}")

# Save RL7 missed samples
rl7_data = []
for i, sample in enumerate(rl7_missed_samples, 1):
    rl7_data.append({
        'Case': i,
        'Text': sample['text'],
        'Golden_Labels': str(sample['golden']),
        'Models_Missed_RL7': str(sample['models_missed_rl7']),
        'Models_Correct_RL7': str(sample['models_correct_rl7']),
        'Qwen3-32B_Few': str(sample['all_predictions']['Qwen3-32B (Few-shot)']),
        'Qwen3-235B_Few': str(sample['all_predictions']['Qwen3-235B-A22B (Few-shot)']),
        'Gemma_Few': str(sample['all_predictions']['Gemma-3-27B (Few-shot)']),
        'Qwen3-32B_Zero': str(sample['all_predictions']['Qwen3-32B (Zero-shot)']),
        'Qwen3-235B_Zero': str(sample['all_predictions']['Qwen3-235B-A22B (Zero-shot)']),
        'Gemma_Zero': str(sample['all_predictions']['Gemma-3-27B (Zero-shot)'])
    })

rl7_df = pd.DataFrame(rl7_data)
rl7_df.to_csv('rl7_missed_samples.csv', index=False)
print(f"RL7 missed samples saved to: rl7_missed_samples.csv ({len(rl7_missed_samples)} samples)")

# Save category analysis
category_data = []
for category, samples in categories.items():
    if samples:
        category_data.append({
            'Category': category,
            'Count': len(samples),
            'Percentage': len(samples) / len(rl7_missed_samples) * 100
        })

category_df = pd.DataFrame(category_data)
category_df.to_csv('rl7_missed_categories.csv', index=False)
print(f"RL7 category analysis saved to: rl7_missed_categories.csv")

print(f"\nAnalysis complete!") 