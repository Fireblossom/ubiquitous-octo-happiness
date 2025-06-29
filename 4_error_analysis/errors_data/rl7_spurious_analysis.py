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

# Find RL7 spurious samples
def find_rl7_spurious_samples():
    """Find all samples where RL7 was spuriously predicted (not in golden but predicted)"""
    
    rl7_spurious_samples = []
    
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
        
        # Check if RL7 is NOT in golden labels but predicted by some models
        if 7 not in golden_labels:
            # Check which models spuriously predicted RL7
            models_spurious_rl7 = []
            models_correct_no_rl7 = []
            
            for model_name, predicted in model_predictions.items():
                if 7 in predicted:
                    models_spurious_rl7.append(model_name)
                else:
                    models_correct_no_rl7.append(model_name)
            
            # If any model spuriously predicted RL7, add to our analysis
            if models_spurious_rl7:
                rl7_spurious_samples.append({
                    'text': text,
                    'golden': golden_labels,
                    'models_spurious_rl7': models_spurious_rl7,
                    'models_correct_no_rl7': models_correct_no_rl7,
                    'all_predictions': model_predictions
                })
    
    return rl7_spurious_samples

# Analyze RL7 spurious samples
print("="*80)
print("RL7 (OCCUPATIONAL TYPIFICATION) SPURIOUS ANALYSIS")
print("="*80)

rl7_spurious_samples = find_rl7_spurious_samples()

print(f"\nTotal RL7 spurious samples: {len(rl7_spurious_samples)}")

# Analyze patterns
all_spurious_count = 0
partial_spurious_count = 0
few_shot_spurious_count = 0
zero_shot_spurious_count = 0

for sample in rl7_spurious_samples:
    # Count how many models spuriously predicted RL7
    spurious_count = len(sample['models_spurious_rl7'])
    total_models = len(sample['all_predictions'])
    
    if spurious_count == total_models:
        all_spurious_count += 1
    else:
        partial_spurious_count += 1
    
    # Check few-shot vs zero-shot
    few_shot_spurious = any('Few-shot' in model for model in sample['models_spurious_rl7'])
    zero_shot_spurious = any('Zero-shot' in model for model in sample['models_spurious_rl7'])
    
    if few_shot_spurious:
        few_shot_spurious_count += 1
    if zero_shot_spurious:
        zero_shot_spurious_count += 1

print(f"\nRL7 Spurious Patterns:")
print(f"- All models spuriously predicted RL7: {all_spurious_count} samples")
print(f"- Some models spuriously predicted RL7: {partial_spurious_count} samples")
print(f"- Few-shot models spuriously predicted RL7: {few_shot_spurious_count} samples")
print(f"- Zero-shot models spuriously predicted RL7: {zero_shot_spurious_count} samples")

# Categorize samples by content
def categorize_rl7_spurious_samples(samples):
    """Categorize RL7 spurious samples by content patterns"""
    categories = {
        'work_related': [],
        'professional_terms': [],
        'social_class': [],
        'economic_terms': [],
        'other': []
    }
    
    for sample in samples:
        text = sample['text'].lower()
        
        # Work-related terms
        work_keywords = ['工作', '上班', '打工', '职业', '行业', '公司', '企业', '老板', '员工', '同事']
        if any(keyword in text for keyword in work_keywords):
            categories['work_related'].append(sample)
        # Professional terms
        elif any(term in text for term in ['医生', '律师', '教师', '工程师', '程序员', '设计师', '经理', '主管']):
            categories['professional_terms'].append(sample)
        # Social class terms
        elif any(term in text for term in ['富人', '穷人', '中产', '精英', '底层', '上层', '阶级', '阶层']):
            categories['social_class'].append(sample)
        # Economic terms
        elif any(term in text for term in ['工资', '收入', '薪水', '待遇', '福利', '奖金', '分红', '投资', '理财']):
            categories['economic_terms'].append(sample)
        else:
            categories['other'].append(sample)
    
    return categories

# Show detailed examples
print(f"\n{'='*80}")
print("DETAILED RL7 SPURIOUS EXAMPLES")
print(f"{'='*80}")

for i, sample in enumerate(rl7_spurious_samples[:20], 1):  # Show first 20
    print(f"\nCase {i}:")
    print(f"Text: {sample['text'][:200]}...")
    print(f"Golden Labels: {sample['golden']}")
    print(f"Models that SPURIOUSLY predicted RL7: {sample['models_spurious_rl7']}")
    print(f"Models that CORRECTLY did not predict RL7: {sample['models_correct_no_rl7']}")
    print("All Model Predictions:")
    for model_name, predicted in sample['all_predictions'].items():
        status = "✗ Spurious RL7" if 7 in predicted else "✓ No RL7"
        print(f"  {model_name}: {predicted} ({status})")
    print("-" * 60)

# Categorize samples
categories = categorize_rl7_spurious_samples(rl7_spurious_samples)

print(f"\n{'='*80}")
print("RL7 SPURIOUS SAMPLES BY CATEGORY")
print(f"{'='*80}")

for category, samples in categories.items():
    if samples:
        print(f"\n{category.upper()} ({len(samples)} samples):")
        for i, sample in enumerate(samples[:3], 1):  # Show first 3 of each category
            print(f"  {i}. {sample['text'][:100]}...")
            print(f"     Golden: {sample['golden']}")
            print(f"     Spuriously predicted by: {len(sample['models_spurious_rl7'])} models")

# Save detailed results
print(f"\n{'='*80}")
print("SAVING DETAILED RESULTS")
print(f"{'='*80}")

# Save RL7 spurious samples
rl7_data = []
for i, sample in enumerate(rl7_spurious_samples, 1):
    rl7_data.append({
        'Case': i,
        'Text': sample['text'],
        'Golden_Labels': str(sample['golden']),
        'Models_Spurious_RL7': str(sample['models_spurious_rl7']),
        'Models_Correct_No_RL7': str(sample['models_correct_no_rl7']),
        'Qwen3-32B_Few': str(sample['all_predictions']['Qwen3-32B (Few-shot)']),
        'Qwen3-235B_Few': str(sample['all_predictions']['Qwen3-235B-A22B (Few-shot)']),
        'Gemma_Few': str(sample['all_predictions']['Gemma-3-27B (Few-shot)']),
        'Qwen3-32B_Zero': str(sample['all_predictions']['Qwen3-32B (Zero-shot)']),
        'Qwen3-235B_Zero': str(sample['all_predictions']['Qwen3-235B-A22B (Zero-shot)']),
        'Gemma_Zero': str(sample['all_predictions']['Gemma-3-27B (Zero-shot)'])
    })

rl7_df = pd.DataFrame(rl7_data)
rl7_df.to_csv('rl7_spurious_samples.csv', index=False)
print(f"RL7 spurious samples saved to: rl7_spurious_samples.csv ({len(rl7_spurious_samples)} samples)")

# Save category analysis
category_data = []
for category, samples in categories.items():
    if samples:
        category_data.append({
            'Category': category,
            'Count': len(samples),
            'Percentage': len(samples) / len(rl7_spurious_samples) * 100
        })

category_df = pd.DataFrame(category_data)
category_df.to_csv('rl7_spurious_categories.csv', index=False)
print(f"RL7 category analysis saved to: rl7_spurious_categories.csv")

print(f"\nAnalysis complete!") 