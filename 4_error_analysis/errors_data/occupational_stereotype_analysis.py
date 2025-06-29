import pandas as pd
import numpy as np

# Load the RL7 missed samples
rl7_data = pd.read_csv('../4_error_analysis/errors_data/rl7_missed_samples.csv')

# Define Occupational Stereotype samples based on the analysis
occupational_stereotype_samples = [
    {
        'case': 3,
        'text': "其实没必要比，厦门本地人只要在厦门有房子很多都想要读完书回厦门工作的，厦门工资总体不高是因为底层人太多了，实际上厦门的国企包括公务员的工资平均都要比福州高很多。去其他地方读书包括香港也呆过几年，包括去世界各地旅游，真心觉得一个集中环境教育交通医疗都在线的城市厦门算是其中一个非常非常好的，尤其是环境，基本上很少看到都市圈环境能做到这样的，人其实也不算多，全国中等吧",
        'golden': {5, 6, 7},
        'keywords': ['底层人', '国企', '公务员', '工资']
    },
    {
        'case': 7,
        'text': "就是。。。基本上现在接到那些骚扰电话都是外地口音哎，不是歧视外地人，只是觉得如果他们回老家，做点其他工作，也比鼓捣留在成都打骚扰电话好嘛",
        'golden': {4, 7},
        'keywords': ['骚扰电话', '其他工作', '打工']
    },
    {
        'case': 14,
        'text': "东城：东直门，雍和宫，朝外SOHO（？） 西城：安贞？安慧？蓟门桥？南礼士路 海淀：各大知名院校，如清华北大等，中关村，五道口，也是程序员聚集地 朝阳：京爷，四合院，城根底下，繁华都市 丰台：有点偏，外来打工人租房子或者住员工宿舍的地方，有些地方还没开发出来，稍微有点沧桑 石景山：不熟悉，也挺偏的，就知道个八角游乐园 门头沟：山里？挺偏的，那边的人挺横，打架不好欺负 房山：太偏了，总感觉快出北京了，那边房子相对较便宜",
        'golden': {1, 5, 7},
        'keywords': ['外来打工人', '员工宿舍', '程序员聚集地']
    }
]

print("="*80)
print("OCCUPATIONAL STEREOTYPE ANALYSIS")
print("="*80)

print(f"\nTotal Occupational Stereotype samples: {len(occupational_stereotype_samples)}")

# Analyze each sample
for i, sample in enumerate(occupational_stereotype_samples, 1):
    print(f"\n{'='*60}")
    print(f"Sample {i}: Case {sample['case']}")
    print(f"{'='*60}")
    
    print(f"Text: {sample['text'][:150]}...")
    print(f"Golden Labels: {sample['golden']}")
    print(f"Occupational Keywords: {sample['keywords']}")
    
    # Get the corresponding row from the CSV
    row = rl7_data[rl7_data['Case'] == sample['case']].iloc[0]
    
    print(f"\nModel Predictions:")
    print(f"  Qwen3-32B (Few-shot):     {row['Qwen3-32B_Few']}")
    print(f"  Qwen3-235B-A22B (Few-shot): {row['Qwen3-235B_Few']}")
    print(f"  Gemma-3-27B (Few-shot):    {row['Gemma_Few']}")
    print(f"  Qwen3-32B (Zero-shot):     {row['Qwen3-32B_Zero']}")
    print(f"  Qwen3-235B-A22B (Zero-shot): {row['Qwen3-235B_Zero']}")
    print(f"  Gemma-3-27B (Zero-shot):    {row['Gemma_Zero']}")
    
    # Check which models got RL7 correct
    models_correct = []
    models_missed = []
    
    if '7' in str(row['Qwen3-32B_Few']):
        models_correct.append('Qwen3-32B (Few-shot)')
    else:
        models_missed.append('Qwen3-32B (Few-shot)')
    
    if '7' in str(row['Qwen3-235B_Few']):
        models_correct.append('Qwen3-235B-A22B (Few-shot)')
    else:
        models_missed.append('Qwen3-235B-A22B (Few-shot)')
    
    if '7' in str(row['Gemma_Few']):
        models_correct.append('Gemma-3-27B (Few-shot)')
    else:
        models_missed.append('Gemma-3-27B (Few-shot)')
    
    if '7' in str(row['Qwen3-32B_Zero']):
        models_correct.append('Qwen3-32B (Zero-shot)')
    else:
        models_missed.append('Qwen3-32B (Zero-shot)')
    
    if '7' in str(row['Qwen3-235B_Zero']):
        models_correct.append('Qwen3-235B-A22B (Zero-shot)')
    else:
        models_missed.append('Qwen3-235B-A22B (Zero-shot)')
    
    if '7' in str(row['Gemma_Zero']):
        models_correct.append('Gemma-3-27B (Zero-shot)')
    else:
        models_missed.append('Gemma-3-27B (Zero-shot)')
    
    print(f"\nResults:")
    print(f"  Models that GOT RL7: {models_correct}")
    print(f"  Models that MISSED RL7: {models_missed}")
    
    # Analyze the pattern
    few_shot_correct = [m for m in models_correct if 'Few-shot' in m]
    zero_shot_correct = [m for m in models_correct if 'Zero-shot' in m]
    
    print(f"  Few-shot models correct: {len(few_shot_correct)}/3")
    print(f"  Zero-shot models correct: {len(zero_shot_correct)}/3")

# Overall statistics
print(f"\n{'='*80}")
print("OVERALL OCCUPATIONAL STEREOTYPE STATISTICS")
print(f"{'='*80}")

total_samples = len(occupational_stereotype_samples)
total_predictions = total_samples * 6  # 6 models per sample

# Count correct predictions
correct_predictions = 0
few_shot_correct = 0
zero_shot_correct = 0

for sample in occupational_stereotype_samples:
    row = rl7_data[rl7_data['Case'] == sample['case']].iloc[0]
    
    # Count correct predictions
    if '7' in str(row['Qwen3-32B_Few']):
        correct_predictions += 1
        few_shot_correct += 1
    if '7' in str(row['Qwen3-235B_Few']):
        correct_predictions += 1
        few_shot_correct += 1
    if '7' in str(row['Gemma_Few']):
        correct_predictions += 1
        few_shot_correct += 1
    if '7' in str(row['Qwen3-32B_Zero']):
        correct_predictions += 1
        zero_shot_correct += 1
    if '7' in str(row['Qwen3-235B_Zero']):
        correct_predictions += 1
        zero_shot_correct += 1
    if '7' in str(row['Gemma_Zero']):
        correct_predictions += 1
        zero_shot_correct += 1

print(f"\nOverall Statistics:")
print(f"  Total predictions: {total_predictions}")
print(f"  Correct predictions: {correct_predictions}")
print(f"  Accuracy: {correct_predictions/total_predictions*100:.1f}%")
print(f"  Few-shot correct: {few_shot_correct}/9 ({few_shot_correct/9*100:.1f}%)")
print(f"  Zero-shot correct: {zero_shot_correct}/9 ({zero_shot_correct/9*100:.1f}%)")

# Model-specific performance
print(f"\nModel-Specific Performance:")
models = ['Qwen3-32B (Few-shot)', 'Qwen3-235B-A22B (Few-shot)', 'Gemma-3-27B (Few-shot)', 
          'Qwen3-32B (Zero-shot)', 'Qwen3-235B-A22B (Zero-shot)', 'Gemma-3-27B (Zero-shot)']

for model in models:
    correct = 0
    for sample in occupational_stereotype_samples:
        row = rl7_data[rl7_data['Case'] == sample['case']].iloc[0]
        if model == 'Qwen3-32B (Few-shot)' and '7' in str(row['Qwen3-32B_Few']):
            correct += 1
        elif model == 'Qwen3-235B-A22B (Few-shot)' and '7' in str(row['Qwen3-235B_Few']):
            correct += 1
        elif model == 'Gemma-3-27B (Few-shot)' and '7' in str(row['Gemma_Few']):
            correct += 1
        elif model == 'Qwen3-32B (Zero-shot)' and '7' in str(row['Qwen3-32B_Zero']):
            correct += 1
        elif model == 'Qwen3-235B-A22B (Zero-shot)' and '7' in str(row['Qwen3-235B_Zero']):
            correct += 1
        elif model == 'Gemma-3-27B (Zero-shot)' and '7' in str(row['Gemma_Zero']):
            correct += 1
    
    print(f"  {model}: {correct}/{total_samples} ({correct/total_samples*100:.1f}%)")

# Pattern analysis
print(f"\nPattern Analysis:")
print(f"  Sample 1 (底层人): Most models missed, only Qwen3-235B-A22B (Few-shot) got it")
print(f"  Sample 2 (骚扰电话): Qwen3-32B (Few-shot) got it, others missed")
print(f"  Sample 3 (外来打工人): Qwen3-235B-A22B (Few-shot) and Gemma-3-27B (Few-shot) got it")

print(f"\nKey Findings:")
print(f"  1. Zero-shot models completely failed on all occupational stereotype samples")
print(f"  2. Few-shot models show some capability but still miss most cases")
print(f"  3. Qwen3-235B-A22B (Few-shot) performed best with 2/3 correct")
print(f"  4. Occupational stereotypes are subtle and require context understanding")
print(f"  5. Keywords like '底层人', '外来打工人', '骚扰电话' are key indicators")

print(f"\nAnalysis complete!") 