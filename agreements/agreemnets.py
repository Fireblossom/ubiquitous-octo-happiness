import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score

# --- 1. Define Parsing Functions ---
def parse_annotator_f(value):
    """Parses the 'Anotator F' column (e.g., "1,4,7" or "-")."""
    if pd.isna(value) or str(value).strip() == "-":
        return set()
    return set(map(int, str(value).split(',')))

def parse_annotator_t(value):
    """Parses the 'Anotator T' column (e.g., "147" means labels 1, 4, 7)."""
    if pd.isna(value) or str(value).strip() == "-":
        return set()
    # Ensure value is treated as a string of digits
    return set(map(int, list(str(int(value))))) # int(value) handles potential float like 4.0

# Load the data
for i in range(1, 4):
    df = pd.read_csv(f'round{str(i)}.csv')
    annotator_f = df['Anotator F'].apply(parse_annotator_f)
    annotator_t = df['Anotator T'].apply(parse_annotator_t)

    # Get all unique labels across all annotations
    all_labels = set()
    for ann in annotator_f:
        all_labels.update(ann)
    for ann in annotator_t:
        all_labels.update(ann)
    all_labels = sorted(list(all_labels))

    print(f"\nLabel Distribution Analysis: Round{str(i)}")
    print("-" * 50)

    # Calculate label distribution for each annotator
    for annotator, name in [(annotator_f, "F"), (annotator_t, "T")]:
        print(f"\nAnnotator {name} Label Distribution:")
        label_counts = {label: 0 for label in range(1, 8)}  # Initialize counts for labels 1-7
        
        # Count occurrences of each label
        for ann in annotator:
            for label in ann:
                if label in label_counts:
                    label_counts[label] += 1
        
        # Print distribution
        total_samples = len(df)
        for label in range(1, 8):
            count = label_counts[label]
            percentage = (count / total_samples) * 100
            print(f"Label {label}: {count} samples ({percentage:.1f}%)")

    # Calculate agreement metrics
    print(f"\nAgreement Analysis: Round{str(i)}")
    print("-" * 50)

    # 1. Calculate Jaccard similarity for each pair of annotations
    jaccard_scores = []
    for f_ann, t_ann in zip(annotator_f, annotator_t):
        if not f_ann and not t_ann:  # Both empty
            jaccard_scores.append(1.0)
        elif not f_ann or not t_ann:  # One empty
            jaccard_scores.append(0.0)
        else:
            intersection = len(f_ann.intersection(t_ann))
            union = len(f_ann.union(t_ann))
            jaccard_scores.append(intersection / union)

    mean_jaccard = np.mean(jaccard_scores)
    print(f"Mean Jaccard Similarity: {mean_jaccard:.3f}")

    # 2. Calculate exact agreement (both annotators chose exactly the same labels)
    exact_agreements = sum(1 for f_ann, t_ann in zip(annotator_f, annotator_t) if f_ann == t_ann)
    exact_agreement_rate = exact_agreements / len(df)
    print(f"Exact Agreement Rate: {exact_agreement_rate:.3f}")

    # 3. Calculate label-wise agreement and Cohen's Kappa for each label
    label_agreements = {}
    label_kappas = {}
    for label in all_labels:
        # Convert annotations to binary (label present/absent) for each annotator
        f_binary = [1 if label in ann else 0 for ann in annotator_f]
        t_binary = [1 if label in ann else 0 for ann in annotator_t]
        
        # Calculate agreement rate
        agreements = sum(1 for f, t in zip(f_binary, t_binary) if f == t)
        label_agreements[label] = agreements / len(df)
        
        # Calculate Cohen's Kappa for this label
        kappa = cohen_kappa_score(f_binary, t_binary)
        label_kappas[label] = kappa

    # Calculate mean Cohen's Kappa
    mean_kappa = np.mean(list(label_kappas.values()))
    print(f"\nOverall Agreement Metrics:")
    print(f"Mean Cohen's Kappa: {mean_kappa:.3f}")
    print(f"Mean Jaccard Similarity: {mean_jaccard:.3f}")
    print(f"Exact Agreement Rate: {exact_agreement_rate:.3f}")

    print("\nLabel-wise Agreement Rates and Cohen's Kappa:")
    for label in sorted(all_labels):
        print(f"Label {label}:")
        print(f"  Agreement Rate: {label_agreements[label]:.3f}")
        print(f"  Cohen's Kappa: {label_kappas[label]:.3f}")