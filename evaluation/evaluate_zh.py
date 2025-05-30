import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

def parse_annotator(value):
    """Parses the 'Golden' column (e.g., "147" means labels 1, 4, 7)."""
    if pd.isna(value) or str(value).strip() == "-":
        return set()
    # Ensure value is treated as a string of digits
    return set(map(int, list(str(int(value))))) # int(value) handles potential float like 4.0

def parse_RL_category(value):
    """Parses the 'RL Category' column (e.g., "认同逻辑1, 认同逻辑4" or "认同逻辑1")."""
    if pd.isna(value) or not str(value).strip():
        return set()
    
    labels = set()
    # Split by comma, then process each part
    items = str(value).split(',')
    for item in items:
        item = item.strip() # Remove leading/trailing whitespace
        if "认同逻辑" in item:
            try:
                # Extract digits after "认同逻辑"
                label_num = int(item.split("认同逻辑")[1])
                labels.add(label_num)
            except ValueError:
                # Handle cases where number parsing fails
                print(f"Warning: Could not parse number from item: '{item}' in '{value}'")
    return labels

# Load the data
# If you have a CSV file, use: df = pd.read_csv('your_file.csv')
df = pd.read_csv('sample_all.csv')
annotator = df['Golden'].apply(parse_annotator)

for column in df.columns[5:]:
    if column.endswith('en'):
        continue
    print(f"\n\n -----------------\nEvaluating {column}")
    predictions = df[column].apply(parse_RL_category)

    # --- 2. Helper Function for Evaluation ---
    def evaluate_predictions(y_true_parsed_list, y_pred_parsed_list, true_label_source_name, pred_label_source_name):
        """
        Calculates and prints F1 scores and classification report for a given pair of true and predicted labels.
        Filters out samples where true labels are empty.
        """
        print(f"\n\n{'='*10} EVALUATING: {pred_label_source_name} (Predictions) vs. {true_label_source_name} (Golden Labels) {'='*10}")

        # Filter out samples where true labels are empty
        filtered_pairs = [(true, pred) for true, pred in zip(y_true_parsed_list, y_pred_parsed_list) if true]
        
        if not filtered_pairs:
            print("No annotated samples found after filtering. Skipping metrics calculation.")
            return
            
        y_true_filtered, y_pred_filtered = zip(*filtered_pairs)
        
        # Determine all unique labels present in this specific pairing
        all_labels_in_pair = set()
        for labels_set in y_true_filtered:
            all_labels_in_pair.update(labels_set)
        for labels_set in y_pred_filtered:
            all_labels_in_pair.update(labels_set)

        if not all_labels_in_pair:
            print("No labels found for this evaluation pair. Skipping metrics calculation.")
            return

        sorted_unique_labels = sorted(list(all_labels_in_pair))

        # Binarize labels for this specific pair
        mlb = MultiLabelBinarizer(classes=sorted_unique_labels)
        y_true_binarized = mlb.fit_transform(y_true_filtered)
        y_pred_binarized = mlb.transform(y_pred_filtered) # Use transform for predictions

        print(f"Classes considered for this evaluation: {mlb.classes_}")
        print(f"Number of samples after filtering non-annotated: {len(y_true_filtered)}")
        # print("Binarized True:\n", y_true_binarized)
        # print("Binarized Pred:\n", y_pred_binarized)

        # Calculate F1 Scores
        print("\n--- F1 Scores ---")
        f1_micro = f1_score(y_true_binarized, y_pred_binarized, average='micro', zero_division=0)
        print(f"F1 Score (micro): {f1_micro:.4f}")

        f1_macro = f1_score(y_true_binarized, y_pred_binarized, average='macro', zero_division=0)
        print(f"F1 Score (macro): {f1_macro:.4f}")

        f1_weighted = f1_score(y_true_binarized, y_pred_binarized, average='weighted', zero_division=0)
        print(f"F1 Score (weighted): {f1_weighted:.4f}")

        f1_samples = f1_score(y_true_binarized, y_pred_binarized, average='samples', zero_division=0)
        print(f"F1 Score (samples): {f1_samples:.4f}")
        print("-" * 20)

        # Full Classification Report
        # Create target names based on the actual classes found by the binarizer for this pair
        report_target_names = [f"RL{label}" for label in mlb.classes_]

        print("\n--- Classification Report ---")
        try:
            report = classification_report(
                y_true_binarized,
                y_pred_binarized,
                target_names=report_target_names,
                zero_division=0
            )
            print(report)
        except ValueError as e:
            print(f"Could not generate classification report: {e}")
            print("This can happen if some classes in `target_names` are not present in `y_true_binarized` or `y_pred_binarized` after binarization.")

        precision_micro_overall = precision_score(y_true_binarized, y_pred_binarized, average='micro', zero_division=0)
        recall_micro_overall = recall_score(y_true_binarized, y_pred_binarized, average='micro', zero_division=0)
        print(f"\nOverall Micro Precision: {precision_micro_overall:.4f}")
        print(f"Overall Micro Recall:    {recall_micro_overall:.4f}")

    # Evaluation 2: RL Category vs. Anotator T
    evaluate_predictions(
        annotator.tolist(),
        predictions.tolist(),
        true_label_source_name="Anotator T",
        pred_label_source_name="RL Category"
    )

print("\n\nScript finished.")