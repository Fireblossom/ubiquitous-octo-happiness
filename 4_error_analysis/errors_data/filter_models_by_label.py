import pandas as pd
import argparse

def filter_models_by_label(csv_file, target_label, output_prefix="filtered"):
    """
    Filter rows where golden label includes a specific label but some models don't predict it
    
    Args:
        csv_file (str): Path to the CSV file
        target_label (str): The label to search for (e.g., '2', '1', '3', etc.)
        output_prefix (str): Prefix for output files
        
    Returns:
        dict: Summary statistics
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Get model columns (all columns except the first few metadata columns)
    model_columns = [col for col in df.columns if col not in ['Unnamed: 0', 'comment_id', 'comment_type', 'text', 'Golden']]
    
    print(f"Found {len(model_columns)} model columns: {model_columns}")
    
    # Function to check if a value contains the target label
    def contains_label(value, label):
        if pd.isna(value) or value == '':
            return False
        return str(label) in str(value)
    
    # Filter rows where Golden contains the target label
    golden_contains_label = df['Golden'].apply(lambda x: contains_label(x, target_label))
    df_golden_label = df[golden_contains_label].copy()
    
    print(f"Found {len(df_golden_label)} rows where Golden label contains '{target_label}'")
    
    # For each row, check if any model doesn't predict the target label
    detailed_results = []
    
    for idx, row in df_golden_label.iterrows():
        golden_has_label = contains_label(row['Golden'], target_label)
        models_missing_label = []
        models_with_label = []
        
        # Check each model's prediction
        for col in model_columns:
            if contains_label(row[col], target_label):
                models_with_label.append(col)
            else:
                models_missing_label.append(col)
        
        # If golden has the label but some models don't predict it
        if golden_has_label and models_missing_label:
            # Create detailed result with all predictions
            result = {
                'row_index': idx,
                'comment_id': row['comment_id'],
                'comment_type': row['comment_type'],
                'text': row['text'],
                'golden': row['Golden'],
                'models_missing_label': models_missing_label,
                'models_with_label': models_with_label,
                'num_models_missing_label': len(models_missing_label),
                'num_models_with_label': len(models_with_label)
            }
            
            # Add individual model predictions
            for col in model_columns:
                result[f'{col}_prediction'] = row[col]
                result[f'{col}_has_{target_label}'] = contains_label(row[col], target_label)
            
            detailed_results.append(result)
    
    print(f"Found {len(detailed_results)} rows where Golden contains '{target_label}' but some models don't predict it")
    
    # Create a summary
    summary_stats = {}
    if detailed_results:
        print(f"\n=== SUMMARY FOR LABEL '{target_label}' ===")
        print(f"Total rows with Golden containing '{target_label}': {len(df_golden_label)}")
        print(f"Rows where some models miss '{target_label}': {len(detailed_results)}")
        percentage = len(detailed_results)/len(df_golden_label)*100 if len(df_golden_label) > 0 else 0
        print(f"Percentage: {percentage:.2f}%")
        
        # Count how often each model misses the label
        model_missing_counts = {}
        model_total_counts = {}
        for result in detailed_results:
            for model in model_columns:
                model_total_counts[model] = model_total_counts.get(model, 0) + 1
                if not result[f'{model}_has_{target_label}']:
                    model_missing_counts[model] = model_missing_counts.get(model, 0) + 1
        
        print(f"\n=== MODEL PERFORMANCE (on rows where Golden has '{target_label}') ===")
        for model in model_columns:
            missing = model_missing_counts.get(model, 0)
            total = model_total_counts.get(model, 0)
            accuracy = ((total - missing) / total * 100) if total > 0 else 0
            print(f"{model}: {missing}/{total} missed '{target_label}' ({accuracy:.1f}% accuracy)")
        
        # Store summary stats
        summary_stats = {
            'target_label': target_label,
            'total_golden_with_label': len(df_golden_label),
            'rows_with_missing_label': len(detailed_results),
            'percentage_missing': percentage,
            'model_performance': {model: {
                'missing': model_missing_counts.get(model, 0),
                'total': model_total_counts.get(model, 0),
                'accuracy': ((model_total_counts.get(model, 0) - model_missing_counts.get(model, 0)) / model_total_counts.get(model, 0) * 100) if model_total_counts.get(model, 0) > 0 else 0
            } for model in model_columns}
        }
        
        # Show examples with detailed predictions
        print(f"\n=== FIRST 3 EXAMPLES WITH DETAILED PREDICTIONS ===")
        for i, result in enumerate(detailed_results[:3]):
            print(f"\nExample {i+1}:")
            print(f"Comment ID: {result['comment_id']}")
            print(f"Type: {result['comment_type']}")
            print(f"Text: {result['text'][:150]}...")
            print(f"Golden: {result['golden']}")
            print(f"Models missing '{target_label}': {result['models_missing_label']}")
            print(f"Models with '{target_label}': {result['models_with_label']}")
            print(f"\nDetailed predictions:")
            for model in model_columns:
                prediction = result[f'{model}_prediction']
                has_label = result[f'{model}_has_{target_label}']
                status = "✓" if has_label else "✗"
                print(f"  {status} {model}: {prediction}")
        
        # Save results
        output_file = f"4_error_analysis/errors_data/{output_prefix}_label_{target_label}_results.csv"
        df_results = pd.DataFrame(detailed_results)
        df_results.to_csv(output_file, index=False)
        print(f"\nDetailed results saved to: {output_file}")
        
        # Create comparison matrix
        matrix_data = []
        for result in detailed_results:
            row_data = {
                'comment_id': result['comment_id'],
                'text': result['text'],
                'golden': result['golden'],
                f'golden_has_{target_label}': target_label in str(result['golden'])
            }
            
            for model in model_columns:
                row_data[model] = '✓' if result[f'{model}_has_{target_label}'] else '✗'
            
            matrix_data.append(row_data)
        
        df_matrix = pd.DataFrame(matrix_data)
        matrix_output_file = f"4_error_analysis/errors_data/{output_prefix}_label_{target_label}_matrix.csv"
        df_matrix.to_csv(matrix_output_file, index=False)
        print(f"Model comparison matrix saved to: {matrix_output_file}")
    
    return detailed_results, summary_stats

def analyze_all_labels(csv_file):
    """
    Analyze all labels that appear in the Golden column
    
    Args:
        csv_file (str): Path to the CSV file
    """
    df = pd.read_csv(csv_file)
    
    # Extract all unique labels from Golden column
    all_labels = set()
    for value in df['Golden'].dropna():
        if isinstance(value, str):
            # Split by comma and extract numbers
            parts = value.split(',')
            for part in parts:
                part = part.strip()
                if part.startswith('认同逻辑'):
                    label = part.replace('认同逻辑', '')
                    if label.isdigit():
                        all_labels.add(int(label))
                elif part.isdigit():
                    all_labels.add(int(part))
    
    print(f"Found labels in Golden column: {sorted(all_labels)}")
    
    # Analyze each label
    all_summaries = {}
    for label in sorted(all_labels):
        print(f"\n{'='*50}")
        print(f"ANALYZING LABEL: {label}")
        print(f"{'='*50}")
        
        results, summary = filter_models_by_label(csv_file, str(label), f"label_{label}")
        all_summaries[label] = summary
    
    # Create overall summary
    print(f"\n{'='*50}")
    print("OVERALL SUMMARY")
    print(f"{'='*50}")
    
    for label, summary in all_summaries.items():
        if summary:
            print(f"\nLabel {label}:")
            print(f"  Total cases: {summary['total_golden_with_label']}")
            print(f"  Missing cases: {summary['rows_with_missing_label']}")
            print(f"  Missing percentage: {summary['percentage_missing']:.1f}%")
            
            # Find best and worst performing models
            performances = summary['model_performance']
            best_model = max(performances.items(), key=lambda x: x[1]['accuracy'])
            worst_model = min(performances.items(), key=lambda x: x[1]['accuracy'])
            
            print(f"  Best model: {best_model[0]} ({best_model[1]['accuracy']:.1f}%)")
            print(f"  Worst model: {worst_model[0]} ({worst_model[1]['accuracy']:.1f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Filter model predictions by label')
    parser.add_argument('--csv_file', default='3_evaluation_results/sample_all.csv', 
                       help='Path to the CSV file')
    parser.add_argument('--label', type=str, help='Label to filter for (e.g., "2", "1", "3")')
    parser.add_argument('--analyze_all', action='store_true', 
                       help='Analyze all labels found in the Golden column')
    
    args = parser.parse_args()
    
    if args.analyze_all:
        analyze_all_labels(args.csv_file)
    elif args.label:
        filter_models_by_label(args.csv_file, args.label)
    else:
        # Default to label "2" if no arguments provided
        print("No label specified, defaulting to label '2'")
        filter_models_by_label(args.csv_file, "2") 