import numpy as np
import pandas as pd
from collections import Counter

def per_abbreviation_analysis(df, idx_test, y_test, y_pred):
    """Analyze performance per abbreviation"""
    print("Per-Abbreviation Performance:")
    for abbrev in ['CC', 'CP', 'SA']:
        mask = df.loc[idx_test, 'abbreviation'].values == abbrev
        abbrev_true = y_test[mask]
        abbrev_pred = y_pred[mask]
        accuracy = (abbrev_true == abbrev_pred).mean()
        print(f"{abbrev}: {accuracy:.4f}")

def confusion_analysis(y_test, y_pred, classes):
    """Find most confused class pairs"""
    from evaluation import confusion_matrix
    cm = confusion_matrix(y_test, y_pred, classes)
    
    print("\nMost Confused Pairs:")
    confusion_pairs = []
    for i, true_class in enumerate(classes):
        for j, pred_class in enumerate(classes):
            if i != j:
                count = cm.iloc[i, j]
                if count > 0:
                    confusion_pairs.append((true_class, pred_class, count))
    
    for true_c, pred_c, count in sorted(confusion_pairs, key=lambda x: x[2], reverse=True)[:10]:
        print(f"{true_c} → {pred_c}: {count}")

def feature_importance(model, vocabulary, top_n=10):
    """Show top discriminative features per class"""
    print("\nTop Discriminative Features:")
    inv_vocab = {v: k for k, v in vocabulary.items()}
    
    for c in model.classes:
        feature_probs = model.feature_counts[c]
        top_indices = np.argsort(feature_probs)[-top_n:][::-1]
        top_features = [inv_vocab.get(idx, f"idx_{idx}") for idx in top_indices]
        print(f"{c}: {top_features}")

def analyze_failures(df, idx_test, y_test, y_pred, n_examples=15):
    """Detailed failure case analysis"""
    from feature_extraction import extract_context
    
    failure_mask = y_pred != y_test
    failure_indices = idx_test[failure_mask]
    
    print(f"\nFailure Analysis ({len(failure_indices)} total failures):")
    
    # Sample failures
    sample_size = min(n_examples, len(failure_indices))
    sample_failures = np.random.choice(failure_indices, size=sample_size, replace=False)
    
    failure_data = []
    for idx in sample_failures:
        row = df.loc[idx]
        pred_idx = np.where(idx_test == idx)[0][0]
        pred = y_pred[pred_idx]
        
        context_words = extract_context(row)
        context_str = ' '.join(context_words[:15])
        
        failure_data.append({
            'Abbreviation': row['abbreviation'],
            'True': row['label'],
            'Predicted': pred,
            'Context': context_str + '...'
        })
    
    failure_df = pd.DataFrame(failure_data)
    print(failure_df.to_string(index=False))
    
    # Failure patterns
    print("\nFailure Patterns:")
    failure_abbrevs = df.loc[failure_indices, 'abbreviation'].value_counts()
    print("By abbreviation:")
    print(failure_abbrevs)

def analyze_successes(df, idx_test, y_test, y_pred, n_examples=10):
    """Show successful predictions"""
    from feature_extraction import extract_context
    
    success_mask = y_pred == y_test
    success_indices = idx_test[success_mask]
    
    sample_size = min(n_examples, len(success_indices))
    sample_successes = np.random.choice(success_indices, size=sample_size, replace=False)
    
    print(f"\nSuccess Examples ({sample_size}):")
    for i, idx in enumerate(sample_successes, 1):
        row = df.loc[idx]
        context_words = extract_context(row)
        context_str = ' '.join(context_words[:15])
        print(f"{i}. [{row['abbreviation']}] {row['label']}")
        print(f"   {context_str}...")

def full_analysis(df, idx_test, y_test, y_pred, model, vocabulary):
    """Run complete analysis"""
    classes = np.unique(y_test)
    
    per_abbreviation_analysis(df, idx_test, y_test, y_pred)
    confusion_analysis(y_test, y_pred, classes)
    feature_importance(model, vocabulary)
    analyze_failures(df, idx_test, y_test, y_pred)
    analyze_successes(df, idx_test, y_test, y_pred)