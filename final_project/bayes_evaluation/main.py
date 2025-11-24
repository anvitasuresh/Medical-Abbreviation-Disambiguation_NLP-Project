import numpy as np
from sklearn.model_selection import train_test_split
from data_loader import load_synthetic, load_real
from feature_extraction import process_data
from models import MostFrequentBaseline, MultinomialNB
from tfidf import compute_tfidf
from evaluation import calculate_metrics, confusion_matrix
from analysis import full_analysis


def run_experiment(dataset_name, df, min_freq=1):
    print(f"\n{dataset_name} Dataset")
    print(f"Samples: {len(df)}")

    X, y, vocab = process_data(df, min_freq=min_freq)
    print(f"Features: {X.shape[1]}")

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, np.arange(len(X)), test_size=0.3, random_state=42, stratify=y
    )

    baseline = MostFrequentBaseline()
    baseline.fit(y_train)
    baseline_pred = baseline.predict(X_test)
    baseline_acc = (baseline_pred == y_test).mean()

    nb_raw = MultinomialNB()
    nb_raw.fit(X_train, y_train)
    nb_pred_raw = nb_raw.predict(X_test)
    nb_acc_raw = (nb_pred_raw == y_test).mean()

    X_train_tfidf = compute_tfidf(X_train)
    X_test_tfidf = compute_tfidf(X_test)

    nb_tfidf = MultinomialNB()
    nb_tfidf.fit(X_train_tfidf, y_train)
    nb_pred_tfidf = nb_tfidf.predict(X_test_tfidf)
    nb_acc_tfidf = (nb_pred_tfidf == y_test).mean()

    print(f"Baseline: {baseline_acc:.4f}")
    print(f"NB (raw): {nb_acc_raw:.4f}")
    print(f"NB (TF-IDF): {nb_acc_tfidf:.4f}")

    classes = np.unique(y_train)
    metrics_raw = calculate_metrics(y_test, nb_pred_raw, classes)
    cm_raw = confusion_matrix(y_test, nb_pred_raw, classes)

    return {
        "baseline_acc": baseline_acc,
        "nb_raw_acc": nb_acc_raw,
        "nb_tfidf_acc": nb_acc_tfidf,
        "metrics": metrics_raw,
        "confusion_matrix": cm_raw,
        "predictions": nb_pred_raw,
        "y_test": y_test,
        "idx_test": idx_test,
        "model": nb_raw,
        "vocabulary": vocab,
    }


if __name__ == "__main__":
    synthetic_df = load_synthetic()
    synthetic_results = run_experiment("Synthetic", synthetic_df)

    real_df = load_real()
    real_results = run_experiment("Real", real_df, min_freq=3)

    print("DETAILED ANALYSIS - REAL DATA")
    full_analysis(
        real_df,
        real_results["idx_test"],
        real_results["y_test"],
        real_results["predictions"],
        real_results["model"],
        real_results["vocabulary"],
    )
