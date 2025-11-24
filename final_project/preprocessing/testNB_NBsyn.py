"""
Test Naive Bayes on NB-generated synthetic data (UNIGRAMS ONLY).

"""

import sys

sys.path.append("bayes_evaluation")

from feature_extraction import process_data
from models import MultinomialNB
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from collections import Counter


def test_nb_unigrams_only(dataset_path, dataset_name, min_freq=1):
    """Test NB on a dataset using UNIGRAMS ONLY."""
    print(f"\n{'='*70}")
    print(f"Testing Naive Bayes on {dataset_name}")
    print(f"{'='*70}")

    # load data
    df = pd.read_csv(dataset_path)
    print(f"Loaded {len(df)} examples")
    print(f"Using UNIGRAMS ONLY (no bigrams/trigrams)")

    # unigram vocabulary
    word_counts = Counter()
    for _, row in df.iterrows():
        text_words = str(row["text"]).lower().split()
        loc = int(row["location"])
        start = max(0, loc - 5)
        end = min(len(text_words), loc + 5 + 1)
        context_words = text_words[start:loc] + text_words[loc + 1 : end]
        for word in context_words:
            word_counts[word] += 1

    vocab = {
        word: idx
        for idx, (word, count) in enumerate(word_counts.items())
        if count >= min_freq
    }

    #  unigrams only
    contexts = []
    labels = []
    for _, row in df.iterrows():
        text_words = str(row["text"]).lower().split()
        loc = int(row["location"])
        start = max(0, loc - 5)
        end = min(len(text_words), loc + 5 + 1)
        context_words = text_words[start:loc] + text_words[loc + 1 : end]
        word_counts_ex = Counter(context_words)
        contexts.append(word_counts_ex)
        labels.append(row["label"])

    X = np.zeros((len(contexts), len(vocab)), dtype=np.float32)
    for i, word_counts_ex in enumerate(contexts):
        for word, count in word_counts_ex.items():
            if word in vocab:
                X[i, vocab[word]] = count
    y = np.array(labels)

    print(f"Vocabulary: {len(vocab)} unigrams")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Train NB
    nb = MultinomialNB(alpha=1.0)
    nb.fit(X_train, y_train)

    # Predict
    y_pred_train = nb.predict(X_train)
    y_pred_test = nb.predict(X_test)

    train_acc = (y_pred_train == y_train).mean()
    test_acc = (y_pred_test == y_test).mean()

    print(f"\nTrain Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"Test Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(
        f"Train-Test Gap: {(train_acc - test_acc):.4f} ({(train_acc - test_acc)*100:.2f}%)"
    )

    return train_acc, test_acc


def main():
    """Test NB on NB-generated synthetic data."""

    print("=" * 70)
    print("NAIVE BAYES ON NB-GENERATED SYNTHETIC (UNIGRAMS ONLY)")
    print("=" * 70)

    # test NB-generated synthetic
    try:
        nb_syn_train, nb_syn_test = test_nb_unigrams_only(
            "./data/nb_synthetic_dataset.csv", "NB-Generated Synthetic", min_freq=1
        )

        print("\n" + "=" * 70)
        print("EXPECTED RESULTS")
        print("=" * 70)

        if nb_syn_test > 0.75:
            print(
                f"\n SUCCESS! NB achieves {nb_syn_test*100:.2f}% on NB-generated data"
            )
            print("   This is much better than the previous 45%")
            print("   The sparsity issue is FIXED!")
        elif nb_syn_test > 0.60:
            print(f"\n  IMPROVED but not great: {nb_syn_test*100:.2f}%")
            print("   May need even more data")
        else:
            print(f"\n STILL BROKEN: {nb_syn_test*100:.2f}%")
            print("   Something is still wrong")

        gap = nb_syn_train - nb_syn_test
        if gap < 0.15:
            print(f"\n Train-test gap is reasonable: {gap*100:.2f}%")
        else:
            print(f"\n  Train-test gap still high: {gap*100:.2f}%")
            print("   May need more data")

    except FileNotFoundError:
        print("nb_synthetic_dataset.csv not found!")
        print("Please run: python preprocessing/generate_NB_synthetic_unigrams.py")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
