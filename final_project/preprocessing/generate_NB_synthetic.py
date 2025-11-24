"""
Generate NB-sampled synthetic data using UNIGRAMS ONLY.

"""

import pandas as pd
import numpy as np
from collections import Counter


def extract_unigram_probabilities(df, min_freq=3):
    """
    extract P(w|class) for UNIGRAMS ONLY from real data.

    returning:
        word_probs: Dict[class -> Dict[word -> probability]]
    """
    # count words per class
    class_word_counts = {}

    for _, row in df.iterrows():
        label = row["label"]
        text_words = str(row["text"]).lower().split()
        loc = int(row["location"])

        # extract 5 word context
        start = max(0, loc - 5)
        end = min(len(text_words), loc + 5 + 1)
        context_words = text_words[start:loc] + text_words[loc + 1 : end]

        if label not in class_word_counts:
            class_word_counts[label] = Counter()

        # count only single words (unigrams)
        for word in context_words:
            word = "".join(c for c in word if c.isalnum() or c == "-")
            if len(word) > 1:
                class_word_counts[label][word] += 1

    # build vocabulary (words appearing >= min_freq times)
    global_vocab = Counter()
    for label, counts in class_word_counts.items():
        global_vocab.update(counts)

    vocab = {word for word, count in global_vocab.items() if count >= min_freq}

    print(f"\n  Built vocabulary: {len(vocab)} unigrams (min_freq={min_freq})")

    # compute P(w|class) with Laplace smoothing
    word_probs = {}
    alpha = 1.0  # smooth

    for label in class_word_counts:
        word_probs[label] = {}

        # word counts for this class
        counts = class_word_counts[label]

        # total count + smoothing
        total = sum(counts[w] for w in vocab) + alpha * len(vocab)

        # compute probabilities
        for word in vocab:
            count = counts[word] if word in counts else 0
            word_probs[label][word] = (count + alpha) / total

    return word_probs, vocab


def generate_synthetic_example(label, word_probs, abbreviation, n_words):
    """Generate one example by sampling unigrams from P(w|class)."""

    words = list(word_probs.keys())
    probs = np.array([word_probs[w] for w in words])
    probs = probs / probs.sum()

    # sample words
    sampled_words = np.random.choice(words, size=n_words, p=probs, replace=True)

    # insert abbreviation at random position
    abbrev_pos = np.random.randint(0, len(sampled_words) + 1)
    all_words = (
        list(sampled_words[:abbrev_pos])
        + [abbreviation]
        + list(sampled_words[abbrev_pos:])
    )

    return " ".join(all_words), abbrev_pos


def generate_dataset(word_probs, examples_per_class=200):
    """Generate complete synthetic dataset."""

    label_to_abbrev = {
        "colorectal cancer": "CC",
        "cell culture": "CC",
        "cervical cancer": "CC",
        "chronic pain": "CP",
        "chest pain": "CP",
        "cerebral palsy": "CP",
        "surface area": "SA",
        "sleep apnea": "SA",
        "substance abuse": "SA",
    }

    synthetic_data = []

    for label in sorted(word_probs.keys()):
        abbreviation = label_to_abbrev[label]
        print(f"  Generating {examples_per_class} examples for {label}...")

        for _ in range(examples_per_class):
            # sample 7-10 words (matching real data length)
            n_words = np.random.randint(7, 11)
            text, location = generate_synthetic_example(
                label, word_probs[label], abbreviation, n_words
            )

            synthetic_data.append(
                {
                    "abbreviation": abbreviation,
                    "text": text,
                    "location": location,
                    "label": label,
                }
            )

    return pd.DataFrame(synthetic_data)


def main():
    """Generate NB synthetic data with unigrams only."""

    print("=" * 70)
    print("NB-SYNTHETIC GENERATION (UNIGRAMS ONLY - 100K EXAMPLES)")
    print("=" * 70)
    print("\nGenerating ~100,000 examples to maximize feature overlap")

    EXAMPLES_PER_CLASS = 11111  # 11111 * 9 = ~100,000 total
    MIN_FREQ = 3
    RANDOM_SEED = 42

    np.random.seed(RANDOM_SEED)

    # load real data
    print("\n" + "=" * 70)
    print("STEP 1: Loading Real Data")
    print("=" * 70)

    df_real = pd.read_csv("./data/filtered_dataset.csv")
    print(f"✓ Loaded {len(df_real)} examples")

    # extract P(w|class) for unigrams only
    print("\n" + "=" * 70)
    print("STEP 2: Extracting Unigram Probabilities")
    print("=" * 70)

    word_probs, vocab = extract_unigram_probabilities(df_real, MIN_FREQ)

    print(f"\n✓ Extracted P(w|class) for {len(word_probs)} classes")
    print(f"✓ Vocabulary: {len(vocab)} unigrams")

    # show top words per class
    print("\n" + "=" * 70)
    print("TOP 10 UNIGRAMS PER CLASS")
    print("=" * 70)

    for label in sorted(word_probs.keys()):
        top = sorted(word_probs[label].items(), key=lambda x: x[1], reverse=True)[:10]
        features = ", ".join([f"{w}({p:.4f})" for w, p in top])
        print(f"\n{label}:")
        print(f"  {features}")

    # generate synthetic data
    print("\n" + "=" * 70)
    print("STEP 3: Generating Synthetic Data")
    print("=" * 70)
    print(f"  Generating {EXAMPLES_PER_CLASS} examples per class...")
    print(f"  Context size: 7-10 words (matching real data)")

    df_synthetic = generate_dataset(word_probs, EXAMPLES_PER_CLASS)

    print(f"\n✓ Generated {len(df_synthetic)} examples")

    output_path = "./data/nb_synthetic_dataset.csv"
    df_synthetic.to_csv(output_path, index=False)

    print("\n" + "=" * 70)
    print("GENERATION COMPLETE")
    print("=" * 70)
    print(f"\n✓ Saved to {output_path}")

    # show samples
    print("\n" + "=" * 70)
    print("SAMPLE EXAMPLES")
    print("=" * 70)

    for i, (idx, row) in enumerate(
        df_synthetic.sample(5, random_state=42).iterrows(), 1
    ):
        print(f"\n{i}. [{row['abbreviation']}] {row['label']}")
        print(f"   {row['text']}")


if __name__ == "__main__":
    main()
