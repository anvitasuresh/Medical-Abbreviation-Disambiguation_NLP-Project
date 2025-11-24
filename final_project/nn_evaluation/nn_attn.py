"""
Attention-Weighted Neural Network with BioWordVec
for Medical Abbreviation Disambiguation

"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from collections import Counter
import re
import os


# ------------------------------
# Data Loading and Preprocessing


def extract_context(row, window_size=5):
    text_words = row["text"].lower().split()
    loc = int(row["location"])
    start = max(0, loc - window_size)
    end = min(len(text_words), loc + window_size + 1)
    context = text_words[start:loc] + text_words[loc + 1 : end]
    return context


def preprocess_word(word):
    word = re.sub(r"[^a-zA-Z]", "", word)
    return word.lower() if len(word) > 1 else ""


def preprocess_context(words):
    return [preprocess_word(w) for w in words if preprocess_word(w)]


# ------------------
# BioWordVec Loader


class BioWordVecEmbeddings:
    """
    class for loading and manages pre-trained BioWordVec embeddings

    difference from mean pooling version:
    - get_context_matrix() preserves individual word vectors so words weighted separately

    """

    def __init__(self):
        self.word_to_idx = {}  # word → index mapping
        self.embeddings = None  # All word vectors
        self.embedding_dim = None  # 200 dimensions

    def load_from_binary(self, filepath, max_vocab=None):

        from gensim.models import KeyedVectors

        model = KeyedVectors.load_word2vec_format(
            filepath, binary=True, limit=max_vocab
        )

        # create mapping from word to index
        # e.g., {"patient": 0, "sclerosis": 1, ...}
        self.word_to_idx = {word: idx for idx, word in enumerate(model.index_to_key)}

        # get the actual embedding vectors as numpy array
        # shape: (vocab_size, embedding_dim)
        self.embeddings = model.vectors

        # store embedding dimensionality (should be 200 for BioWordVec)
        self.embedding_dim = model.vector_size

        print(
            f"  Loaded {len(self.word_to_idx)} words, {self.embedding_dim} dimensions"
        )
        return self

    def get_context_matrix(self, words, max_len=10):
        """
        conver context words to a matrix of embeddings (not averaged)


        attention mechanism needs to evaluate each word individually
        and then it computes importance weights for each word.
        then it does weighted sum (like mean, but with learned weights)

        """
        word_vectors = []

        # get embedding for each word
        for word in words[:max_len]:
            if word in self.word_to_idx:
                # word exists in vocabulary - use its embedding
                word_vectors.append(self.embeddings[self.word_to_idx[word]])
            else:
                # use zero vector if word does not exist
                word_vectors.append(np.zeros(self.embedding_dim))

        # track how many real words we have (before padding)
        actual_len = len(word_vectors)

        # using a mask: 1 for real words, 0 for padding (ignore pads)
        mask = np.zeros(max_len)
        mask[:actual_len] = 1

        # pad with zeros if needed to reach max_len
        while len(word_vectors) < max_len:
            word_vectors.append(np.zeros(self.embedding_dim))

        return np.array(word_vectors), mask


# --------------------------------
# Attention-Weighted Neural Network


class AttentionWeightedNN:
    """
    neural network with attention mechanism for context pooling

    Architecture:
        Context words (up to 10 words)
           [convert to embeddings]
        Context matrix (10 * 200)
           [attention - learns importance of each word]
        Attention weights (10 values that sum to 1)
           [weighted sum using attention weights]
        Pooled vector (200 dims)
           [feed-forward network]
        Hidden layer (256 neurons + ReLU)
        Output (9 classes + Softmax)

    """

    def __init__(self, embedding_dim, hidden_dim, output_dim, learning_rate=0.1):
        """
        initialize neural network with attention mechanism.

        arguments:
            embedding_dim: size of word embeddings (200)
            hidden_dim: number of neurons in hidden layer (256)
            output_dim: number of output classes (9)
            learning_rate: step size for gradient descent
        """
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lr = learning_rate
        self.initial_lr = learning_rate

        # att. parameters
        # learn to compute importance scores for each word
        # W_attn maps each word embedding (200 dims) to a single score
        # formula: score = embedding @ W_attn + b_attn
        self.W_attn = np.random.randn(embedding_dim, 1) * 0.01
        self.b_attn = np.zeros(1)

        # feed forward layers (after attention pooling, just one layer)
        # keeps model capacity similar despite added attention complexity

        # hiddeen layer
        self.W1 = np.random.randn(embedding_dim, hidden_dim) * np.sqrt(
            2.0 / embedding_dim
        )
        self.b1 = np.zeros(hidden_dim)

        # output layer
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(output_dim)

        self.classes = None

    def compute_attention_weights(self, context_matrix, mask):
        """
        compute attention weights for each word in context

        1. compute an importance score for each word
        2. mask out padding positions (set score to very negative)
        3. apply softmax to convert scores to probabilities (weights sum to 1)


        """
        # STEP 1: compute attention scores
        # matrix mult: (batch, 10, 200) @ (200, 1) = (batch, 10, 1)
        scores = (context_matrix @ self.W_attn).squeeze(-1) + self.b_attn

        # STEP 2: mask padding positions so it does not affect sum
        scores = np.where(mask == 1, scores, -1e9)

        # STEP 3: convert scores to probabilities using softmax

        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        attention_weights = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        return attention_weights

    def attention_pooling(self, context_matrix, mask):
        """
        pool context words using learned attention weights

        like mean pooling, but with learned weights instead of equal pnes

        returns:
            pooled: (batch_size, embedding_dim) - weighted sum of words
            attention_weights: (batch_size, max_len) - for analysis
        """
        # compute attention weights for each word
        attention_weights = self.compute_attention_weights(context_matrix, mask)

        # weighted sum: multiply each word vector by its attention weight
        # attention_weights shape: (batch, max_len)
        # need to add dimension: (batch, max_len, 1)
        # broadcast multiply with context_matrix: (batch, max_len, 200)
        weighted_context = context_matrix * attention_weights[:, :, np.newaxis]

        # sum across words to get single vector per sample
        pooled = np.sum(weighted_context, axis=1)

        return pooled, attention_weights

    def relu(self, x):
        """
        ReLU activation function
        formula: relu(x) = max(0, x)
        """
        return np.maximum(0, x)

    def relu_derivative(self, x):
        """
        ReLU derivative for backpropagation
        """
        return (x > 0).astype(float)

    def softmax(self, x):
        """
        softmax activation for output layer.
        """
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, context_matrix, mask):
        """
        forward pass: make predictions using attention mechanism.

        darchitecture:
            context matrix (batch * 10 * 200)  - Individual word embeddings
               [compute attention weights]
            Attention weights (batch * 10)  ← importance of each word
               [weighted sum]
            Pooled vector (batch * 200)  ← context representation
               [multiply by W1, add b1]
            z1 (batch * 256)  ← pre-activation
               [apply ReLU]
            a1 (batch * 256)  ← hidden layer
               [multiply by W2, add b2]
            z2 (batch * 9)  ← logits
               [apply softmax]
            a2 (batch * 9)  ← probabilities

        """
        # attn pooling
        self.pooled, self.attention_weights = self.attention_pooling(
            context_matrix, mask
        )

        # feed forward part
        # same as mean pooling from here on
        self.z1 = self.pooled @ self.W1 + self.b1
        self.a1 = self.relu(self.z1)

        # output layer with softmax
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.softmax(self.z2)

        return self.a2

    def backward(self, context_matrix, mask, y_onehot):
        """
        backward pass: compute gradients for all weights

        do not backpropagate through attention

        arguments:
            context_matrix: input embeddings
            mask: padding mask
            y_onehot: true labels

        return:
            gradients for feed-forward weights
        """
        m = context_matrix.shape[0]  # batch size

        # OUTPUT LAYER GRADIENTS
        dz2 = self.a2 - y_onehot
        dW2 = (self.a1.T @ dz2) / m
        db2 = np.mean(dz2, axis=0)

        # HIDDEN LAYER GRADIENTS
        da1 = dz2 @ self.W2.T
        dz1 = da1 * self.relu_derivative(self.z1)
        dW1 = (self.pooled.T @ dz1) / m
        db1 = np.mean(dz1, axis=0)

        return dW1, db1, dW2, db2

    def compute_loss(self, y_pred, y_onehot):
        """
        compute cross-entrop loss; how different the predicted probabilities are from the true labels
        """
        y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
        return -np.mean(np.sum(y_onehot * np.log(y_pred), axis=1))

    def fit(
        self,
        X_train_ctx,
        mask_train,
        y_train,
        X_val_ctx=None,
        mask_val=None,
        y_val=None,
        epochs=100,
        batch_size=64,
        lr_decay=0.98,
        verbose=True,
    ):
        """
        train the attention neural network.

        similar to mean pooling training, but with key differences:
        - input is context matrices not vectors
        - pass mask along with data
        - attention weights are computed during forward pass

        """
        # convert labels to one-hot encoding
        self.classes = np.unique(y_train)
        class_to_idx = {c: i for i, c in enumerate(self.classes)}
        y_indices = np.array([class_to_idx[label] for label in y_train])

        n_samples = len(y_train)
        y_onehot = np.zeros((n_samples, self.output_dim))
        y_onehot[np.arange(n_samples), y_indices] = 1

        n_batches = int(np.ceil(n_samples / batch_size))

        best_val_acc = 0
        best_weights = None

        # TRAINING LOOP
        for epoch in range(epochs):
            # learning rate decay
            self.lr = self.initial_lr * (lr_decay**epoch)

            # shuffle
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train_ctx[indices]
            mask_shuffled = mask_train[indices]
            y_shuffled = y_onehot[indices]

            total_loss = 0

            # MINI-BATCH TRAINING
            for batch in range(n_batches):
                start = batch * batch_size
                end = min(start + batch_size, n_samples)

                # batch data with masks
                X_batch = X_shuffled[start:end]
                mask_batch = mask_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                # forward pass using attention
                y_pred = self.forward(X_batch, mask_batch)

                # compute loss
                loss = self.compute_loss(y_pred, y_batch)
                total_loss += loss

                # backward pass
                dW1, db1, dW2, db2 = self.backward(X_batch, mask_batch, y_batch)

                # update weights (gradient descent)
                self.W1 -= self.lr * dW1
                self.b1 -= self.lr * db1
                self.W2 -= self.lr * dW2
                self.b2 -= self.lr * db2

            # VALIDATION; same as mean pooling also
            if verbose and (epoch + 1) % 10 == 0:
                train_acc = self.score(X_train_ctx, mask_train, y_train)
                avg_loss = total_loss / n_batches

                if X_val_ctx is not None and y_val is not None:
                    val_acc = self.score(X_val_ctx, mask_val, y_val)
                    print(
                        f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, "
                        f"Train: {train_acc:.4f}, Val: {val_acc:.4f}, LR: {self.lr:.6f}"
                    )

                    # save best weights (early stopping)
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        # include attention parameters in saved weights
                        best_weights = (
                            self.W1.copy(),
                            self.b1.copy(),
                            self.W2.copy(),
                            self.b2.copy(),
                            self.W_attn.copy(),
                            self.b_attn.copy(),
                        )
                else:
                    print(
                        f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, "
                        f"Train: {train_acc:.4f}, LR: {self.lr:.6f}"
                    )

        # restore best weights
        if best_weights is not None:
            self.W1, self.b1, self.W2, self.b2, self.W_attn, self.b_attn = best_weights
            print(
                f"\nRestored best weights with validation accuracy: {best_val_acc:.4f}"
            )

        return self

    def predict(self, context_matrix, mask):
        """
        make predictions for class labels

        """
        probs = self.forward(context_matrix, mask)
        return self.classes[np.argmax(probs, axis=1)]

    def score(self, context_matrix, mask, y):
        """
        compute accuracy

        """
        return np.mean(self.predict(context_matrix, mask) == y)


# -------------------
# Analysis Functions


def print_confusion_analysis(y_test, y_pred, classes):
    """
    print confusion matrix and top confusion pairs

    """
    print("\n" + "=" * 60)
    print("CONFUSION MATRIX ANALYSIS")
    print("=" * 60)

    cm = confusion_matrix(y_test, y_pred, labels=classes)

    print("\nConfusion Matrix:")
    print("(Rows = True, Columns = Predicted)\n")

    print("True \\ Pred".ljust(20), end="")
    for c in classes:
        print(c[:8].ljust(10), end="")
    print()

    for i, true_class in enumerate(classes):
        print(true_class[:18].ljust(20), end="")
        for j in range(len(classes)):
            print(str(cm[i, j]).ljust(10), end="")
        print()

    print("\n" + "=" * 60)
    print("TOP 10 CONFUSION PAIRS")
    print("=" * 60)
    confusions = []
    for i, true_class in enumerate(classes):
        for j, pred_class in enumerate(classes):
            if i != j and cm[i, j] > 0:
                confusions.append((true_class, pred_class, cm[i, j]))

    confusions.sort(key=lambda x: x[2], reverse=True)

    for idx, (true_c, pred_c, count) in enumerate(confusions[:10], 1):
        print(f"{idx:2d}. {true_c:20s} → {pred_c:20s}: {count:4d}")


def print_per_abbreviation_accuracy(y_test, y_pred):
    """
    calculate and print per-abbreviation accuracy
    """
    print("\n" + "=" * 60)
    print("PER-ABBREVIATION ACCURACY")
    print("=" * 60)

    class_to_abbrev = {
        "cell culture": "CC",
        "colorectal cancer": "CC",
        "cervical cancer": "CC",
        "chronic pain": "CP",
        "chest pain": "CP",
        "cerebral palsy": "CP",
        "surface area": "SA",
        "sleep apnea": "SA",
        "substance abuse": "SA",
    }

    results = []
    for abbrev in ["CC", "CP", "SA"]:
        abbrev_classes = [c for c, a in class_to_abbrev.items() if a == abbrev]
        abbrev_mask = np.isin(y_test, abbrev_classes)

        if np.sum(abbrev_mask) > 0:
            abbrev_acc = np.mean(y_pred[abbrev_mask] == y_test[abbrev_mask])
            n_samples = np.sum(abbrev_mask)
            results.append((abbrev, abbrev_acc, n_samples))
            print(f"{abbrev}: {abbrev_acc:.1%} (n={n_samples})")

    return results


def analyze_attention_weights(
    model, X_test_ctx, mask_test, contexts_test, y_test, y_pred
):
    """
    analyze attention weights to see which words get high attention; what did mechanism learn?

    """
    print("\n" + "=" * 60)
    print("ATTENTION WEIGHT ANALYSIS")
    print("=" * 60)

    # getting weights
    model.forward(X_test_ctx, mask_test)
    attn_weights = model.attention_weights

    # examples with highest attention variance (most discriminative)
    attn_variance = np.var(attn_weights, axis=1)
    top_indices = np.argsort(attn_variance)[-5:]

    print("\nExamples where attention is most discriminative:")
    print("(High variance = model focuses on specific keywords)\n")

    for i, idx in enumerate(top_indices[::-1], 1):
        context_words = contexts_test[idx]
        weights = attn_weights[idx]
        true_label = y_test[idx]
        pred_label = y_pred[idx]

        print(f"\nExample {i}:")
        print(f"  True: {true_label}")
        print(f"  Predicted: {pred_label}")
        print(f"  Context words with attention weights:")

        for word, weight, mask_val in zip(context_words, weights, mask_test[idx]):
            if mask_val > 0 and word:  # only show real words
                print(f"    {word:20s} {weight:.3f}")


# --------------
# Main Pipeline


def run_experiment(dataset_path, dataset_name, embedder):
    """
    run everything again :)
    """

    print("\n" + "=" * 70)
    print(f"EXPERIMENT: {dataset_name}")
    print("=" * 70)

    # load data
    print(f"\n1. Loading {dataset_name}...")
    df = pd.read_csv(dataset_path)
    print(f"Loaded {len(df)} samples")

    # contexts
    print("\n2. Extracting contexts...")
    contexts = []
    labels = []

    for _, row in df.iterrows():
        context_words = extract_context(row, window_size=5)
        context_words = preprocess_context(context_words)
        contexts.append(context_words)
        labels.append(row["label"])

    labels = np.array(labels)
    print(f"Extracted {len(contexts)} contexts")

    # split data
    print("\n3. Splitting data...")
    contexts_train, contexts_test, y_train, y_test = train_test_split(
        contexts, labels, test_size=0.3, random_state=42, stratify=labels
    )
    print(f"Train: {len(contexts_train)}, Test: {len(contexts_test)}")

    # context matrices
    print("\n4. Creating context matrices...")
    MAX_SEQ_LEN = 10

    X_train_ctx = []
    mask_train = []
    for ctx in contexts_train:
        ctx_matrix, mask = embedder.get_context_matrix(ctx, max_len=MAX_SEQ_LEN)
        X_train_ctx.append(ctx_matrix)
        mask_train.append(mask)

    X_test_ctx = []
    mask_test = []
    for ctx in contexts_test:
        ctx_matrix, mask = embedder.get_context_matrix(ctx, max_len=MAX_SEQ_LEN)
        X_test_ctx.append(ctx_matrix)
        mask_test.append(mask)

    X_train_ctx = np.array(X_train_ctx)
    mask_train = np.array(mask_train)
    X_test_ctx = np.array(X_test_ctx)
    mask_test = np.array(mask_test)

    print(f"X_train_ctx shape: {X_train_ctx.shape}")

    # train model
    print("\n5. Training Attention-Weighted Neural Network...")

    num_classes = len(np.unique(y_train))

    model = AttentionWeightedNN(
        embedding_dim=embedder.embedding_dim,
        hidden_dim=256,
        output_dim=num_classes,
        learning_rate=0.1,
    )

    model.fit(
        X_train_ctx,
        mask_train,
        y_train,
        X_val_ctx=X_test_ctx,
        mask_val=mask_test,
        y_val=y_test,
        epochs=100,
        batch_size=64,
        lr_decay=0.98,
        verbose=True,
    )

    # evaluate
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    train_acc = model.score(X_train_ctx, mask_train, y_train)
    test_acc = model.score(X_test_ctx, mask_test, y_test)

    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test accuracy:  {test_acc:.4f}")

    # per-class performance
    print("\n" + "=" * 60)
    print("PER-CLASS PERFORMANCE")
    print("=" * 60)

    y_pred = model.predict(X_test_ctx, mask_test)

    for c in model.classes:
        tp = np.sum((y_pred == c) & (y_test == c))
        fp = np.sum((y_pred == c) & (y_test != c))
        fn = np.sum((y_pred != c) & (y_test == c))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        print(f"{c}:")
        print(f"  Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    # confusion matrix
    print_confusion_analysis(y_test, y_pred, model.classes)

    print_per_abbreviation_accuracy(y_test, y_pred)

    # attention weight analysis
    analyze_attention_weights(
        model, X_test_ctx, mask_test, contexts_test, y_test, y_pred
    )

    return test_acc, model


def main():
    """
    main function to run everything on synthetic and real

    """
    print("=" * 70)
    print("ATTENTION NEURAL NETWORK - COMPREHENSIVE EVALUATION")
    print("=" * 70)

    BIOWORDVEC_PATH = "bio_embedding_extrinsic"
    MAX_VOCAB = 500000

    if not os.path.exists(BIOWORDVEC_PATH):
        print(f"ERROR: BioWordVec not found at {BIOWORDVEC_PATH}")
        return

    # check if filtered dataset exists, if not, run filter script in preprocessing folder
    if not os.path.exists("data/filtered_dataset.csv"):
        print("\n  filtered_dataset.csv not found")
        print("Running preprocessing/filter_data.py to generate it...")

        import subprocess

        subprocess.run(["python", "preprocessing/filter_data.py"], check=True)

        if not os.path.exists("data/filtered_dataset.csv"):
            print("ERROR: Failed to generate filtered_dataset.csv")
            return

        print(" filtered_dataset.csv generated successfully\n")

    print("\nLoading BioWordVec embeddings...")
    embedder = BioWordVecEmbeddings()
    embedder.load_from_binary(BIOWORDVEC_PATH, max_vocab=MAX_VOCAB)

    # run experiments
    results = {}

    # experiment 1: synthetic data
    if os.path.exists("data/synthetic_dataset.csv"):
        synthetic_acc, _ = run_experiment(
            "data/synthetic_dataset.csv", "SYNTHETIC DATA", embedder
        )
        results["synthetic"] = synthetic_acc
    else:
        print("\n  Synthetic dataset not found at data/synthetic_dataset.csv")
        print("Skipping synthetic data experiment...")

    # experiment 2: NB-generated synthetic data

    if os.path.exists("data/nb_synthetic_dataset.csv"):
        nb_synthetic_acc, _ = run_experiment(
            "data/nb_synthetic_dataset.csv", "NB-GENERATED SYNTHETIC", embedder
        )
        results["nb_synthetic"] = nb_synthetic_acc
    else:
        print("\n  nb_synthetic_dataset.csv not found!")
        print("Please run: python preprocessing/generate_NB_synthetic.py")

    # experiment 3: real data
    if os.path.exists("data/filtered_dataset.csv"):
        real_acc, _ = run_experiment(
            "data/filtered_dataset.csv", "REAL DATA (FILTERED)", embedder
        )
        results["real"] = real_acc
    else:
        print("\n  Filtered dataset not found at data/filtered_dataset.csv")

    # final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY - ATTENTION NN")
    print("=" * 70)

    print("\n{:<30s} {:>12s}".format("Dataset", "Accuracy"))
    print("-" * 44)

    if "synthetic" in results:
        print(
            "{:<30s} {:>11.2f}%".format(
                "Template Synthetic", results["synthetic"] * 100
            )
        )

    if "nb_synthetic" in results:
        print(
            "{:<30s} {:>11.2f}%".format(
                "NB-Generated Synthetic", results["nb_synthetic"] * 100
            )
        )

    if "real" in results:
        print("{:<30s} {:>11.2f}%".format("Real Data", results["real"] * 100))

    # analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    if "synthetic" in results and "real" in results:
        gap = results["synthetic"] - results["real"]
        print(f"\nTemplate Synthetic → Real Data Gap: {gap:.4f} ({gap*100:.2f}%)")

    if "nb_synthetic" in results:
        if "synthetic" in results:
            nb_vs_template = results["nb_synthetic"] - results["synthetic"]
            print(
                f"\nNB-Generated → Template Gap: {nb_vs_template:.4f} ({nb_vs_template*100:.2f}%)"
            )

        if "real" in results:
            nb_vs_real = results["nb_synthetic"] - results["real"]
            print(
                f"\nNB-Generated → Real Data Gap: {nb_vs_real:.4f} ({nb_vs_real*100:.2f}%)"
            )


if __name__ == "__main__":
    main()
