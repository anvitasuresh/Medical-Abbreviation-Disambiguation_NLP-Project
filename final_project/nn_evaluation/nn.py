"""
Mean Pooling Neural Network with BioWordVec
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
    """
    get context words around the target abbreviation.

    example:
    "patient has multiple sclerosis symptoms"
    and "sclerosis" is at position 3, with window_size=2:
    - left: ["patient", "has"]
    - right: ["symptoms"]
    - need to exclude the target word itself

    arguments:
        row: DataFrame row containing 'text' and 'location'
        window_size: Number of words to include on each side

    returns:
        list of context words (excluding the target abbreviation)
    """
    # convert text to lowercase and split into words
    text_words = row["text"].lower().split()

    # get the position of the abbreviation
    loc = int(row["location"])

    # calculate window boundaries (don't go negative or past end)
    start = max(0, loc - window_size)
    end = min(len(text_words), loc + window_size + 1)

    # extract context: words before + words after (skip the abbreviation itself)
    context = text_words[start:loc] + text_words[loc + 1 : end]

    return context


def preprocess_word(word):
    """
    clean a single word by removing non-alphabetic characters.

    examples:
        "patient's" → "patients"
        "3-year" → "year"
        "MS." → "ms"

    arguments:
        word: Single word string

    return:
        clean lowercase word, or empty string if too short
    """
    # remove all non-letter characters (numbers, punctuation, etc.)
    word = re.sub(r"[^a-zA-Z]", "", word)

    # return lowercase version only if word has 2+ characters
    return word.lower() if len(word) > 1 else ""


def preprocess_context(words):
    """
    clean all words in a context list

    argumentss:
        words: List of raw words from text

    returns:
        list of cleaned words (empty strings removed)
    """
    return [preprocess_word(w) for w in words if preprocess_word(w)]


# ----------------
# BioWordVec Loader


class BioWordVecEmbeddings:
    """
    class for loading and manages pre-trained BioWordVec embeddings

    each word is represented as a 200-dimensional vector where
    semantically similar medical terms are close together in space
    (trained on medical literature & hospital records)
    """

    def __init__(self):
        """empty embedding storage."""
        self.word_to_idx = {}
        self.embeddings = None
        self.embedding_dim = None

    def load_from_binary(self, filepath, max_vocab=None):
        """
        load pre-trained BioWordVec embeddings from binary file

        using gensim library to efficiently load Word2Vec format files

        """
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

    def get_mean_embedding(self, words):
        """
        get mean-pooled embedding for a list of words

        mean pooling: averaging all word vectors together, not like attention model

        example:
            words = ["patient", "sclerosis", "symptoms"]
            vectors = [vec1, vec2, vec3]
            result = (vec1 + vec2 + vec3) / 3

         words contribute equally (weight = 1/n for each word).

        arguments:
            words: List of words in context

        returns:
            Single 200-dimensional vector representing the entire context
        """
        vectors = []

        # get embedding for each word (if it exists in vocabulary)
        for w in words:
            if w in self.word_to_idx:
                # look up the word's index and get its embedding vector
                vectors.append(self.embeddings[self.word_to_idx[w]])

        # take mean of all vectors, words are equally weighted
        if vectors:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(self.embedding_dim)


# ---------------------------
# Mean Pooling Neural Network


class MeanPoolingNN:
    """
    two-layer feed-forward neural network with mean pooling

    design:
        input (200 dims from BioWordVec mean pooling)
          ↓
        hiden L1 (512 neurons + ReLU activation)
          ↓
        hidden L2 (256 neurons + ReLU activation)
          ↓
        output layer (9 classes + Softmax)


    1. taking averaged word embeddings as input
    2. learning hierarchical representations through hidden layers
    3. outputting probability distribution over abbreviation meanings
    """

    def __init__(
        self, input_dim, hidden1_dim, hidden2_dim, output_dim, learning_rate=0.1
    ):
        """
        initialize neural network with random weights

        """
        self.input_dim = input_dim
        self.hidden1_dim = hidden1_dim
        self.hidden2_dim = hidden2_dim
        self.output_dim = output_dim
        self.lr = learning_rate
        self.initial_lr = learning_rate

        # formula: weights ~ N(0, sqrt(2/n_in))

        # first hidden layer weights
        # shape: (200, 512) - connects input to first hidden layer
        self.W1 = np.random.randn(input_dim, hidden1_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden1_dim)

        # second hidden layer weights
        # shape: (512, 256) - connects first to second hidden layer
        self.W2 = np.random.randn(hidden1_dim, hidden2_dim) * np.sqrt(2.0 / hidden1_dim)
        self.b2 = np.zeros(hidden2_dim)

        # output layer weights
        # shape: (256, 9) - connects second hidden layer to output
        self.W3 = np.random.randn(hidden2_dim, output_dim) * np.sqrt(2.0 / hidden2_dim)
        self.b3 = np.zeros(output_dim)

        self.classes = None  # this is for storing class labels during training

    def relu(self, x):
        """
        ReLU (Rectified Linear Unit) activation function

        formula: relu(x) = max(0, x)

        using reLu:
        - it introduces non-linearity (allows network to learn complex patterns)
        - computationally efficient (simple max operation)
        - avoid vanishing gradient problem

        """
        return np.maximum(0, x)

    def relu_derivative(self, x):
        """
        derivative of ReLU for backpropagation.

        formula:
            d/dx relu(x) = 1 if x > 0
                         = 0 if x ≤ 0

        this tells us how much to adjust weights during training
        """
        return (x > 0).astype(float)

    def softmax(self, x):
        """
        softmax activation for output layer

        converts raw scores (logits) into probabilities that sum to 1.

        formula: softmax(x_i) = exp(x_i) / sum(exp(x_j) for all j)

        """
        # Subtract max for numerical stability
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))

        # Normalize to get probabilities
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X):
        """
        forward pass: compute predictions from input.

        this is how the network makes predictions

        architecture:
            X (batch_size * 200)  - mean-pooled embeddings
               [multiply by W1, add b1]
            z1 (batch_size * 512)  - pre-activation
               [apply ReLU]
            a1 (batch_size * 512)  - first hidden layer activations
               [multiply by W2, add b2]
            z2 (batch_size * 256)  - pre-activation
               [apply ReLU]
            a2 (batch_size * 256)  - second hidden layer activations
               [multiply by W3, add b3]
            z3 (batch_size * 9)    - logits (raw scores)
               [apply softmax]
            a3 (batch_size * 9)    - probabilities (sum to 1)

        """
        # first hidden layer
        # z = Wx + b (linear transformation)
        self.z1 = X @ self.W1 + self.b1
        # ReLU activation
        self.a1 = self.relu(self.z1)

        # second hidden layer
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.relu(self.z2)

        # output layer
        self.z3 = self.a2 @ self.W3 + self.b3
        # softmax converts to probabilities
        self.a3 = self.softmax(self.z3)

        return self.a3

    def backward(self, X, y_onehot):
        """
        backward pass: compute gradients for all weights

        compute how to adjust each weight to reduce the error (loss)

        we work backwards from output to input and see how much each layer contributes to error

        arguements:
            X: Input batch
            y_onehot: true labels in one-hot format
                     Example: [0, 0, 1, 0, ...] means class 2

        return:
            gradients for all weights (dW1, db1, dW2, db2, dW3, db3)
        """
        m = X.shape[0]  # batch size

        # OUTPUT LAYER GRADIENTS
        # derivative of cross-entropy loss with softmax
        # basically: (predicted - actual)
        dz3 = self.a3 - y_onehot

        # gradient for output weights
        # dL/dW3 = a2^T @ dz3 (chain rule)
        dW3 = (self.a2.T @ dz3) / m

        # gradient for output biases
        db3 = np.mean(dz3, axis=0)

        # SECOND HIDDEN LAYER GRADIENTS
        # propagate error backwards through weights
        da2 = dz3 @ self.W3.T

        # multiply by ReLU derivative (gradient flows through activation)
        dz2 = da2 * self.relu_derivative(self.z2)

        # gradients for second hidden layer weights
        dW2 = (self.a1.T @ dz2) / m
        db2 = np.mean(dz2, axis=0)

        # FIRST HIDDEN LAYER GRADIENTS
        # propagate error backwards
        da1 = dz2 @ self.W2.T

        # multiply by ReLU derivative
        dz1 = da1 * self.relu_derivative(self.z1)

        # gradients for first hidden layer weights
        dW1 = (X.T @ dz1) / m
        db1 = np.mean(dz1, axis=0)

        return dW1, db1, dW2, db2, dW3, db3

    def compute_loss(self, y_pred, y_onehot):
        """
        compute cross-entropy loss; how different the predicted probabilities are from the true labels

        formula: -sum(y_true * log(y_pred))

        arguments:
            y_pred: Predicted probabilities
            y_onehot: True labels (one-hot encoded)

        return:
            average loss across batch
        """
        # clip predictions to avoid log(0) which is undefined
        y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)

        # Compute cross-entropy and average over batch
        return -np.mean(np.sum(y_onehot * np.log(y_pred), axis=1))

    def fit(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        epochs=100,  # number of passes through training data
        batch_size=64,  # number of samples per gradient update
        lr_decay=0.98,  # learning rate decay factor (0.98 = reduce by 2% each epoch)
        verbose=True,
    ):
        """
        train the neural network using mini-batch gradient descent

        training:
        1. labels to one-hot encoding
        2. for each epoch:
           a. shuffle training data (prevents learning order) and split into mini-batches (faster than full batch)
           c. for each batch:
              - forward pass (make predictions)
              - compute loss (how wrong were we?)
              - backward pass (compute gradients)
              - update weights (gradient descent step)
           d. evaluate on validation set
           e. decay learning rate (take smaller steps over time)
        3. restore best weights (early stopping)

        """
        # get unique classes and create mapping
        self.classes = np.unique(y_train)
        class_to_idx = {c: i for i, c in enumerate(self.classes)}

        # convert string labels to indices
        y_indices = np.array([class_to_idx[label] for label in y_train])

        # create one-hot encoded labels
        n_samples = len(y_train)
        y_onehot = np.zeros((n_samples, self.output_dim))
        y_onehot[np.arange(n_samples), y_indices] = 1

        # calc number of batches per epoch
        n_batches = int(np.ceil(n_samples / batch_size))

        # for early stopping: track best validation accuracy
        best_val_acc = 0
        best_weights = None

        # TRAINING LOOP
        for epoch in range(epochs):
            # learning rate decay: reduce LR over time for fine-tuning
            # formula: lr = initial_lr * (decay^epoch)
            self.lr = self.initial_lr * (lr_decay**epoch)

            # shuffle training data (different order each epoch, doesn't learn order)
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_onehot[indices]

            total_loss = 0

            # MINI-BATCH TRAINING
            for batch in range(n_batches):
                # current batch
                start = batch * batch_size
                end = min(start + batch_size, n_samples)

                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                # forward pass: make predictions
                y_pred = self.forward(X_batch)

                # compute loss: how wrong were we?
                loss = self.compute_loss(y_pred, y_batch)
                total_loss += loss

                # backward pass: compute gradients
                dW1, db1, dW2, db2, dW3, db3 = self.backward(X_batch, y_batch)

                # gradient descent: update weights to reduce loss
                # new weight = old weight - learning_rate × gradient
                self.W1 -= self.lr * dW1
                self.b1 -= self.lr * db1
                self.W2 -= self.lr * dW2
                self.b2 -= self.lr * db2
                self.W3 -= self.lr * dW3
                self.b3 -= self.lr * db3

            # VALIDATION
            if verbose and (epoch + 1) % 10 == 0:
                # compute accuracy on training set
                train_acc = self.score(X_train, y_train)
                avg_loss = total_loss / n_batches

                if X_val is not None and y_val is not None:
                    # compute accuracy on validation set
                    val_acc = self.score(X_val, y_val)
                    print(
                        f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, "
                        f"Train: {train_acc:.4f}, Val: {val_acc:.4f}, LR: {self.lr:.6f}"
                    )

                    # do early stop: save best weights, less overfitting
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc

                        best_weights = (
                            self.W1.copy(),
                            self.b1.copy(),
                            self.W2.copy(),
                            self.b2.copy(),
                            self.W3.copy(),
                            self.b3.copy(),
                        )
                else:
                    print(
                        f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, "
                        f"Train: {train_acc:.4f}, LR: {self.lr:.6f}"
                    )

        # restore best weights from validation
        if best_weights is not None:
            self.W1, self.b1, self.W2, self.b2, self.W3, self.b3 = best_weights
            print(
                f"\nRestored best weights with validation accuracy: {best_val_acc:.4f}"
            )

        return self

    def predict(self, X):
        """
        make predictions on new data; predict class labels

        """
        # forward pass to get probabilities
        probs = self.forward(X)

        # return class with highest probability for each sample
        # argmax finds index of maximum value
        return self.classes[np.argmax(probs, axis=1)]

    def score(self, X, y):
        """
        compute accuracy

        accuracy = (number of correct predictions) / (total predictions)
        """
        return np.mean(self.predict(X) == y)


# ------------------
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

    # find top confusion pairs (where model makes most errors)
    print("\n" + "=" * 60)
    print("TOP 10 CONFUSION PAIRS")
    print("=" * 60)
    confusions = []
    for i, true_class in enumerate(classes):
        for j, pred_class in enumerate(classes):
            # count off-diagonal (actual errors)
            if i != j and cm[i, j] > 0:
                confusions.append((true_class, pred_class, cm[i, j]))

    confusions.sort(key=lambda x: x[2], reverse=True)

    # print top 10
    for idx, (true_c, pred_c, count) in enumerate(confusions[:10], 1):
        print(f"{idx:2d}. {true_c:20s} → {pred_c:20s}: {count:4d}")


def print_per_abbreviation_accuracy(y_test, y_pred):
    """
    calculate and print per-abbreviation accuracy to analyze performance separately for each abbreviation

    """
    print("\n" + "=" * 60)
    print("PER-ABBREVIATION ACCURACY")
    print("=" * 60)

    # classes to abbreviations
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
        # get all classes for this abbreviation
        abbrev_classes = [c for c, a in class_to_abbrev.items() if a == abbrev]

        abbrev_mask = np.isin(y_test, abbrev_classes)

        if np.sum(abbrev_mask) > 0:
            # calculate accuracy only on this abbreviation's samples
            abbrev_acc = np.mean(y_pred[abbrev_mask] == y_test[abbrev_mask])
            n_samples = np.sum(abbrev_mask)
            results.append((abbrev, abbrev_acc, n_samples))
            print(f"{abbrev}: {abbrev_acc:.1%} (n={n_samples})")

    return results


def print_error_analysis(y_test, y_pred, contexts_test):
    """
    print sample errors for qualitative analysis

    """
    print("\n" + "=" * 60)
    print("SAMPLE ERRORS (First 5)")
    print("=" * 60)

    # where prediction != truth
    errors = np.where(y_pred != y_test)[0]

    # orint first 5 errors
    for i, idx in enumerate(errors[:5]):
        print(f"\nError {i+1}:")
        print(f"  True: {y_test[idx]}")
        print(f"  Predicted: {y_pred[idx]}")
        print(f"  Context: {' '.join(contexts_test[idx][:10])}")


# -------------
# Main Pipeline


def run_experiment(dataset_path, dataset_name, embedder):
    """
    run everything :)

    steps:
    1. load data
    2. extract contexts
    3. split train/test
    4. create features (mean pooling)
    5. train model
    6. evaluate and analyze

    """

    print("\n" + "=" * 70)
    print(f"EXPERIMENT: {dataset_name}")
    print("=" * 70)

    # STEP 1: load data
    print(f"\n1. Loading {dataset_name}...")
    df = pd.read_csv(dataset_path)
    print(f"Loaded {len(df)} samples")

    # STEP 2: extract contexts
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

    # STEP 3: split data
    print("\n3. Splitting data...")
    contexts_train, contexts_test, y_train, y_test = train_test_split(
        contexts,
        labels,
        test_size=0.3,  # 30% for testing
        random_state=42,  # fixed seed for reproducibility
        stratify=labels,  # keep class distribution same in train/test
    )
    print(f"Train: {len(contexts_train)}, Test: {len(contexts_test)}")

    # STEP 4: create features (mean pooling)
    print("\n4. Creating mean pooling features...")
    # convert each context (list of words) to single 200-dim vector
    X_train = np.array([embedder.get_mean_embedding(ctx) for ctx in contexts_train])
    X_test = np.array([embedder.get_mean_embedding(ctx) for ctx in contexts_test])
    print(f"Shape: {X_train.shape}")

    # STEP 5: train model
    print("\n5. Training Mean Pooling Neural Network...")
    print("   Architecture: 512-256 (two hidden layers)")

    model = MeanPoolingNN(
        input_dim=X_train.shape[1],
        hidden1_dim=512,
        hidden2_dim=256,
        output_dim=len(np.unique(y_train)),  # 9 (number of classes)
        learning_rate=0.1,
    )

    model.fit(
        X_train,
        y_train,
        X_val=X_test,
        y_val=y_test,
        epochs=100,
        batch_size=64,  # update weights after every 64 samples
        lr_decay=0.98,
        verbose=True,
    )

    # STEP 6: evaluate
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)

    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test accuracy:  {test_acc:.4f}")

    # per-class performance
    print("\n" + "=" * 60)
    print("PER-CLASS PERFORMANCE")
    print("=" * 60)

    y_pred = model.predict(X_test)

    for c in model.classes:
        # calculate precision, recall, F1 for each class
        tp = np.sum((y_pred == c) & (y_test == c))  # TP
        fp = np.sum((y_pred == c) & (y_test != c))  # FP
        fn = np.sum((y_pred != c) & (y_test == c))  # FN

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

    # error analysis
    print_error_analysis(y_test, y_pred, contexts_test)

    return test_acc, model


def main():
    """
    main function to run everything on synthetic and real

    """
    print("=" * 70)
    print("MEAN POOLING NEURAL NETWORK - full eval")
    print("=" * 70)

    BIOWORDVEC_PATH = "bio_embedding_extrinsic"
    MAX_VOCAB = 500000

    # does file exist..
    if not os.path.exists(BIOWORDVEC_PATH):
        print(f"ERROR: BioWordVec not found at {BIOWORDVEC_PATH}")
        return

    # check if filtered dataset exists, if not, run filter script
    if not os.path.exists("data/filtered_dataset.csv"):
        print("\n  filtered_dataset.csv not found")
        print("Running preprocessing/filter_data.py to generate it...")

        import subprocess

        subprocess.run(["python", "preprocessing/filter_data.py"], check=True)

        if not os.path.exists("data/filtered_dataset.csv"):
            print("ERROR: Failed to generate filtered_dataset.csv")
            return

        print(" filtered_dataset.csv generated successfully\n")

    # load embeddings once (used for all experiments)
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

    # check if NB synthetic exists
    if os.path.exists("data/nb_synthetic_dataset.csv"):
        nb_synthetic_acc, _ = run_experiment(  # ← Unpack tuple
            "data/nb_synthetic_dataset.csv", "NB-GENERATED SYNTHETIC", embedder
        )
        results["nb_synthetic"] = nb_synthetic_acc
    else:
        print("\n  nb_synthetic_dataset.csv not found!")
        print("Please run: python preprocessing/generate_NB_synthetic.py")
        results["nb_synthetic"] = None

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
    print("FINAL SUMMARY - MEAN POOLING NN")
    print("=" * 70)

    # print results table
    print("\n{:<30s} {:>12s}".format("Dataset", "Accuracy"))
    print("-" * 44)

    if "synthetic" in results:
        print(
            "{:<30s} {:>11.2f}%".format(
                "Template Synthetic", results["synthetic"] * 100
            )
        )

    if "nb_synthetic" in results and results["nb_synthetic"] is not None:
        print(
            "{:<30s} {:>11.2f}%".format(
                "NB-Generated Synthetic", results["nb_synthetic"] * 100
            )
        )

    if "real" in results:
        print("{:<30s} {:>11.2f}%".format("Real Data", results["real"] * 100))

    # print analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    if "synthetic" in results and "real" in results:
        gap = results["synthetic"] - results["real"]
        print(f"\nTemplate Synthetic → Real Data Gap: {gap:.4f} ({gap*100:.2f}%)")

    if "nb_synthetic" in results and results["nb_synthetic"] is not None:
        if "synthetic" in results:
            nb_vs_template = results["nb_synthetic"] - results["synthetic"]
            print(
                f"\nNB-Generated → Template Gap: {nb_vs_template:.4f} ({nb_vs_template*100:.2f}%)"
            )
            if abs(nb_vs_template) < 0.05:
                print("  Similar performance: Both near-perfect on synthetic data")
            else:
                print("  NB-generated is harder than template synthetic")

        if "real" in results:
            nb_vs_real = results["nb_synthetic"] - results["real"]
            print(
                f"\nNB-Generated → Real Data Gap: {nb_vs_real:.4f} ({nb_vs_real*100:.2f}%)"
            )
            if nb_vs_real > 0.10:
                print("  Large gap: Real data has complexity beyond NB's model")
                print("  NB's independence assumptions don't capture real patterns")
            elif nb_vs_real > 0.05:
                print("  Moderate gap: NB-generated is somewhat easier than real")
            else:
                print("  Small gap: NB-generated captures real data difficulty well")
                print("  NB's model approximates real data distribution")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
