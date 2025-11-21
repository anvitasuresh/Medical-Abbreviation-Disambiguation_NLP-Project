import numpy as np

def extract_context(row, window_size=5):
    text_words = row['text'].lower().split()
    loc = int(row['location'])
    start = max(0, loc - window_size)
    end = min(len(text_words), loc + window_size + 1)
    return text_words[start:loc] + text_words[loc+1:end]

def get_ngrams(words, n):
    return [' '.join(words[i:i+n]) for i in range(len(words) - n + 1)]

def build_vocabulary(contexts, min_freq=1):
    from collections import Counter
    ngram_counts = Counter()
    for context_ngrams in contexts:
        for ngram in context_ngrams:
            ngram_counts[ngram] += 1
    
    filtered = [ngram for ngram, count in ngram_counts.items() if count >= min_freq]
    return {ngram: idx for idx, ngram in enumerate(filtered)}

def vectorize(context_ngrams, vocab):
    vector = np.zeros(len(vocab), dtype=np.float32)
    for ngram in context_ngrams:
        if ngram in vocab:
            vector[vocab[ngram]] += 1
    return vector

def process_data(df, vocabulary=None, min_freq=1):
    contexts = []
    labels = []
    
    for _, row in df.iterrows():
        context_words = extract_context(row)
        ngrams = context_words.copy()
        ngrams.extend(get_ngrams(context_words, 2))
        ngrams.extend(get_ngrams(context_words, 3))
        contexts.append(ngrams)
        labels.append(row['label'])
    
    if vocabulary is None:
        vocabulary = build_vocabulary(contexts, min_freq)
    
    X = np.array([vectorize(ctx, vocabulary) for ctx in contexts], dtype=np.float32)
    y = np.array(labels)
    
    return X, y, vocabulary