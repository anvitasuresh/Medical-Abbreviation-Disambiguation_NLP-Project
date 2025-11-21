import numpy as np

def compute_tfidf(X_counts, batch_size=5000):
    n_samples, n_features = X_counts.shape
    
    df = (X_counts > 0).sum(axis=0)
    idf = np.log((n_samples + 1) / (df + 1)) + 1
    
    tfidf = np.zeros_like(X_counts, dtype=np.float32)
    
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch = X_counts[start:end]
        doc_lengths = batch.sum(axis=1, keepdims=True)
        doc_lengths[doc_lengths == 0] = 1
        tfidf[start:end] = (batch / doc_lengths) * idf
    
    return tfidf