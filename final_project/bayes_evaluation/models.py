import numpy as np
from collections import Counter

class MostFrequentBaseline:
    def fit(self, y_train):
        self.most_frequent = Counter(y_train).most_common(1)[0][0]
    
    def predict(self, X_test):
        return np.array([self.most_frequent] * len(X_test))

class MultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        
    def fit(self, X_train, y_train):
        self.classes = np.unique(y_train)
        self.class_probs = {}
        self.feature_counts = {}
        
        for c in self.classes:
            X_c = X_train[y_train == c]
            self.class_probs[c] = len(X_c) / len(y_train)
            feature_sum = X_c.sum(axis=0) + self.alpha
            self.feature_counts[c] = np.log(feature_sum / feature_sum.sum())
    
    def predict(self, X_test):
        predictions = []
        for x in X_test:
            scores = {c: np.log(self.class_probs[c]) + np.sum(x * self.feature_counts[c]) 
                     for c in self.classes}
            predictions.append(max(scores, key=scores.get))
        return np.array(predictions)