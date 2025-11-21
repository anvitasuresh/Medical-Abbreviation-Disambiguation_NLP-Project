import pandas as pd

def load_synthetic():
    return pd.read_csv('./data/synthetic_dataset.csv')

def load_real():
    return pd.read_csv('./data/filtered_dataset.csv')