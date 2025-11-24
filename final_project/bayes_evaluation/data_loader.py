import pandas as pd


def load_synthetic():
    return pd.read_csv("./data/synthetic_dataset.csv")


def load_real():
    return pd.read_csv("./data/filtered_dataset.csv")


def load_nb_synthetic():
    return pd.read_csv("./data/nb_synthetic_dataset.csv")
