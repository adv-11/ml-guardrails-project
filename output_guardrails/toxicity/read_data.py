import pandas as pd

def read_data(data_path):
    """
    Returns train/test dataframes:
    - train_df,
    - test_df, 
    - test_labels_df
    """
    train_filename = "train.csv"
    test_filename = "test.csv"
    test_labels_filename = "test_labels.csv"

    train_df = pd.read_csv(data_path + train_filename)
    test_df = pd.read_csv(data_path + test_filename)
    test_labels_df = pd.read_csv(data_path + test_labels_filename)

    return train_df, test_df, test_labels_df

# Dataset information:
# - Source: Jigsaw Toxic Comment Classification Challenge
# - URL: https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data
# - License: CC0 (check Kaggle page for exact terms)
