"""Python script equivalent of evaluate_model.ipynb

Currently parameters need to be manually set...

An output file will be written into a folder named "output".

Example output: output/logreg_0_C1_tol0.0001_mi500.txt
- interpretation: Logistic Regression with class weight balancing, C=1, tol=0.0001, and max_iter=500
"""

import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import ClassifierChain
from sklearn.metrics import classification_report, roc_auc_score#, roc_curve
from sklearn.preprocessing import label_binarize
#import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
from scipy.sparse import hstack, csr_matrix

SCRIPT_DIR = Path(__file__).resolve().parent

SEED = 42

# parameters:
RANDOM = False
IMBALANCE = 0  # 0=class_weight, 1=undersampling
FEATURE = 1  # 0=TF-IDF, 1=embedding, 2=both
MODEL = 2  # 0=LogReg, 1=XGB, 2=RF

def select_model(order):
    """initialize ML model"""
    base = None
    if MODEL == 1:
        base = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, gamma=0, min_child_weight=1, 
                             scale_pos_weight=(1 if IMBALANCE == 1 else 10), 
                             random_state=(None if RANDOM else SEED))
    elif MODEL == 2:
        base = RandomForestClassifier(n_estimators=100, max_depth=6, min_samples_split=2, min_samples_leaf=1, 
                                      max_features="sqrt", bootstrap=True, 
                                      class_weight=(None if IMBALANCE == 1 else "balanced"), 
                                      random_state=(None if RANDOM else SEED))
    else:
        base = LogisticRegression(C=1, tol=0.0001, max_iter=501, solver="lbfgs", 
                                  class_weight=(None if IMBALANCE == 1 else "balanced"), 
                                  random_state=(None if RANDOM else SEED))
    
    chain = ClassifierChain(estimator=base, order=order, chain_method="predict_proba", cv=None, 
                            random_state=(None if RANDOM else SEED))
    return chain


def load_data():
    """
    Returns train/test dataframes:
    - train_df,
    - test_df, 
    - test_labels_df
    """
    data_path = "data"
    train_filename = "train.csv"
    test_filename = "test.csv"
    test_labels_filename = "test_labels.csv"

    train_df = pd.read_csv(SCRIPT_DIR / data_path / train_filename)
    test_df = pd.read_csv(SCRIPT_DIR / data_path / test_filename)
    test_labels_df = pd.read_csv(SCRIPT_DIR / data_path / test_labels_filename)

    # ignore test values with -1 labels
    test_df_usable = test_df.loc[test_labels_df['toxic'] != -1]
    test_labels_df_usable = test_labels_df.loc[test_labels_df['toxic'] != -1]

    return train_df, test_df_usable, test_labels_df_usable


def undersample(combined_X, combined_y):
    """Undersample comments with no positive labels"""
    neg_indices = combined_y[(combined_y == 0).all(axis=1)].index.tolist()
    pos_indices = combined_y[(combined_y != 0).any(axis=1)].index.tolist()

    if RANDOM == False:
        np.random.seed(SEED)
    
    neg_indices = np.random.choice(neg_indices, len(pos_indices), replace=False)

    test_size = 0.2
    split_point = int(test_size * len(neg_indices))
    np.random.shuffle(neg_indices)
    np.random.shuffle(pos_indices)
    train_indices = np.concat([neg_indices[split_point:], pos_indices[split_point:]])
    test_indices = np.concat([neg_indices[:split_point], pos_indices[:split_point]])
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)

    # establish train and test datasets
    train_X = combined_X.iloc[train_indices]
    test_X = combined_X.iloc[test_indices]
    train_y = combined_y.iloc[train_indices]
    test_y = combined_y.iloc[test_indices]
    return train_X, test_X, train_y, test_y


def get_features(train_X, test_X):
    train_X_tf = test_X_tf = train_X_embed = test_X_embed = None
    if FEATURE == 0 or FEATURE == 2:
        tf_idf = TfidfVectorizer(encoding="utf-8", analyzer="word", ngram_range=(1,1), lowercase=True, stop_words="english")
        train_X_tf = tf_idf.fit_transform(train_X)
        test_X_tf = tf_idf.transform(test_X)
    if FEATURE == 1 or FEATURE == 2:
        embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        train_X_embed = embed_model.encode(train_X.tolist(), show_progress_bar=True, convert_to_numpy=True)
        test_X_embed = embed_model.encode(test_X.tolist(), show_progress_bar=True, convert_to_numpy=True)
        scaler = StandardScaler()
        train_X_embed = scaler.fit_transform(train_X_embed)
        test_X_embed = scaler.transform(test_X_embed)
    if FEATURE == 0:
        return train_X_tf, test_X_tf
    elif FEATURE == 1:
        return train_X_embed, test_X_embed
    else:
        train_X_both = hstack([train_X_tf, csr_matrix(train_X_embed)])
        test_X_both = hstack([test_X_tf, csr_matrix(test_X_embed)])
        return train_X_both, test_X_both


def get_output_filename(model_params):
    """Return a unique filename based on model parameters"""
    def get_param(param):
        return model_params[f"estimator__{param}"]
    models = ["logreg", "xgb", "rf"]
    features = ["tf", "emb", "both"]
    filename = f"{models[MODEL]}_{features[FEATURE]}_{IMBALANCE}"
    if MODEL == 1:
        filename += f"_est{get_param('n_estimators')}_md{get_param('max_depth')}"
        filename += f"_lr{get_param('learning_rate')}_g{get_param('gamma')}"
    elif MODEL == 2:
        filename += f"_est{get_param('n_estimators')}_md{get_param('max_depth')}"
        filename += f"_mss{get_param('min_samples_split')}_msl{get_param('min_samples_leaf')}"
    else:
        filename += f"_C{get_param('C')}_tol{get_param('tol')}_mi{get_param('max_iter')}"
    filename += ".txt"
    return filename


def main():
    # load dataset
    print("Loading data...")
    train_df, test_df, test_labels_df = load_data()    
    label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    # combine pre-selected train/test data
    combined_X = pd.concat([train_df['comment_text'], test_df['comment_text']])
    combined_y = pd.concat([train_df[label_cols], test_labels_df[label_cols]])

    # establish train and test datasets
    train_X = test_X = train_y = test_y = None
    if IMBALANCE == 1:
        train_X, test_X, train_y, test_y = undersample(combined_X, combined_y)
    else:
        if RANDOM == False:
            train_X, test_X, train_y, test_y = train_test_split(combined_X, combined_y, train_size=0.8, test_size=0.2, random_state=SEED)
        else:
            train_X, test_X, train_y, test_y = train_test_split(combined_X, combined_y, train_size=0.8, test_size=0.2, random_state=SEED)
    #print(train_X.shape, test_X.shape, train_y.shape, test_y.shape)
    msg_data = f"Train/Test datasets: {train_X.shape}, {test_X.shape}, {train_y.shape}, {test_y.shape}\n"

    # get features
    print("Getting features...")
    train_X, test_X = get_features(train_X, test_X)
    msg_features = f"\nShape of features: {train_X.shape}, {test_X.shape}\n"

    # initialize and train model
    order = [4, 2, 0, 1, 5, 3]
    model = select_model(order)

    model_params = model.get_params(deep=True)
    msg_params = "\nModel Parameters\n" + "-" * 20 + "\n"  #print("Model Parameters", "-" * 20, sep='\n')
    for key, val in model_params.items():
        msg_params += f"{key}: {val}\n"  #print(f"{key}: {val}")

    print("Training model...")
    start = time.perf_counter()
    model.fit(train_X, train_y)
    end = time.perf_counter()
    msg_time = f"\nTraining Time: {(end-start)//60}m {(end-start)%60:.3f}s\n"

    # test and evaluate model
    print("Evaluating model...")
    pred = model.predict(test_X)
    pred_proba = model.predict_proba(test_X)

    msg_accuracy = f"\nAccuracy: {round(model.score(test_X, test_y), 4)}\n"
    #print(f"\nAccuracy: {round(model.score(test_X_tf, test_y), 4)}")
    msg_report = classification_report(y_true=test_y, y_pred=pred, target_names=[label_cols[idx] for idx in order], zero_division=np.nan)
    #print(classification_report(y_true=test_y, y_pred=pred, target_names=[label_cols[idx] for idx in order], zero_division=np.nan))
    
    y_true_bin = label_binarize(test_y, classes=[label_cols[idx] for idx in order])

    # Compute ROC curve and AUC for each class
    msg_auc = "\nAUC:\n"  #print("\nAUC:")
    for i in range(y_true_bin.shape[1]):
        roc_auc = roc_auc_score(y_true_bin[:, i], pred_proba[:, i])
        msg_auc += f"Class {i}: {label_cols[order[i]].ljust(15)} (score = {roc_auc:.2f})\n"
        #print(f"Class {i}: {label_cols[order[i]].ljust(15)} (score = {roc_auc:.2f})")
    
    # output msg strings
    msg = msg_data + msg_features + msg_params + msg_time + msg_accuracy + msg_report + msg_auc
    output_path = SCRIPT_DIR / "output" / get_output_filename(model_params)
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(msg)
    print(f"Script output written to {output_path}")


if __name__ == "__main__":
    main()
