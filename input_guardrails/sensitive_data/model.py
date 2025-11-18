import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import MultiLabelBinarizer


# Load training and testing data in panda dataframes
train_path = "Training_Set2_Final_Cleaned.xlsx"
test_path = "Testing_Set2_Final_Cleaned.xlsx"

train_df = pd.read_excel(train_path)
test_df = pd.read_excel(test_path)

print(train_df.head())
print(test_df.head())


# Extract entity types from true predictions
def extract_labels(true_pred):
    '''
    Extract entity labels like credit card, email, phone, address, ssn from the string.
    '''
    if not isinstance(true_pred, str):
        return []
    return re.findall(r"'(\w+)'", true_pred)
# apply the function above, extract_labels, to the True Predictions column
# to create a clean list of labels
train_df["labels"] = train_df["True Predictions"].apply(extract_labels)
test_df["labels"] = test_df["True Predictions"].apply(extract_labels)


# Encode multi-label targets
# converts the list of labels into binary arrays
# e.g. [credit card, email] -> [1, 1, 0, 0, 0]
# fit_transform learns all possible label classes from training data
# transform encodes test data accordingly
mlb = MultiLabelBinarizer()
y_train = mlb.fit_transform(train_df["labels"])
y_test = mlb.transform(test_df["labels"])


# TF-IDF vectorization
# converts Text column into numerical vectors using term frequency-inverse document frequency
# max_features=5000 keeps top 5000 words/bigrams, ngram_range=(1,2) uses single words and pairs of words(bigrams),
# stop_words="english" removes stopwords (common words filtered out in natural language processing to improve efficiency and accuracy)
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
X_train = vectorizer.fit_transform(train_df["Text"].astype(str))
X_test = vectorizer.transform(test_df["Text"].astype(str))


# hyperparameter tuning
# tries multiple hyperparameter combinations to find the best random forest configuration
# from sklearn.model_selection import GridSearchCV
# from sklearn.ensemble import RandomForestClassifier

# param_grid = {
#     'n_estimators': [100, 200],
#     'max_depth': [None, 10, 20],
#     'min_samples_split': [2, 5],
#     'min_samples_leaf': [1, 2],
#     'bootstrap': [True, False]
# }

# grid_search = GridSearchCV(RandomForestClassifier(), param_grid=param_grid, cv=5)
# grid_search.fit(X_train, y_train)

# print("Best Parameters:", grid_search.best_params_)
# print("Best Estimator:", grid_search.best_estimator_)



# Train random forest (multi-label)
# creates random forest with 100 trees
rf = RandomForestClassifier(n_estimators=100, random_state=42, # max_features="sqrt",
    max_depth = None, # max_leaf_nodes=6,
    min_samples_leaf=1, min_samples_split=2
)
# MOC wrapper to support multi-label classification
# fits model to training TF-IDF vectors and labels
multi_rf = MultiOutputClassifier(rf)
multi_rf.fit(X_train, y_train)


# Evaluate model
# makes predictions on test set
y_pred = multi_rf.predict(X_test)


# prints precision, recall, and f1-score per label type
print("\n=== Classification Report (Multi-Label) ===")
print(classification_report(y_test, y_pred, target_names=mlb.classes_))


# Plot F1-scores
# calculates f1-score for each label separately
f1_scores = f1_score(y_test, y_pred, average=None)
f1_df = pd.DataFrame({'Entity Type': mlb.classes_, 'F1-Score': f1_scores})


# Show sample predictions
# converts predicted binary arrays back to readable label names
# creates a table comparing true vs predicted labels for sample text
pred_labels = mlb.inverse_transform(y_pred)
results = pd.DataFrame({
    "Text": test_df["Text"],
    "True Labels": test_df["labels"],
    "Predicted Labels": pred_labels
})

# print("\nSample Predictions:")
# print(results.head())
