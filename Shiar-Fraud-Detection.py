import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from skopt import BayesSearchCV
import joblib

# Load dataset in chunks for incremental training
def load_data_in_chunks(file_path, chunk_size=1000):
    """ Generator function to load dataset in chunks to avoid memory overload. """
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        X = chunk.iloc[:, :-1].values  # Features
        y = chunk.iloc[:, -1].values   # Labels
        yield X, y

# Define a custom kernel function to improve class separability
def custom_kernel(X1, X2):
    """ Custom cubic polynomial kernel to improve separability. """
    return (1 + np.dot(X1, X2.T)) ** 3

# Train SVM incrementally
def incremental_svm_training(file_path, batch_size=1000, max_batches=100):
    """ Incremental training using SGDClassifier to handle large datasets. """
    scaler = StandardScaler()
    model = SGDClassifier(loss='hinge', class_weight='balanced', max_iter=1, warm_start=True)
    
    batch_count = 0
    for X_batch, y_batch in load_data_in_chunks(file_path, batch_size):
        X_batch = scaler.fit_transform(X_batch)  # Standardize features
        model.partial_fit(X_batch, y_batch, classes=np.array([0, 1]))  # Incremental training
        batch_count += 1
        if batch_count >= max_batches:
            break  # Stop after max_batches to prevent memory overload
    
    return model, scaler

# Bayesian Optimization for Hyperparameter Tuning
def optimize_svm(file_path):
    """ Uses Bayesian Optimization to find the best hyperparameters for SVM. """
    search_space = {
        'C': (0.1, 1000, "log-uniform"),  # Log scale improves search efficiency
        'gamma': (1e-4, 1, "log-uniform")
    }
    
    optimizer = BayesSearchCV(
        SVC(kernel="poly", degree=3, class_weight='balanced', probability=True),
        search_space,
        n_iter=30,
        cv=3,
        scoring='f1',  # Optimize for F1-score
        n_jobs=-1  # Use all available CPUs
    )

    # Load one chunk of training data for tuning
    X_train, y_train = next(load_data_in_chunks(file_path, chunk_size=5000))
    X_train = StandardScaler().fit_transform(X_train)

    optimizer.fit(X_train, y_train)  # Perform Bayesian Optimization
    
    print(f"Best Parameters Found: {optimizer.best_params_}")
    return optimizer.best_params_

# Evaluate model performance
def evaluate_model(model, scaler, file_path, test_size=5000):
    """ Evaluates the trained model using precision, recall, F1-score, and AUC-ROC. """
    X_test, y_test = next(load_data_in_chunks(file_path, test_size))
    X_test = scaler.transform(X_test)
    y_pred = model.predict(X_test)
    y_pred_prob = model.decision_function(X_test)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_prob)

    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}, AUC-ROC: {auc:.4f}")
    return precision, recall, f1, auc

# Main Execution
file_path = "fraud_detection_dataset_tiny.csv"  # Use the dataset file
svm_model, scaler = incremental_svm_training(file_path)
best_params = optimize_svm(file_path)
evaluate_model(svm_model, scaler, file_path)
