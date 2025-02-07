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
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        X = chunk.iloc[:, :-1].values  # Features
        y = chunk.iloc[:, -1].values   # Labels
        yield X, y

# Define a custom kernel function to improve class separability
def custom_kernel(X1, X2):
    return (1 + np.dot(X1, X2.T)) ** 3  # Custom polynomial kernel (cubic)

# Train SVM incrementally
def incremental_svm_training(file_path, batch_size=1000, max_batches=100):
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
    def objective(C, gamma):
        model = SVC(kernel=custom_kernel, C=C, gamma=gamma, class_weight='balanced', probability=True)
        X_train, y_train = next(load_data_in_chunks(file_path, chunk_size=5000))
        X_train = StandardScaler().fit_transform(X_train)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_train)
        return -f1_score(y_train, y_pred)  # Negative because BayesOpt minimizes

    search_space = {
        'C': (0.1, 1000),
        'gamma': (1e-4, 1)
    }

    optimizer = BayesSearchCV(
        SVC(kernel=custom_kernel, class_weight='balanced'), 
        search_space, 
        n_iter=30, 
        cv=3,
        scoring=lambda est, X, y: objective(est.C, est.gamma)  # ✅ Objective function is now used correctly
    )

    optimizer.fit(*next(load_data_in_chunks(file_path, chunk_size=5000)))
    return optimizer.best_params_

# Evaluate model performance
def evaluate_model(model, scaler, file_path, test_size=5000):
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
print("🔍 Performing Incremental Training...")
svm_model, scaler = incremental_svm_training(file_path)

print("🔍 Optimizing Hyperparameters...")
best_params = optimize_svm(file_path)
print(f"✅ Best Parameters Found: {best_params}")

# Train Final Model with Optimized Parameters
print("🚀 Training Final Model with Optimized Parameters...")
final_svm = SVC(kernel=custom_kernel, C=best_params['C'], gamma=best_params['gamma'], class_weight='balanced', probability=True)

X_train, y_train = next(load_data_in_chunks(file_path, chunk_size=5000))
X_train = scaler.transform(X_train)
final_svm.fit(X_train, y_train)

# Save Final Model
joblib.dump(final_svm, "final_svm_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Model training complete. Evaluating...")
evaluate_model(final_svm, scaler, file_path)
