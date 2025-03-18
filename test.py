import streamlit as st
import mlflow
import mlflow.sklearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

def load_data():
    mnist = fetch_openml('mnist_784', version=1)
    return mnist.data, mnist.target.astype(int)

X, y = load_data()

st.sidebar.title("📌 Menu Ứng Dụng")
app_mode = st.sidebar.radio("Chọn chức năng", [
    "Classification MNIST", "Clustering Algorithms",
    "Neural Network", "PCA t-SNE", "Semi-Supervised"
])

def log_mlflow(model, X_valid, y_valid, model_name):
    with mlflow.start_run():
        mlflow.log_param("Model", model_name)
        mlflow.sklearn.log_model(model, model_name)
        y_pred = model.predict(X_valid)
        acc = accuracy_score(y_valid, y_pred)
        mlflow.log_metric("Accuracy", acc)

if app_mode == "Semi-Supervised":
    st.title("🧠 Pseudo Labelling với Neural Network")
    
    # Chia tập train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Chọn 1% dữ liệu có nhãn ban đầu
    def sample_labeled_data(X, y, fraction=0.01):
        X_labeled, y_labeled = [], []
        for label in np.unique(y):
            idx = np.where(y == label)[0]
            chosen_idx = np.random.choice(idx, max(1, int(len(idx) * fraction)), replace=False)
            X_labeled.append(X[chosen_idx])
            y_labeled.append(y[chosen_idx])
        return np.vstack(X_labeled), np.hstack(y_labeled)
    
    X_labeled, y_labeled = sample_labeled_data(X_train, y_train)
    X_unlabeled = np.array([x for i, x in enumerate(X_train) if i not in set(y_labeled)])
    
    # Tham số huấn luyện
    threshold = st.slider("Ngưỡng gán nhãn (Confidence Threshold)", 0.5, 1.0, 0.95, step=0.05)
    max_iterations = st.slider("Số vòng lặp tối đa", 1, 10, 5)
    
    if st.button("🚀 Huấn luyện Pseudo Labelling"):
        model = MLPClassifier(hidden_layer_sizes=(50, 50), activation='relu', solver='adam', max_iter=50)
        iteration = 0
        while iteration < max_iterations and len(X_unlabeled) > 0:
            model.fit(X_labeled, y_labeled)
            probs = model.predict_proba(X_unlabeled)
            max_probs = np.max(probs, axis=1)
            confident_idx = np.where(max_probs >= threshold)[0]
            
            if len(confident_idx) == 0:
                break
            
            X_confident = X_unlabeled[confident_idx]
            y_confident = np.argmax(probs[confident_idx], axis=1)
            
            X_labeled = np.vstack([X_labeled, X_confident])
            y_labeled = np.hstack([y_labeled, y_confident])
            X_unlabeled = np.delete(X_unlabeled, confident_idx, axis=0)
            iteration += 1
        
        acc = accuracy_score(y_test, model.predict(X_test))
        st.write(f"🎯 Độ chính xác cuối cùng: {acc:.4f}")
        log_mlflow(model, X_test, y_test, "Pseudo Labelling NN")
        
        st.success("✅ Huấn luyện hoàn tất!")