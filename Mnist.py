import streamlit as st
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neural_network import MLPClassifier

# Tải dữ liệu MNIST (Cache để tăng tốc)
@st.cache_data
def load_data():
    mnist = fetch_openml('mnist_784', version=1)
    return mnist.data, mnist.target.astype(int)

X, y = load_data()

# Sidebar - Menu điều hướng
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

# --- CHỨC NĂNG PHÂN LOẠI ---
if app_mode == "Classification MNIST":
    st.title("🔍 Phân loại chữ số MNIST")
    
    if "page" not in st.session_state:
        st.session_state.page = "theory"
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📖 Lý thuyết"):
            st.session_state.page = "theory"
    with col2:
        if st.button("⚙️ Huấn luyện mô hình"):
            st.session_state.page = "training"
    
    if st.session_state.page == "theory":
        st.header("📖 Lý thuyết về phân loại chữ số MNIST")
        st.markdown("""
        Bộ dữ liệu MNIST chứa 70.000 hình ảnh chữ số viết tay (0-9), mỗi ảnh có kích thước 28x28 pixel.
        Đây là một trong những bộ dữ liệu phổ biến nhất trong lĩnh vực Machine Learning và Deep Learning.
        
        **1. Định dạng dữ liệu**
        - Mỗi ảnh là một ma trận 28x28 pixel, được chuyển thành vector có 784 chiều.
        - Giá trị pixel nằm trong khoảng từ 0 đến 255 (có thể được chuẩn hóa về khoảng [0,1]).
        
        **2. Các thuật toán được sử dụng:**
        - **Decision Tree**: Dễ hiểu, dễ triển khai, nhưng có thể bị overfitting.
        - **SVM (Support Vector Machine)**: Hiệu suất cao nhưng có thể chậm trên tập dữ liệu lớn.
        
        **3. Ứng dụng thực tế**
        - Nhận dạng chữ viết tay trong hệ thống nhập liệu tự động.
        - Chuyển đổi tài liệu viết tay sang văn bản số hóa.
        - Hỗ trợ hệ thống nhận diện chữ số trên séc ngân hàng, biển số xe.
        """)
    
    
    elif st.session_state.page == "training":
        st.header("⚙️ Huấn luyện mô hình")
        
        # Chia dữ liệu
        train_size = st.slider("Train (%)", 50, 80, 70)
        val_size = st.slider("Validation (%)", 10, 30, 15)
        sample_size = st.slider("Số mẫu train", 1000, 10000, 5000)
        
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=train_size/100, stratify=y)
        X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=val_size/(100-train_size), stratify=y_temp)
        
        # Chọn mô hình
        model_choice = st.selectbox("Chọn mô hình", ["Decision Tree", "SVM"])
        
        if st.button("🚀 Huấn luyện"):
            if model_choice == "Decision Tree":
                model = DecisionTreeClassifier()
            else:
                model = SVC()
            
            model.fit(X_train[:sample_size], y_train[:sample_size])
            acc = accuracy_score(y_valid, model.predict(X_valid))
            st.write(f"🎯 Độ chính xác trên tập validation: {acc:.4f}")
            
            # Log MLFlow
            log_mlflow(model, X_valid, y_valid, model_choice)
            
            # Demo dự đoán
            st.subheader("🎨 Demo dự đoán")
            indices = np.random.choice(len(X_test), 5, replace=False)
            X_test = np.array(X_test)  # Chuyển thành numpy array để tránh lỗi KeyError
            fig, axes = plt.subplots(1, 5, figsize=(10, 2))
            for i, idx in enumerate(indices):
                axes[i].imshow(X_test[idx].reshape(28, 28), cmap="gray")
                axes[i].axis("off")
                pred = model.predict([X_test[idx]])[0]
                axes[i].set_title(f"Dự đoán: {pred}")
            st.pyplot(fig)


# --- CHỨC NĂNG PHÂN CỤM ---
if app_mode == "Clustering Algorithms":
    st.title("🔍 Phân cụm chữ số MNIST")
    
    if "page" not in st.session_state:
        st.session_state.page = "theory"
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📖 Lý thuyết"):
            st.session_state.page = "theory"
    with col2:
        if st.button("⚙️ Huấn luyện mô hình"):
            st.session_state.page = "training"
    
    if st.session_state.page == "theory":
        st.header("📖 Lý thuyết về phân cụm")
        st.markdown("""
        Phân cụm là một kỹ thuật trong Machine Learning nhằm nhóm các điểm dữ liệu tương đồng lại với nhau. Trong bài toán MNIST, phân cụm giúp chúng ta phát hiện các nhóm chữ số tương tự mà không cần nhãn trước.
    
    **1. K-means**
    - Là thuật toán phân cụm phổ biến nhất.
    - Hoạt động bằng cách gán dữ liệu vào K cụm dựa trên khoảng cách tới tâm cụm.
    - Cập nhật lại tâm cụm dựa trên trung bình của các điểm trong cụm cho đến khi hội tụ.
    - Nhược điểm: Cần xác định trước số cụm K, có thể bị ảnh hưởng bởi giá trị ban đầu.
    
    **2. DBSCAN**
    - DBSCAN (Density-Based Spatial Clustering of Applications with Noise) là thuật toán phân cụm dựa trên mật độ.
    - Hoạt động bằng cách tìm kiếm các vùng dữ liệu có mật độ cao và mở rộng cụm từ các điểm lõi.
    - Có khả năng phát hiện nhiễu tốt hơn K-means.
    - Nhược điểm: Cần xác định trước hai tham số: Epsilon (khoảng cách tối đa để xác định hàng xóm) và min_samples (số điểm tối thiểu để tạo thành cụm).
    
    **Ứng dụng thực tế:**
    - Nhận diện chữ viết tay khi không có nhãn sẵn.
    - Phát hiện bất thường trong dữ liệu số.
    - Tạo nhóm người dùng có hành vi tương đồng trên hệ thống.""")
    
    elif st.session_state.page == "training":
        st.header("⚙️ Huấn luyện mô hình")
        clustering_method = st.selectbox("Chọn thuật toán phân cụm", ["K-means", "DBSCAN"])
        num_samples = st.slider("Số lượng mẫu sử dụng", 1000, 10000, 5000, step=1000)
        sample_indices = np.random.choice(len(X), num_samples, replace=False)
        X_sample = X.iloc[sample_indices]
        
        @st.cache_data
        def reduce_dimensionality(X_data, n_components=2):
            pca = PCA(n_components=n_components)
            return pca.fit_transform(X_data)
        
        X_pca = reduce_dimensionality(X_sample)
        
        if clustering_method == "K-means":
            k = st.slider("Số cụm (K)", 2, 20, 10)
            model = KMeans(n_clusters=k, random_state=42)
        else:
            eps = st.slider("Epsilon (DBSCAN)", 0.1, 5.0, 1.0)
            min_samples = st.slider("Số mẫu tối thiểu", 2, 20, 5)
            model = DBSCAN(eps=eps, min_samples=min_samples)
        
        if st.button("🚀 Thực hiện phân cụm"):
            labels = model.fit_predict(X_sample)
            log_mlflow(model, X_sample, labels, clustering_method)
            
            st.success("✅ Phân cụm hoàn thành!")
            fig, ax = plt.subplots()
            scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.5)
            legend1 = ax.legend(*scatter.legend_elements(), title="Cụm")
            ax.add_artist(legend1)
            st.pyplot(fig)
# --- CHỨC NĂNG PCA & t-SNE ---
if app_mode == "PCA t-SNE":
    st.title("📉 PCA & t-SNE trên MNIST")
    
    method = st.radio("Chọn phương pháp giảm chiều:", ["PCA", "t-SNE"])
    num_components = st.slider("Số chiều giảm xuống", 2, 50, 2)
    
    if st.button("🔄 Thực hiện giảm chiều"):
        if method == "PCA":
            reducer = PCA(n_components=num_components)
        else:
            reducer = TSNE(n_components=2, perplexity=30, random_state=42)
        
        X_reduced = reducer.fit_transform(X[:5000])
        
        # Log MLFlow
        with mlflow.start_run():
            mlflow.log_param("Method", method)
            mlflow.log_param("Components", num_components)
        
        # Hiển thị kết quả
        st.subheader("📊 Kết quả trực quan hóa")
        fig, ax = plt.subplots()
        scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y[:5000], cmap="jet", alpha=0.5)
        fig.colorbar(scatter)
        st.pyplot(fig)
# --- NEURAL NETWORK ---
if app_mode == "Neural Network":
    st.title("🧠 Neural Network trên MNIST")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    hidden_layers = st.slider("Số neuron mỗi lớp ẩn", 10, 200, 50)
    num_layers = st.slider("Số lớp ẩn", 1, 5, 2)
    activation = st.selectbox("Activation Function", ["relu", "tanh", "logistic"])
    optimizer = st.selectbox("Optimizer", ["adam", "sgd", "lbfgs"])
    epochs = st.slider("Số epoch", 10, 100, 50)
    
    if st.button("🚀 Huấn luyện mô hình"):
        model = MLPClassifier(hidden_layer_sizes=(hidden_layers,) * num_layers, 
                              activation=activation, solver=optimizer,
                              max_iter=epochs, random_state=42)
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        
        st.write(f"🎯 Độ chính xác: {acc:.4f}")
        log_mlflow(model, X_test, y_test, "Neural Network")
        
        # Biểu đồ Loss Curve
        st.subheader("📉 Biểu đồ Loss Curve")
        fig, ax = plt.subplots()
        ax.plot(model.loss_curve_)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Quá trình huấn luyện")
        st.pyplot(fig)
        
        # Demo dự đoán
        st.subheader("🎨 Demo dự đoán")
        indices = np.random.choice(len(X_test), 5, replace=False)
        fig, axes = plt.subplots(1, 5, figsize=(10, 2))
        for i, idx in enumerate(indices):
            axes[i].imshow(X_test[idx].reshape(28, 28), cmap="gray")
            axes[i].axis("off")
            pred = model.predict([X_test[idx]])[0]
            axes[i].set_title(f"Dự đoán: {pred}")
        st.pyplot(fig)

