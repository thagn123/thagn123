# 🧠 LỘ TRÌNH HỌC AI – TUẦN 1 (CHUYÊN SÂU & CÓ NHIỆM VỤ NÂNG CAO)

## 📦 MỤC TIÊU TUẦN 1
- Làm chủ môi trường phát triển AI (Python, VS Code, GitHub, Colab)  
- Nắm chắc kiến thức về xử lý dữ liệu (NumPy, Pandas)  
- Biết trực quan hóa và tiền xử lý dữ liệu  
- Hoàn thành mini-project đầu tiên  

---

## 📅 Ngày 1: Chuẩn bị môi trường & công cụ AI  
🎯 **Mục tiêu:** Thành thạo cài đặt và cấu hình công cụ  
### ✅ Checklist cơ bản
- [ ] Cài Python 3.10+ và pip  
- [ ] Cài Git và tạo tài khoản GitHub  
- [ ] Cài Visual Studio Code + Extension Python & Jupyter  
- [ ] Cài Jupyter Notebook (pip install notebook)  
- [ ] Tạo tài khoản Google Colab  
- [ ] Tạo repo GitHub `ai_learning_journey` & clone về  
- [ ] Tạo file `hello_ai.ipynb` với nội dung `print("Hello AI 🤖")`  
- [ ] Push file lên GitHub, kiểm tra Travis/CI (nếu có)  

### 🚀 Nhiệm vụ nâng cao
- [ ] Cấu hình VS Code tự động sync với GitHub  
- [ ] Tạo template notebook `dayX_template.ipynb` (có sections: mục tiêu, lý thuyết, thực hành, log)  
- [ ] Khám phá và cài thêm 1 extension hỗ trợ AI (Ví dụ: Python Docstring Generator)  

### 🧠 Câu hỏi tư duy
- Colab và Jupyter Notebook khác nhau như thế nào?  
- Vì sao nên sử dụng Git trong quá trình học AI?  

### 📝 Nhật ký học tập
- Ghi thời gian hoàn thành  
- Ghi lỗi phát sinh & cách xử lý  

---

## 📅 Ngày 2: Kiến thức cơ bản về AI & Machine Learning  
🎯 **Mục tiêu:** Phân biệt rõ AI – ML – DL và quy trình học có giám sát  
### 📚 Lý thuyết cốt lõi
- Định nghĩa AI, ML, Deep Learning  
- Supervised vs Unsupervised Learning  
- Khái niệm: Feature, Label, Dataset, Model, Training, Inference  

### ✅ Checklist cơ bản
- [ ] Vẽ sơ đồ AI → ML → DL  
- [ ] Tìm ví dụ thực tế cho mỗi lĩnh vực (3 ứng dụng)  
- [ ] Tạo 1 CSV nhỏ (10 dòng) gồm feature & label  
- [ ] Viết notebook `day2_ai_vs_ml.ipynb` tóm tắt kiến thức  

### 🚀 Nhiệm vụ nâng cao
- [ ] Viết đoạn blog ngắn (200 chữ) giải thích AI vs ML vs DL  
- [ ] Xây dựng script Python sinh ngẫu nhiên dataset “Dự đoán đậu đại học”  

### 🧠 Câu hỏi tư duy
- Khi nào nên dùng Unsupervised Learning?  
- Dataset “đẹp” cần đảm bảo yếu tố gì?  

---

## 📅 Ngày 3: Pandas & NumPy chuyên sâu  
🎯 **Mục tiêu:** Thành thạo thao tác với DataFrame và mảng  
### 📚 Lý thuyết cốt lõi
- Series vs DataFrame, Indexing (`.loc`, `.iloc`)  
- Xử lý missing: `isnull()`, `dropna()`, `fillna()`  
- Grouping: `groupby()`, `pivot_table()`  
- NumPy: slicing, broadcasting, masking  

### ✅ Checklist cơ bản
- [ ] Đọc `titanic.csv`, hiển thị 10 dòng đầu  
- [ ] `.info()`, `.shape()`, `.describe()`  
- [ ] Lọc `Age > 30` & `Fare > 50` với `.loc`  
- [ ] Tính trung bình `Fare` theo `Pclass`  
- [ ] Fill missing `Age` bằng median  
- [ ] Tạo mảng NumPy 5×5 ngẫu nhiên, mask các giá trị > 0.5  

### 🚀 Nhiệm vụ nâng cao
- [ ] Merge 2 DataFrame: passenger info + cabin info  
- [ ] Viết hàm loại bỏ outlier theo IQR  
- [ ] Tạo biểu đồ phân bố tuổi theo giới tính (using pandas plot)  

### 🧠 Câu hỏi tư duy
- `.apply()` khác gì `.map()` và vòng lặp?  
- Khi nào nên dùng `merge` vs `concat`?  

### 📝 Nhật ký học tập
- Ghi lại bước khó nhất & cách vượt qua  

---

## 📅 Ngày 4: Trực quan hóa dữ liệu  
🎯 **Mục tiêu:** Hiểu dữ liệu qua biểu đồ, phát hiện xu hướng  
### ✅ Checklist cơ bản
- [ ] Cài `seaborn`, `matplotlib`  
- [ ] Vẽ histogram `Age` (`sns.histplot`)  
- [ ] Vẽ scatterplot: `Age` vs `Fare` màu theo `Survived`  
- [ ] Vẽ countplot `Pclass` hoặc `Sex`  
- [ ] Vẽ heatmap correlation (`sns.heatmap(df.corr())`)  
- [ ] Dùng `plt.subplots()` để vẽ nhiều biểu đồ trên cùng figure  
- [ ] Lưu biểu đồ thành `.png` bằng `plt.savefig("chart.png")`  

### 🚀 Nhiệm vụ nâng cao
- [ ] Vẽ side-by-side chart: tỉ lệ sống sót theo giới tính & `Pclass`  
- [ ] Xây dựng file `data_viz_utils.py` với 3 hàm: plot_hist, plot_scatter, plot_heatmap  

### 🧠 Câu hỏi tư duy
- Loại biểu đồ nào phù hợp với dữ liệu phân loại?  
- Làm sao tối ưu chú thích và màu sắc để biểu đồ dễ hiểu?  

---

## 📅 Ngày 5: Linear Regression (Hồi quy tuyến tính)  
🎯 **Mục tiêu:** Huấn luyện & đánh giá mô hình hồi quy  
### ✅ Checklist cơ bản
- [ ] Tạo dataset giả lập `area vs price`  
- [ ] Train model `LinearRegression` từ sklearn  
- [ ] In hệ số (`coef_`) & intercept  
- [ ] Vẽ scatter + đường hồi quy  
- [ ] Tính MSE, MAE, R² score  

### 🚀 Nhiệm vụ nâng cao
- [ ] Thêm biến: `num_rooms`, `age_of_house` vào dataset  
- [ ] So sánh performance giữa Linear và Polynomial Regression  
- [ ] Tạo script `regression_utils.py` chứa 2 hàm: train_model & evaluate_model  

### 🧠 Câu hỏi tư duy
- Khi nào mô hình underfit / overfit?  
- Mối quan hệ nào không thể dùng hồi quy tuyến tính?  

---

## 📅 Ngày 6: Logistic Regression (Phân loại nhị phân)  
🎯 **Mục tiêu:** Huấn luyện & đánh giá mô hình phân loại  
### ✅ Checklist cơ bản
- [ ] Tiền xử lý `titanic.csv` (encode `Sex`, chọn features)  
- [ ] Train `LogisticRegression`  
- [ ] Dự đoán `Survived` & tính accuracy, precision, recall  
- [ ] Vẽ confusion matrix bằng `sns.heatmap()`  

### 🚀 Nhiệm vụ nâng cao
- [ ] Xây pipeline: preprocessing → model → evaluation  
- [ ] Thử với các threshold khác nhau & vẽ ROC curve  
- [ ] Viết script `classification_utils.py` cho reusable code  

### 🧠 Câu hỏi tư duy
- Khi nào ưu tiên precision hơn recall?  
- Logistic vs Linear Regression khác gì về output?  

---

## 📅 Ngày 7: Mini Project – Dự đoán đậu/rớt đại học  
🎯 **Mục tiêu:** Hoàn thiện 1 pipeline AI end-to-end  
### ✅ Checklist cơ bản
- [ ] Tạo file `students.csv`: Tên, Toán, Lý, Hóa, Văn, Đậu/Rớt (0/1)  
- [ ] Đọc & xử lý dữ liệu (encode, fillna)  
- [ ] Train model `LogisticRegression`  
- [ ] Dự đoán cho 5 học sinh mới  
- [ ] Vẽ biểu đồ so sánh kết quả  

### 🚀 Nhiệm vụ nâng cao
- [ ] Tạo giao diện CLI (input điểm → output kết quả)  
- [ ] Sinh draft báo cáo `project_report.md` gồm: mô tả data, phương pháp, kết quả  

### 🧠 Câu hỏi tư duy
- Nếu dùng mô hình khác (Decision Tree, SVM) có cải thiện không?  
- Có bias nào trong dữ liệu?  

---

# 🧠 LỘ TRÌNH HỌC AI – TUẦN 2 (CHUYÊN SÂU & NHIỆM VỤ NÂNG CAO)

## 📦 MỤC TIÊU TUẦN 2
- Thành thạo thuật toán Machine Learning nâng cao (Decision Tree, KNN, SVM, Clustering…)  
- Hiểu sâu quy trình huấn luyện, đánh giá và tối ưu mô hình  
- Biết dùng Pipeline, GridSearch, Cross-Validation  

---

## 📅 Ngày 8: K-Nearest Neighbors (KNN) & Naive Bayes  
🎯 **Mục tiêu:** Hiểu & áp dụng KNN, Naive Bayes cho phân loại  

### 📚 Lý thuyết cốt lõi  
- KNN: khoảng cách, K chọn thế nào?  
- Naive Bayes: nguyên lý Bayes, giả thiết độc lập  

### ✅ Checklist cơ bản  
- [ ] Cài `scikit-learn` nếu chưa có  
- [ ] Load dataset Iris hoặc Titanic  
- [ ] Train KNN (k=3,5,7), so sánh accuracy  
- [ ] Train GaussianNB & MultinomialNB trên dữ liệu text đơn giản  
- [ ] Tính precision, recall cho cả hai  

### 🚀 Nhiệm vụ nâng cao  
- [ ] Thử KNN với khoảng cách Minkowski (p=1,2,3)  
- [ ] Viết script so sánh KNN & Naive Bayes trên cùng dataset  
- [ ] Đánh giá tốc độ train/predict của cả hai  

### 🧠 Câu hỏi tư duy  
- Tại sao NB vẫn tốt dù feature không độc lập hoàn toàn?  
- KNN có thể dùng cho dữ liệu lớn không?  

### 📝 Nhật ký học tập  
- Ghi k giá trị tốt nhất & kết quả đánh giá  
- Đánh giá ưu/nhược mỗi thuật toán  

---

## 📅 Ngày 9: Decision Tree & Random Forest  
🎯 **Mục tiêu:** Nắm vững cây quyết định và ensemble  

### 📚 Lý thuyết cốt lõi  
- CART, entropy vs Gini  
- Overfitting ở Decision Tree  
- Random Forest: bagging, bootstrap, feature bagging  

### ✅ Checklist cơ bản  
- [ ] Train `DecisionTreeClassifier` trên Titanic/Iris  
- [ ] Vẽ cây (export_graphviz + Graphviz)  
- [ ] Train `RandomForestClassifier` với 100 trees  
- [ ] So sánh accuracy & feature importance  
- [ ] Thử thay đổi `max_depth`, `n_estimators`  

### 🚀 Nhiệm vụ nâng cao  
- [ ] Implement simple bagging thủ công với 5 tree  
- [ ] Viết script tự động plot feature importance  
- [ ] Chạy experiment: RF vs DT trên 3 dataset khác nhau  

### 🧠 Câu hỏi tư duy  
- Tại sao Random Forest ít overfit hơn Decision Tree?  
- Khi nào nên tăng depth vs số cây?  

### 📝 Nhật ký học tập  
- Ghi lại hyper-params tốt nhất & lý do  

---

## 📅 Ngày 10: Support Vector Machine (SVM)  
🎯 **Mục tiêu:** Hiểu & sử dụng SVM cho phân loại  

### 📚 Lý thuyết cốt lõi  
- Margin, support vectors  
- Kernel (linear, RBF, polynomial)  
- C parameter & ảnh hưởng  

### ✅ Checklist cơ bản  
- [ ] Train `SVC(kernel='linear')` trên Iris  
- [ ] Train `SVC(kernel='rbf')`, thử `C` = [0.1,1,10]  
- [ ] Vẽ boundary 2D cho 2 feature  
- [ ] Train `SVR` cho bài toán hồi quy nhỏ  

### 🚀 Nhiệm vụ nâng cao  
- [ ] So sánh runtime giữa kernel khác nhau  
- [ ] Implement grid search tay cho kernel & C  
- [ ] Vẽ ảnh hưởng C vs margin trên diagram  

### 🧠 Câu hỏi tư duy  
- Kernel trick hoạt động thế nào?  
- Khi nào SVM không phù hợp?  

### 📝 Nhật ký học tập  
- Ghi kernel & C tối ưu & kết quả  

---

## 📅 Ngày 11: Clustering (K-Means & Hierarchical)  
🎯 **Mục tiêu:** Áp dụng clustering để khám phá dữ liệu  

### 📚 Lý thuyết cốt lõi  
- K-Means: K chọn sao? Elbow Method  
- Hierarchical: agglomerative vs divisive  

### ✅ Checklist cơ bản  
- [ ] Load dataset 2D (Ví dụ: Iris without nhãn)  
- [ ] Train `KMeans(n_clusters=3)`, vẽ cluster scatter  
- [ ] Compute inertia, vẽ elbow plot  
- [ ] Train `AgglomerativeClustering`, plot dendrogram  

### 🚀 Nhiệm vụ nâng cao  
- [ ] Thử DBSCAN cho dataset có noise  
- [ ] Viết hàm đánh giá Silhouette Score  
- [ ] So sánh kết quả KMeans vs Agglomerative trên 2 dataset  

### 🧠 Câu hỏi tư duy  
- Khi nào dùng KMeans vs DBSCAN?  
- Silhouette Score nói lên gì?  

### 📝 Nhật ký học tập  
- Ghi phương pháp chọn K & interpret cluster  

---

## 📅 Ngày 12: Đánh giá & Tối ưu mô hình  
🎯 **Mục tiêu:** Hiểu cross-validation, bias-variance tradeoff  

### 📚 Lý thuyết cốt lõi  
- K-Fold CV, Stratified CV  
- Bias vs Variance  
- Learning Curve, Validation Curve  

### ✅ Checklist cơ bản  
- [ ] Sử dụng `cross_val_score` cho RF/SVM  
- [ ] Vẽ learning curve với `learning_curve`  
- [ ] Vẽ validation curve cho hyper-param  

### 🚀 Nhiệm vụ nâng cao  
- [ ] Implement StratifiedKFold tay  
- [ ] Viết script plot bias-variance tradeoff diagram  
- [ ] So sánh CV vs simple train/test split  

### 🧠 Câu hỏi tư duy  
- Tại sao cần stratify?  
- CV có thể gây over-optimistic không?  

### 📝 Nhật ký học tập  
- Ghi lại insights từ learning/validation curve  

---

## 📅 Ngày 13: Pipeline & GridSearchCV  
🎯 **Mục tiêu:** Tự động hóa quy trình & tuning  

### 📚 Lý thuyết cốt lõi  
- `Pipeline` trong sklearn  
- GridSearch vs RandomizedSearch  

### ✅ Checklist cơ bản  
- [ ] Xây pipeline: scaler → model  
- [ ] Dùng `GridSearchCV` tìm param tối ưu cho RF/SVM  
- [ ] Dùng `RandomizedSearchCV` cho LightGBM (nếu có)  

### 🚀 Nhiệm vụ nâng cao  
- [ ] Thử `HalvingGridSearchCV` (scikit-learn ≥ 0.24)  
- [ ] Viết wrapper tự động train & report kết quả  
- [ ] Lưu model tốt nhất bằng `joblib`  

### 🧠 Câu hỏi tư duy  
- Khi nào dùng Randomized thay vì Grid?  
- Pipeline có thể chứa feature engineering không?  

### 📝 Nhật ký học tập  
- Ghi param search space & best params  

---

## 📅 Ngày 14: Mini Project – Phân loại ảnh đơn giản  
🎯 **Mục tiêu:** Ứng dụng ML nâng cao cho ảnh (feature từ color histogram)  

### ✅ Checklist cơ bản  
- [ ] Lấy dataset ảnh nhỏ (dog vs cat, hoặc MNIST small)  
- [ ] Trích feature: color histogram / flatten pixel  
- [ ] Train ML model (RF, SVM)  
- [ ] Đánh giá & vẽ confusion matrix  

### 🚀 Nhiệm vụ nâng cao  
- [ ] Thử kết hợp PCA giảm chiều trước khi train  
- [ ] So sánh performance giữa RF, SVM, KNN  

### 🧠 Câu hỏi tư duy  
- Feature nào quan trọng nhất với ảnh đơn giản?  
- PCA có mất thông tin gì?  

---

# 🧠 LỘ TRÌNH HỌC AI – TUẦN 3 (CHUYÊN SÂU & NHIỆM VỤ NÂNG CAO)

## 📦 MỤC TIÊU TUẦN 3
- Nắm vững Deep Learning cơ bản (MLP, CNN, RNN)  
- Hiểu rõ cách build, train, debug mô hình neural network  
- Biết sử dụng TensorFlow/Keras và PyTorch song song  

---

## 📅 Ngày 15: Môi trường Deep Learning & Hello World  
🎯 **Mục tiêu:** Cấu hình & làm quen TF/Keras & PyTorch  

### 📚 Lý thuyết cốt lõi  
- Graph vs Eager Execution  
- TensorFlow 2.x + Keras API  
- PyTorch cơ bản: Tensor, autograd  

### ✅ Checklist cơ bản  
- [ ] Cài `tensorflow` & `torch`  
- [ ] Viết Hello World MLP: 1 hidden layer (TF & PyTorch)  
- [ ] So sánh code syntax giữa 2 framework  

### 🚀 Nhiệm vụ nâng cao  
- [ ] Cài GPU support và kiểm tra CUDA  
- [ ] Viết script chuyển mô hình TF → ONNX → load PyTorch  

### 🧠 Câu hỏi tư duy  
- Eager vs Graph mode khác gì?  
- Kết quả random seed reproducibility làm sao?  

### 📝 Nhật ký học tập  
- Ghi lại lỗi GPU & cách fix  

---

## 📅 Ngày 16: MLP & Backpropagation  
🎯 **Mục tiêu:** Hiểu bản chất MLP & gradient descent  

### 📚 Lý thuyết cốt lõi  
- Forward pass, loss function  
- Backpropagation algorithm  
- Learning rate ảnh hưởng ra sao  

### ✅ Checklist cơ bản  
- [ ] Xây MLP 2 hidden layer trên MNIST (TF & PyTorch)  
- [ ] Train 1 epoch, in loss & accuracy  
- [ ] Plot loss curve theo epoch  

### 🚀 Nhiệm vụ nâng cao  
- [ ] Implement from-scratch 1 layer backprop bằng numpy  
- [ ] Thử learning rate scheduling (Step, Exponential)  

### 🧠 Câu hỏi tư duy  
- Vanishing gradient là gì?  
- Khi nào nên dùng batch vs stochastic gradient descent?  

### 📝 Nhật ký học tập  
- Ghi insight từ loss curve  

---

## 📅 Ngày 17: Convolutional Neural Network (CNN)  
🎯 **Mục tiêu:** Xây & train CNN cơ bản  

### 📚 Lý thuyết cốt lõi  
- Convolution, pooling, padding  
- Stride, kernel size ảnh hưởng thế nào  

### ✅ Checklist cơ bản  
- [ ] Xây CNN đơn giản (Conv→ReLU→Pool→FC) trên CIFAR-10/MNIST  
- [ ] Train 5 epochs, evaluate test accuracy  
- [ ] Vẽ training & validation curve  

### 🚀 Nhiệm vụ nâng cao  
- [ ] Implement data augmentation (flip, rotate)  
- [ ] Thử thêm BatchNorm & Dropout  

### 🧠 Câu hỏi tư duy  
- Vì sao pooling cần thiết?  
- Augmentation ảnh hưởng gì tới generalization?  

### 📝 Nhật ký học tập  
- Ghi lại cấu trúc model & kết quả  

---

## 📅 Ngày 18: Data Augmentation & Callbacks  
🎯 **Mục tiêu:** Tăng cường dữ liệu & quản lý training  

### 📚 Lý thuyết cốt lõi  
- Overfitting vs Augmentation  
- Callback: EarlyStopping, ModelCheckpoint  

### ✅ Checklist cơ bản  
- [ ] Thiết lập `ImageDataGenerator` (Keras)  
- [ ] Cài EarlyStopping & Save best model  
- [ ] Train với augmentation, compare accuracy  

### 🚀 Nhiệm vụ nâng cao  
- [ ] Tự viết augmentation function bằng albumentations  
- [ ] Viết callback custom (print LR mỗi epoch)  

### 🧠 Câu hỏi tư duy  
- Khi nào dừng training?  
- Save best model dựa trên metric nào?  

### 📝 Nhật ký học tập  
- Ghi lại ảnh hưởng augmentation  

---

## 📅 Ngày 19: Recurrent Neural Network (RNN/LSTM)  
🎯 **Mục tiêu:** Hiểu & áp dụng RNN cho chuỗi thời gian  

### 📚 Lý thuyết cốt lõi  
- RNN cell, vanishing/exploding gradient  
- LSTM/GRU giải quyết vấn đề gì  

### ✅ Checklist cơ bản  
- [ ] Xây RNN/LSTM đơn giản trên dataset chuỗi số (sinh dữ liệu sine)  
- [ ] Train 1 epoch, plot loss  
- [ ] So sánh RNN vs LSTM accuracy  

### 🚀 Nhiệm vụ nâng cao  
- [ ] Dự đoán chuỗi thời gian giá cổ phiếu (dữ liệu mẫu)  
- [ ] Viết custom LSTM cell bằng PyTorch  

### 🧠 Câu hỏi tư duy  
- Khi nào dùng LSTM vs GRU?  
- Sequence length ảnh hưởng thế nào?  

### 📝 Nhật ký học tập  
- Ghi lại khó khăn khi train chuỗi  

---

## 📅 Ngày 20: Transfer Learning (CNN Pretrained)  
🎯 **Mục tiêu:** Sử dụng mô hình CNN pretrained cho task nhỏ  

### 📚 Lý thuyết cốt lõi  
- Fine-tuning vs Feature Extraction  
- Pretrained trên ImageNet  

### ✅ Checklist cơ bản  
- [ ] Load ResNet50/VGG16 pretrained (Keras)  
- [ ] Freeze layers, train head trên dataset nhỏ  
- [ ] Unfreeze 1 block cuối, fine-tune  

### 🚀 Nhiệm vụ nâng cao  
- [ ] Triển khai model sang TensorFlow Lite  
- [ ] So sánh accuracy & speed inference  

### 🧠 Câu hỏi tư duy  
- Khi nào nên fine-tune toàn bộ model?  
- Pretrained model có bias không?  

### 📝 Nhật ký học tập  
- Ghi lại kết quả before/after fine-tuning  

---

## 📅 Ngày 21: Mini Project – Nhận diện số viết tay với CNN  
🎯 **Mục tiêu:** Hoàn thiện end-to-end CNN project  

### ✅ Checklist cơ bản
- [ ] Load MNIST, tạo DataLoader  
- [ ] Xây model CNN, train 10 epochs  
- [ ] Đánh giá & vẽ confusion matrix  

### 🚀 Nhiệm vụ nâng cao
- [ ] Implement live demo: webcam capture số viết tay → predict  
- [ ] Đóng gói model & demo thành Flask app  

### 🧠 Câu hỏi tư duy
- Làm sao deploy model lên edge device?  
- Làm thế nào tối ưu inference speed?  

---

🎉 **Hoàn thành Tuần 2 & Tuần 3!**  
Bạn đã có nền tảng ML nâng cao và Deep Learning cơ bản – sẵn sàng cho Tuần 4 (CV, NLP, MLOps…).  
Chúc bạn học tập hăng say và hãy tick ✔ mỗi khi hoàn thành!  
# 🧠 LỘ TRÌNH HỌC AI – TUẦN 4 (ỨNG DỤNG CV, NLP & TRIỂN KHAI)

## 📦 MỤC TIÊU TUẦN 4
- Ứng dụng Computer Vision nâng cao (object detection, segmentation)  
- Ứng dụng NLP cơ bản & nâng cao (embeddings, transformers)  
- Triển khai mô hình thành API & tích hợp vào web/mobile  

---

## 📅 Ngày 22: Object Detection với YOLO / SSD  
🎯 **Mục tiêu:** Hiểu & triển khai object detection  

### ✅ Checklist cơ bản
- [ ] Cài `opencv-python`, `torchvision`, `ultralytics`  
- [ ] Load model YOLOv5/V7 pretrained  
- [ ] Chạy inference trên ảnh sample, hiển thị bounding boxes  
- [ ] Thay đổi confidence threshold & NMS  

### 🚀 Nhiệm vụ nâng cao
- [ ] Fine-tune YOLO trên dataset custom (5–10 ảnh)  
- [ ] Viết script tự động crop & lưu từng object detected  

---

## 📅 Ngày 23: Semantic Segmentation (UNet / Mask R-CNN)  
🎯 **Mục tiêu:** Phân đoạn ảnh pixel-level  

### ✅ Checklist cơ bản
- [ ] Cài `segmentation_models_pytorch` hoặc `detectron2`  
- [ ] Load UNet pretrained trên dataset medical/COCO  
- [ ] Inference & visualize mask  

### 🚀 Nhiệm vụ nâng cao
- [ ] Fine-tune UNet cho bài toán segmentation nhỏ (ví dụ: phân đoạn lá cây)  
- [ ] Tạo notebook so sánh performance UNet vs Mask R-CNN  

---

## 📅 Ngày 24: NLP – Embeddings & Transformers cơ bản  
🎯 **Mục tiêu:** Hiểu word embedding & mô hình Transformers  

### ✅ Checklist cơ bản
- [ ] Cài `transformers`, `tokenizers`  
- [ ] Load `bert-base-uncased`: tokenize & embed câu mẫu  
- [ ] Tính cosine-similarity giữa 2 câu  

### 🚀 Nhiệm vụ nâng cao
- [ ] Fine-tune BERT nhỏ (DistilBERT) cho bài sentiment analysis (IMDB)  
- [ ] Viết script đo tốc độ inference & so sánh CPU vs GPU  

---

## 📅 Ngày 25: NLP – Sequence-to-Sequence & Summarization  
🎯 **Mục tiêu:** Áp dụng seq2seq cho tóm tắt văn bản  

### ✅ Checklist cơ bản
- [ ] Load `sshleifer/distilbart-cnn-12-6` từ HuggingFace  
- [ ] Tóm tắt 1 bài báo ngắn (~300 từ)  
- [ ] So sánh length input vs output  

### 🚀 Nhiệm vụ nâng cao
- [ ] Fine-tune model với dataset tóm tắt tiếng Việt (nếu có)  
- [ ] Triển khai demo simple web form để thử summarization  

---

## 📅 Ngày 26: Triển khai mô hình thành API (Flask / FastAPI)  
🎯 **Mục tiêu:** Xây RESTful API cho mô hình AI  

### ✅ Checklist cơ bản
- [ ] Tạo FastAPI project, cài `uvicorn`  
- [ ] Load model (CV/NLP) trong endpoint `/predict`  
- [ ] Test với `curl` / Postman  

### 🚀 Nhiệm vụ nâng cao
- [ ] Thêm middleware logging request/response  
- [ ] Containerize API với Docker & chạy local  

---

## 📅 Ngày 27: Kết nối Frontend (React / HTML)  
🎯 **Mục tiêu:** Tích hợp API AI vào giao diện người dùng  

### ✅ Checklist cơ bản
- [ ] Tạo page React cơ bản (CRA hoặc Next.js)  
- [ ] Gọi API `/predict` & hiển thị kết quả  
- [ ] Thêm spinner loading khi đợi response  

### 🚀 Nhiệm vụ nâng cao
- [ ] Thiết kế UI mobile-responsive  
- [ ] Triển khai frontend & backend lên Heroku / Netlify  

---

## 📅 Ngày 28: Mini Project – Chatbot AI đa tác vụ  
🎯 **Mục tiêu:** Kết hợp CV + NLP + API  

### ✅ Checklist cơ bản
- [ ] Xây chatbot nhận ảnh & văn bản  
- [ ] Nếu nhận ảnh: chạy object detection & trả labels  
- [ ] Nếu nhận văn bản: chạy sentiment analysis hoặc summarization  

### 🚀 Nhiệm vụ nâng cao
- [ ] Thêm stateful conversation (lưu context)  
- [ ] Triển khai lên Telegram/Messenger bằng webhook  

---

# 🧠 LỘ TRÌNH HỌC AI – TUẦN 5 (MLOPS & TRIỂN KHAI SẢN PHẨM)

## 📦 MỤC TIÊU TUẦN 5
- Hiểu & áp dụng MLOps: CI/CD, Docker, Kubernetes  
- Tối ưu & giám sát mô hình production  
- Đóng gói, versioning, logging, monitoring  

---

## 📅 Ngày 29: Docker & Containerization  
🎯 **Mục tiêu:** Đóng gói ứng dụng AI vào Docker  

### ✅ Checklist cơ bản
- [ ] Viết `Dockerfile` cho Flask/FastAPI app  
- [ ] Build image & chạy container local  
- [ ] Push image lên Docker Hub  

### 🚀 Nhiệm vụ nâng cao
- [ ] Multi-stage build để giảm size image  
- [ ] Tạo `docker-compose` with API + Redis cache  

---

## 📅 Ngày 30: CI/CD cho Mô hình AI  
🎯 **Mục tiêu:** Tự động hóa pipeline deploy  

### ✅ Checklist cơ bản
- [ ] Cấu hình GitHub Actions / GitLab CI  
- [ ] Build & test notebook/script tự động  
- [ ] Deploy Docker image lên staging server  

### 🚀 Nhiệm vụ nâng cao
- [ ] Thêm unit test cho endpoint `/predict`  
- [ ] Triển khai blue/green deployment  

---

## 📅 Ngày 31: Kubernetes cơ bản  
🎯 **Mục tiêu:** Chạy ứng dụng AI trên K8s cluster  

### ✅ Checklist cơ bản
- [ ] Cài `kubectl`, minikube hoặc Docker Desktop K8s  
- [ ] Viết `Deployment` & `Service` YAML  
- [ ] Deploy API container lên cluster  

### 🚀 Nhiệm vụ nâng cao
- [ ] Cài HPA (Horizontal Pod Autoscaler)  
- [ ] Cấu hình Ingress để expose dịch vụ  

---

## 📅 Ngày 32: Monitoring & Logging  
🎯 **Mục tiêu:** Giám sát sức khỏe & hiệu năng model  

### ✅ Checklist cơ bản
- [ ] Cài Prometheus & Grafana (minikube addon)  
- [ ] Expose metrics từ FastAPI (Prometheus client)  
- [ ] Tạo dashboard Grafana xem latency, error rate  

### 🚀 Nhiệm vụ nâng cao
- [ ] Thiết lập alert (email/Slack) khi error rate cao  
- [ ] Lưu logs model inference vào ELK stack  

---

## 📅 Ngày 33: Model Versioning & Registry  
🎯 **Mục tiêu:** Quản lý phiên bản model khoa học  

### ✅ Checklist cơ bản
- [ ] Cài MLflow hoặc DVC  
- [ ] Track experiment: params, metrics, artifacts  
- [ ] Publish model lên MLflow registry  

### 🚀 Nhiệm vụ nâng cao
- [ ] Tạo UI so sánh các phiên bản model  
- [ ] Tích hợp versioning vào CI/CD  

---

## 📅 Ngày 34: AutoML & Hyperparameter Tuning  
🎯 **Mục tiêu:** Thử nghiệm AutoML & tuning tự động  

### ✅ Checklist cơ bản
- [ ] Thử `sklearn.model_selection.RandomizedSearchCV`  
- [ ] Thử `Optuna` hoặc `Ray Tune`  
- [ ] Track results bằng MLflow  

### 🚀 Nhiệm vụ nâng cao
- [ ] Viết optimizer custom với Optuna  
- [ ] So sánh AutoML vs manual tuning  

---

## 📅 Ngày 35: Mini Project – MLOps Pipeline End-to-End  
🎯 **Mục tiêu:** Tạo pipeline hoàn chỉnh từ code → deploy → monitoring  

### ✅ Checklist cơ bản
- [ ] Code model + API + Dockerfile  
- [ ] CI/CD tự động build & deploy staging  
- [ ] Giám sát cơ bản với Prometheus  

### 🚀 Nhiệm vụ nâng cao
- [ ] Tạo blueprint triển khai production (blue/green)  
- [ ] Viết báo cáo MLOps cho project  

---

# 🧠 LỘ TRÌNH HỌC AI – TUẦN 6 (NGHIÊN CỨU & XU HƯỚNG MỚI)

## 📦 MỤC TIÊU TUẦN 6
- Khám phá Học tự giám sát, Diffusion models, LLMs  
- Đọc & hiểu paper, viết note research  
- Thực hành state-of-the-art AI  

---

## 📅 Ngày 36: Học tự giám sát (Self-Supervised Learning)  
🎯 **Mục tiêu:** Nắm ý tưởng & giải thuật cơ bản  

### ✅ Checklist cơ bản
- [ ] Đọc bài “A Simple Framework for Contrastive Learning of Visual Representations” (SimCLR)  
- [ ] Cài `lightly` hoặc `solo-learn`  
- [ ] Train SimCLR trên CIFAR-10 (subset)  

### 🚀 Nhiệm vụ nâng cao
- [ ] Implement tiêu đề loss function contrastive bằng PyTorch  
- [ ] So sánh feature quality trước/sau contrastive pre-training  

---

## 📅 Ngày 37: Diffusion Models cơ bản  
🎯 **Mục tiêu:** Hiểu nguyên lý diffusion & sampling  

### ✅ Checklist cơ bản
- [ ] Đọc paper “Denoising Diffusion Probabilistic Models”  
- [ ] Cài `diffusers` (HuggingFace)  
- [ ] Generate 10 ảnh MNIST với DDPM  

### 🚀 Nhiệm vụ nâng cao
- [ ] Fine-tune Stable Diffusion nhỏ cho theme custom  
- [ ] Viết script visualize forward/backward process  

---

## 📅 Ngày 38: Large Language Models (LLMs) – Khả năng & Giới hạn  
🎯 **Mục tiêu:** Hiểu kiến trúc & ứng dụng LLM  

### ✅ Checklist cơ bản
- [ ] Đọc blog “Attention Is All You Need” tóm tắt  
- [ ] Load GPT-2 small với `transformers`  
- [ ] Sinh văn bản ngắn (100 từ) cho prompt  

### 🚀 Nhiệm vụ nâng cao
- [ ] Fine-tune GPT-2 small trên dữ liệu domain  
- [ ] Đánh giá perplexity & human eval  

---

## 📅 Ngày 39: Retrieval-Augmented Generation (RAG)  
🎯 **Mục tiêu:** Kết hợp LLM + vector DB để trả lời kiến thức  

### ✅ Checklist cơ bản
- [ ] Cài `faiss` hoặc `pinecone-client`  
- [ ] Index 100 doc text sample  
- [ ] Fetch & trả về context + generate câu trả lời  

### 🚀 Nhiệm vụ nâng cao
- [ ] Tạo pipeline RAG end-to-end (ingest → index → query)  
- [ ] So sánh performance với/không có retrieval  

---

## 📅 Ngày 40: Viết & Đánh giá Research Paper  
🎯 **Mục tiêu:** Tập thói quen đọc & note paper  

### ✅ Checklist cơ bản
- [ ] Chọn 1 paper SOTA gần đây (ArXiv last week)  
- [ ] Đọc & tóm tắt: mục tiêu, phương pháp, kết quả  
- [ ] Viết blog post 500–800 từ  

### 🚀 Nhiệm vụ nâng cao
- [ ] Triển khai code re-implement core idea paper  
- [ ] So sánh kết quả với báo cáo paper  

---

## 📅 Ngày 41–42: Tổng kết & Lên Kế hoạch Tiếp Theo  
🎯 **Mục tiêu:** Đánh giá & định hướng nghiên cứu hoặc sản phẩm  

### ✅ Checklist cơ bản
- [ ] Viết report cả 6 tuần: kiến thức & project  
- [ ] Đánh giá strengths & weaknesses  
- [ ] Xác định hướng chuyên sâu: CV, NLP, RL, MLOps…  
- [ ] Lập roadmap 3–6 tháng tiếp theo  

### 🚀 Nhiệm vụ nâng cao
- [ ] Chuẩn bị proposal nghiên cứu hoặc demo sản phẩm  
- [ ] Kết nối mentor / cộng đồng AI phù hợp  

---

🎉 **Chúc mừng** bạn đã hoàn thành 6 tuần lộ trình AI chuyên sâu! Tick ✔ và tiếp tục hành trình chinh phục AI nâng cao!   ```

