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

**Lưu ý cách sử dụng**  
- Copy toàn bộ vào file `Week1_Advanced.md`  
- Mỗi ngày tạo folder `dayX/` chứa:  
  - `dayX_practice.ipynb`  
  - `dayX_log.md`  
- Với Project ngày 7, tạo thư mục `mini_project/`  

Chúc bạn học tập hiệu quả và hãy tick ✔ mỗi khi hoàn thành!
