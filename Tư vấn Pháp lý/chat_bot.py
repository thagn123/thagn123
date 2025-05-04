import os
import re
import csv
from datetime import datetime
import PyPDF2
from docx import Document
from pdf2docx import Converter
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np

###########################################
# PHẦN 1: CHUYỂN ĐỔI VÀ TRÍCH XUẤT VĂN BẢN TỪ THƯ MỤC data_cawl
###########################################

DATA_FOLDER = "data_cawl"
EXTRACTED_TEXT_FILE = os.path.join(DATA_FOLDER, "extracted_data.txt")


def convert_pdf_to_docx(pdf_path, docx_path):
    """Chuyển đổi file PDF sang DOCX để trích xuất nội dung."""
    try:
        cv = Converter(pdf_path)
        cv.convert(docx_path, start=0, end=None)
        cv.close()
        print(f"✅ Đã chuyển đổi PDF sang DOCX: {docx_path}")
    except Exception as e:
        print(f"⚠️ Lỗi khi chuyển đổi PDF {pdf_path}: {e}")


def extract_text_from_docx(docx_path):
    """Trích xuất nội dung từ file DOCX."""
    text = ""
    try:
        doc = Document(docx_path)
        for para in doc.paragraphs:
            if para.text.strip():
                text += para.text.strip() + "\n"
        print(f"✅ Đã trích xuất nội dung từ: {docx_path}")
    except Exception as e:
        print(f"⚠️ Lỗi khi đọc DOCX {docx_path}: {e}")
    return text


def process_all_documents(folder_path):
    """
    Duyệt qua thư mục data_cawl, chuyển PDF sang DOCX (nếu cần)
    và trích xuất nội dung từ các file DOCX.
    """
    all_texts = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        # Nếu là PDF thì chuyển sang DOCX trước
        if filename.lower().endswith(".pdf"):
            docx_path = file_path[:-4] + ".docx"  # thay .pdf thành .docx
            convert_pdf_to_docx(file_path, docx_path)
            all_texts.append(extract_text_from_docx(docx_path))
        elif filename.lower().endswith(".docx"):
            all_texts.append(extract_text_from_docx(file_path))
    return "\n".join(all_texts)


def save_extracted_text(text, filename=EXTRACTED_TEXT_FILE):
    """Lưu văn bản đã trích xuất vào file TXT."""
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"✅ Đã lưu văn bản trích xuất vào: {filename}")
    except Exception as e:
        print(f"⚠️ Lỗi khi lưu văn bản: {e}")


def load_extracted_text(filename=EXTRACTED_TEXT_FILE):
    """Tải văn bản đã trích xuất từ file TXT."""
    try:
        with open(filename, "r", encoding="utf-8") as f:
            text = f.read()
        print(f"✅ Đã tải nội dung từ: {filename}")
        return text
    except Exception as e:
        print(f"⚠️ Lỗi khi tải văn bản từ file: {e}")
        return ""


# Nếu file văn bản đã có, tải trực tiếp từ file; nếu không, xử lý gốc và lưu lại.
if os.path.exists(EXTRACTED_TEXT_FILE):
    combined_text = load_extracted_text()
else:
    combined_text = process_all_documents(DATA_FOLDER)
    save_extracted_text(combined_text)


###########################################
# PHẦN 2: TIỀN XỬ LÝ VĂN BẢN – CHIA THÀNH CÁC ĐOẠN
###########################################

def split_into_paragraphs(text, min_length=50):
    """Chia văn bản thành các đoạn có ý nghĩa, bỏ qua các đoạn quá ngắn."""
    paragraphs = [p.strip() for p in text.split("\n") if len(p.strip()) >= min_length]
    return paragraphs


paragraphs = split_into_paragraphs(combined_text)
print(f"📜 Số đoạn văn bản pháp luật trích xuất: {len(paragraphs)}")

###########################################
# PHẦN 3: TẠO EMBEDDINGS CHO CÁC ĐOẠN VĂN BẢN
###########################################

embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
print("\n Đang tạo embeddings cho văn bản pháp luật (quá trình có thể mất vài giây)...")
paragraph_embeddings = embed_model.encode(paragraphs, convert_to_tensor=True)
print(" Hoàn thành việc tạo embeddings.")


###########################################
# PHẦN 4: TRÍCH XUẤT THÔNG TIN 'ĐIỀU'
###########################################

def extract_article_info(text):
    """
    Sử dụng regex để tìm kiếm mẫu như "Điều 15" hoặc "Điều15.1" trong đoạn văn.
    Trả về thông tin 'Điều' nếu tìm thấy, ngược lại trả về None.
    """
    pattern = r'Điều\s*\d+([.,]\d+)?'
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(0)
    return None


###########################################
# PHẦN 5: CẬP NHẬT CÂU HỎI – CÂU TRẢ LỜI VÀO FILE CSV (QA DATABASE)
###########################################

QA_DB_FILE = "qa_database.csv"


def load_qa_database(filename):
    """Đọc dữ liệu từ file CSV nếu tồn tại."""
    data = []
    if os.path.exists(filename):
        with open(filename, mode='r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                data.append(row)
    return data


def save_qa_database(filename, data):
    """Ghi danh sách các câu hỏi – câu trả lời vào file CSV."""
    fieldnames = ['question', 'answer', 'created_at', 'updated_at']
    with open(filename, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(row)


def update_qa_database(filename, new_question, new_answer):
    """
    Nếu câu hỏi đã có (so sánh không phân biệt hoa thường), cập nhật câu trả lời và thời gian;
    Nếu chưa có thì thêm mới.
    """
    data = load_qa_database(filename)
    found = False
    for entry in data:
        if entry['question'].strip().lower() == new_question.strip().lower():
            entry['answer'] = new_answer
            entry['updated_at'] = datetime.now().isoformat()
            found = True
            break
    if not found:
        now_str = datetime.now().isoformat()
        data.append({
            'question': new_question,
            'answer': new_answer,
            'created_at': now_str,
            'updated_at': now_str
        })
    save_qa_database(filename, data)
    return data


###########################################
# PHẦN 6: TRUY XUẤT THÔNG TIN ĐẦY ĐỦ TỪ VĂN BẢN PHÁP LUẬT
###########################################

def retrieve_law_info(query, doc_threshold=0.5, top_k=3):
    """
    Nhận câu hỏi từ người dùng, chuyển đổi thành embedding và tính toán
    cosine similarity với các đoạn văn bản pháp luật. Sau đó, lấy top_k đoạn
    có điểm cao nhưng ghép chúng lại thành _một_ câu trả lời duy nhất.
    """
    query_embedding = embed_model.encode(query, convert_to_tensor=True)
    cosine_scores = util.cos_sim(query_embedding, paragraph_embeddings)[0]
    top_scores, top_indices = torch.topk(cosine_scores, k=top_k)

    # Lọc các đoạn có điểm cao vượt ngưỡng
    filtered_paragraphs = [paragraphs[idx] for score, idx in zip(top_scores, top_indices) if
                           score.item() >= doc_threshold]

    if not filtered_paragraphs:
        return "⚠️ Xin lỗi, tôi không tìm thấy thông tin phù hợp trong các văn bản pháp luật."

    # Ghép lại các đoạn đã lọc thành một câu trả lời duy nhất
    aggregated_answer = "\n\n".join(filtered_paragraphs)
    final_answer = "📜 [Thông tin từ văn bản pháp luật]\n" + aggregated_answer

    # Cập nhật vào Q&A database
    update_qa_database(QA_DB_FILE, query, final_answer)
    return final_answer


###########################################
# PHẦN 7: GIAO DIỆN CHATBOT
###########################################

def chatbot_loop():
    print("\n🤖 Chào mừng bạn đến với Chatbot tư vấn pháp luật!")
    print("Gõ 'exit' hoặc 'thoát' để kết thúc cuộc trò chuyện.")

    while True:
        user_query = input("\n📝 Bạn: ").strip()
        if user_query.lower() in ['exit', 'thoát']:
            print("👋 Chatbot: Cảm ơn bạn đã sử dụng dịch vụ!")
            break
        # Lấy ra _một_ câu trả lời duy nhất, đầy đủ và ghép từ các đoạn liên quan
        answer = retrieve_law_info(user_query)
        print(f"\n💡 Chatbot:\n{answer}")


if __name__ == "__main__":
    chatbot_loop()