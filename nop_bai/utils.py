import re
from underthesea import word_tokenize

def clean_text(text):
    text = str(text).lower()                  # chuyển chữ thường
    text = re.sub(r"[^\w\s]", " ", text)      # bỏ dấu câu, ký tự đặc biệt
    text = re.sub(r"\d+", " ", text)          # bỏ chữ số
    text = re.sub(r"\s+", " ", text).strip()  # bỏ khoảng trắng thừa
    return word_tokenize(text, format="text") # tách từ tiếng Việt