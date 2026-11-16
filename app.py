import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib
from utils import clean_text

st.set_page_config(page_title="Phân loại cảm xúc sinh viên", page_icon="🎓", layout="wide")

# ================= GIAO DIỆN (CSS) =================
st.markdown("""
<style>
.header {
    background: linear-gradient(135deg, #4f46e5, #06b6d4);
    padding: 28px 24px; border-radius: 16px; color: white; text-align: center;
    margin-bottom: 20px;
}
.header h1 { margin: 0; font-size: 2.1rem; color: white; }
.header p { margin: 6px 0 0 0; opacity: 0.92; }
.card {
    padding: 22px; border-radius: 14px; text-align: center; margin: 14px 0;
    border: 2px solid; font-size: 1.7rem; font-weight: 700;
}
.neg { background: #fee2e2; border-color: #ef4444; color: #b91c1c; }
.neu { background: #fef3c7; border-color: #f59e0b; color: #b45309; }
.pos { background: #dcfce7; border-color: #22c55e; color: #15803d; }
.small { font-size: 0.9rem; font-weight: 400; opacity: 0.85; margin-top: 4px; }
.chip {
    display: inline-block; padding: 4px 12px; margin: 3px 4px 3px 0;
    border-radius: 999px; background: #e0e7ff; color: #3730a3;
    font-size: 0.9rem; font-weight: 600;
}
</style>
<div class="header">
    <h1>🎓 Hệ thống phân loại cảm xúc phản hồi sinh viên</h1>
    <p>Đồ án Nhập môn học máy • Dữ liệu: uitnlp/vietnamese_students_feedback</p>
</div>
""", unsafe_allow_html=True)

# ================= TẢI MÔ HÌNH =================
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

NHAN = {
    0: ("Tiêu cực", "😞", "neg"),
    1: ("Trung tính", "😐", "neu"),
    2: ("Tích cực", "😊", "pos"),
}

if "history" not in st.session_state:
    st.session_state["history"] = []
if "text" not in st.session_state:
    st.session_state["text"] = ""

# ================= HÀM HỖ TRỢ =================
def diem_so(x):
    """Trả về điểm tương đối (tổng = 1) cho từng nhãn."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[0]
    s = model.decision_function(x)[0]
    e = np.exp(s - np.max(s))
    return e / e.sum()

def tu_khoa(x, lop, top=6):
    """Các từ đóng góp nhiều nhất cho nhãn dự đoán (chỉ cho Pipeline TF-IDF + mô hình tuyến tính)."""
    vec, clf = model[0], model[-1]
    X = vec.transform(x)
    ten = vec.get_feature_names_out()
    i = list(clf.classes_).index(lop)
    dong_gop = np.asarray(X.multiply(clf.coef_[i]).todense()).ravel()
    idx = np.argsort(dong_gop)[::-1][:top]
    return [(ten[j].replace("_", " "), dong_gop[j]) for j in idx if dong_gop[j] > 0]

# ================= THANH BÊN =================
with st.sidebar:
    st.header("ℹ️ Thông tin mô hình")
    try:
        st.write("**Vector hóa:**", type(model[0]).__name__)
        st.write("**Thuật toán:**", type(model[-1]).__name__)
    except Exception:
        st.write("**Mô hình:**", type(model).__name__)
    st.write("**Số nhãn:** 3 (Tiêu cực, Trung tính, Tích cực)")
    st.divider()
    st.header("🕘 Lịch sử phân tích")
    if st.session_state["history"]:
        for h in reversed(st.session_state["history"][-8:]):
            st.write(f'{h["icon"]} {h["text"][:45]}...')
        if st.button("Xóa lịch sử"):
            st.session_state["history"] = []
            st.rerun()
    else:
        st.caption("Chưa có câu nào.")

# ================= CÁC TAB =================
tab1, tab2, tab3, tab4 = st.tabs(
    ["🔍 Phân tích 1 câu", "📂 Phân tích hàng loạt", "📊 Kết quả thực nghiệm", "📘 Giới thiệu"]
)

# ---------- TAB 1 ----------
with tab1:
    MAU = [
        "Giảng viên dạy rất nhiệt tình và dễ hiểu",
        "Môn học bình thường, không có gì đặc biệt",
        "Thầy cô dạy quá nhanh, em không theo kịp",
        "Cơ sở vật chất cũ, phòng học rất nóng",
    ]

    def dien_mau(c):
        st.session_state["text"] = c

    st.markdown("**Thử nhanh với câu mẫu:**")
    cols = st.columns(4)
    for i, c in enumerate(MAU):
        cols[i].button(c, key=f"mau{i}", on_click=dien_mau, args=(c,), use_container_width=True)

    text = st.text_area("Nhập phản hồi của sinh viên:", key="text", height=120,
                        placeholder="Ví dụ: Giảng viên dạy rất hay...")

    if st.button("🔍 Phân tích cảm xúc", type="primary", use_container_width=True):
        if text.strip() == "":
            st.warning("Bạn chưa nhập nội dung.")
        else:
            x = [clean_text(text)]
            kq = int(model.predict(x)[0])
            ten, icon, css = NHAN[kq]
            st.session_state["history"].append({"text": text, "icon": icon})

            c1, c2 = st.columns([1, 1])
            with c1:
                st.markdown(
                    f'<div class="card {css}">{icon} {ten}'
                    f'<div class="small">Kết quả dự đoán của mô hình</div></div>',
                    unsafe_allow_html=True,
                )
                st.caption("Câu sau khi tiền xử lý: " + x[0])
            with c2:
                try:
                    diem = diem_so(x)
                    st.markdown("**Điểm tin cậy tương đối từng nhãn:**")
                    for lop, d in zip(model.classes_, diem):
                        t, ic, _ = NHAN[int(lop)]
                        st.progress(float(d), text=f"{ic} {t}: {d*100:.1f}%")
                except Exception:
                    pass

            try:
                kw = tu_khoa(x, kq)
                if kw:
                    st.markdown("**Từ khóa ảnh hưởng nhất đến kết quả:**")
                    st.markdown("".join(f'<span class="chip">{w}</span>' for w, _ in kw),
                                unsafe_allow_html=True)
            except Exception:
                pass

# ---------- TAB 2 ----------
with tab2:
    st.markdown("Phân tích nhiều câu cùng lúc. Chọn một trong hai cách nhập.")
    cach = st.radio("Cách nhập", ["Dán nhiều câu", "Tải file CSV"], horizontal=True)

    ds_cau = []
    if cach == "Dán nhiều câu":
        raw = st.text_area("Mỗi dòng là một câu phản hồi:", height=200,
                           placeholder="Giảng viên dạy rất hay\nMôn học quá khó\n...")
        ds_cau = [l.strip() for l in raw.split("\n") if l.strip()]
    else:
        f = st.file_uploader("Chọn file CSV", type=["csv"])
        if f is not None:
            df_in = pd.read_csv(f)
            cot = st.selectbox("Chọn cột chứa câu phản hồi", df_in.columns)
            ds_cau = df_in[cot].dropna().astype(str).tolist()

    if st.button("🚀 Phân tích tất cả", type="primary", use_container_width=True):
        if not ds_cau:
            st.warning("Chưa có câu nào để phân tích.")
        else:
            with st.spinner("Đang phân tích..."):
                preds = model.predict([clean_text(t) for t in ds_cau])
            kq_df = pd.DataFrame({
                "Câu phản hồi": ds_cau,
                "Cảm xúc": [NHAN[int(p)][0] for p in preds],
            })
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Tổng số câu", len(kq_df))
            for col, k in zip([m2, m3, m4], [2, 1, 0]):
                n = int((kq_df["Cảm xúc"] == NHAN[k][0]).sum())
                col.metric(f"{NHAN[k][1]} {NHAN[k][0]}", n)

            st.bar_chart(kq_df["Cảm xúc"].value_counts())
            st.dataframe(kq_df, use_container_width=True)
            st.download_button("⬇️ Tải kết quả (CSV)",
                               kq_df.to_csv(index=False).encode("utf-8-sig"),
                               file_name="ket_qua_phan_tich.csv", mime="text/csv")

# ---------- TAB 3 ----------
with tab3:
    st.subheader("Bảng so sánh các mô hình")
    if os.path.exists("ket_qua.csv"):
        kq = pd.read_csv("ket_qua.csv")
        st.dataframe(kq, use_container_width=True)
        tot = kq.sort_values("F1_val", ascending=False).iloc[0]
        st.success(f'Tổ hợp tốt nhất theo F1 trên tập validation: '
                   f'**{tot["Features"]} + {tot["Model"]}** (F1 test = {tot["F1_test"]})')
    else:
        st.info("Chưa có file ket_qua.csv trong thư mục dự án.")

    anh = [
        ("so_sanh_f1.png", "So sánh F1 giữa các phương pháp vector hóa"),
        ("confusion_matrix.png", "Ma trận nhầm lẫn của mô hình tốt nhất"),
        ("phan_phoi_nhan.png", "Phân phối nhãn cảm xúc"),
        ("do_dai_cau.png", "Phân phối độ dài câu"),
    ]
    cols = st.columns(2)
    k = 0
    for fn, cap in anh:
        if os.path.exists(fn):
            cols[k % 2].image(fn, caption=cap, use_container_width=True)
            k += 1

# ---------- TAB 4 ----------
with tab4:
    st.subheader("Về đề tài")
    st.write("Bài toán phân loại cảm xúc của phản hồi sinh viên tiếng Việt thành 3 nhãn: "
             "**Tiêu cực**, **Trung tính**, **Tích cực**.")
    st.subheader("Quy trình học máy")
    st.markdown("""
1. **Khám phá dữ liệu (EDA):** cấu trúc, phân phối nhãn, độ dài câu
2. **Tiền xử lý văn bản:** chữ thường, bỏ ký tự đặc biệt, tách từ tiếng Việt
3. **Vector hóa:** Bag-of-Words, TF-IDF, Sentence Embeddings
4. **Huấn luyện:** KNN, Naive Bayes, SVM, Random Forest
5. **Đánh giá:** Accuracy, Precision, Recall, F1, ma trận nhầm lẫn
6. **Triển khai:** ứng dụng web Streamlit này
""")