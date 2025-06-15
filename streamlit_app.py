import os
from dotenv import load_dotenv
import google.generativeai as genai
import streamlit as st
import pandas as pd
import torch
import clip
from PIL import Image
import numpy as np
import requests
from io import BytesIO

load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=google_api_key)

# --- Tải dữ liệu phim ---
@st.cache_data
def load_movie_data():
    df = pd.read_csv("imdb_top_1000.csv")
    return df[["Series_Title", "Poster_Link", "Genre", "Runtime", "IMDB_Rating"]]

df = load_movie_data()

# --- Tải model CLIP ---
@st.cache_resource
def load_clip():
    return clip.load("ViT-B/32", device="cpu")

model, preprocess = load_clip()


st.title("🎬 CINEPHILE – Chatbot Gợi Ý Phim")
st.header("Tư vấn phim bằng văn bản")
model_chat = genai.GenerativeModel(
    "gemini-2.0-flash-lite",
    system_instruction="""
                    Bạn là một tư vấn phim chuyên nghiệp, hãy gợi ý các bộ phim phù hợp với yêu cầu người dùng, bao gồm tên phim, thể loại, độ dài và điểm IMDb.
                    1. Nếu có các thông tin
"""
)

input_text = st.text_input("Nhập nội dung của bạn tại đây (ví dụ: 'Tôi thích phim hoạt hình khoảng 90 phút')", key=input)
submit = st.button("Gửi")

response = None
if input_text and submit:
    response = model_chat.generate_content(input_text)

mess = st.empty()

if response:
    mess.markdown(response.text)

# Lịch sử Chatbot
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Giao diện nút
if st.button("Xoá lịch sử"):
    st.session_state.chat_history = []  # Xoá dữ liệu trong session
    st.success("Đã xoá lịch sử!")

for role, msg in st.session_state.chat_history:
    st.markdown(f"**{role}**: {msg}")

if submit:
    response = model_chat.generate_content(input_text)
    st.session_state.chat_history.append(("🧑", input_text))
    st.session_state.chat_history.append(("🤖", response.text))


# --- PHẦN 2: Gợi ý từ ảnh poster ---
st.header("Tìm phim tương tự từ poster")

uploaded_file = st.file_uploader("Tải ảnh poster phim bạn thích lên", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Poster bạn đã tải", use_column_width=True)

    # Vector hóa ảnh
    image_input = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        image_vec = model[0].encode_image(image_input).numpy()

    # So sánh với vector đã mã hóa
    try:
        vectors = np.load("poster_vectors.npy")
    except:
        st.error("Thiếu file poster_vectors.npy. Hãy chạy generate_vectors.py trước.")
        st.stop()

    similarities = vectors @ image_vec.T
    top_indices = similarities.flatten().argsort()[-5:][::-1]

    st.subheader("Gợi ý phim tương tự:")
    for idx in top_indices:
        row = df.iloc[idx]
        st.markdown(f"** {row['Series_Title']}**  ")
        st.image(row["Poster_Link"], width=200)
        st.markdown(f"- **Thể loại:** {row['Genre']}")
        st.markdown(f"- **Thời lượng:** {row['Runtime']}")
        st.markdown(f"- **IMDb:** ⭐ {row['IMDB_Rating']}")
        st.markdown("---")