import os
import streamlit as st
from langchain_chroma import Chroma


# Page Configuration & Headline
st.set_page_config(page_title="Chat with PDF Dashboard", layout="centered")
st.title("Chat with PDF")
st.write("Upload your PDF document and chat with it easily.")

# Upload Section
st.subheader("Upload PDF Document")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

# Logic to save PDF in local folder
if uploaded_file is not None:
  # Local folder name "uploads"
  save_folder = "uploads"
  os.makedirs(save_folder, exist_ok=True)

  file_path = os.path.join(save_folder, uploaded_file.name)

  # Write/save file to local directory
  with open(file_path, "wb") as f:
    f.write(uploaded_file.getbuffer())

  st.success(
      f"✅ File successfully saved in local folder: `{save_folder}/"
      f"{uploaded_file.name}`"
  )

  # Chat/Query section below
  st.markdown("---")
  st.subheader("Chat with your PDF")

  # User query input box
  user_query = st.text_input("Ask a question about your uploaded PDF:")

  if user_query:
    st.info(f"You asked: '{user_query}' (Backend integration will be done in the next step!)")

else:
  st.info("Please upload a PDF file first.")