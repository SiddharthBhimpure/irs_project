import streamlit as st
from src.helper import get_pdf_text, get_text_chunks, get_vector_store, get_conversational_chain

def main():
    st.set_page_config(page_title="Information Retrieval System")
    st.header("📄 Information-Retrieval-System (Gemini-Powered)")
    user_question = st.text_input("Ask a question from the PDF files:")

    if user_question and "conversation" in st.session_state:
        response = st.session_state.conversation.run(user_question)
        st.write("💬 Answer:", response)

    with st.sidebar:
        st.title("📎 MENU")
        pdf_docs = st.file_uploader("Upload PDF files", accept_multiple_files=True)
        if st.button("Submit and Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vector_store = get_vector_store(text_chunks)
                st.session_state.conversation = get_conversational_chain(vector_store)
                st.success("✅ Done!")

if __name__ == "__main__":
    main()
