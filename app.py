import streamlit as st
import os
import pickle
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

st.title("RAG App")

def main():
    st.header("Chat with PDF")

    # Prompt the user to input their OpenAI API key
    openai_api_key = st.text_input("Enter your OpenAI API key:")
    if not openai_api_key:
        st.warning("Please enter your OpenAI API key.")
        st.stop()

    # Set the OpenAI API key as an environment variable
    os.environ["OPENAI_API_KEY"] = openai_api_key
    
    load_dotenv()

    # Upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')

    if pdf is not None:
        # Read PDF content
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Display the extracted text
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        st.write(chunks)

        store_name = pdf.name[:-4]
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
        else:
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)

        # Accept user questions/query
        query = st.text_input("Ask questions about your PDF file:")
        st.write(query)
        if query:
            docs = VectorStore.similarity_search(query=query, k=3)
            st.write(docs)
            llm = OpenAI()
            chain = load_qa_chain(llmm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
            
            print(cb)
            st.write(response)

if __name__ == "__main__":
    main()

















