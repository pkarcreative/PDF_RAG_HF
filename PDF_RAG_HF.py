import streamlit as st
from tqdm import tqdm 
import pandas as pd
from pypdf import PdfReader
from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
import requests
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


def pdf_rag_context(reader, model_id, question, top_relevant_chunks):

    pdf_content = [page.extract_text() for page in reader.pages]    
    # Create the raw knowledge base similarly (below) to use that with huggingface and langchain functions
    RAW_KNOWLEDGE_BASE = [
        LangchainDocument(page_content=pdf_content[i]) for i in range(len(pdf_content)) ]
    
    
    # We use a hierarchical list of separators specifically tailored for splitting Markdown documents
    # This list is taken from LangChain's MarkdownTextSplitter class
    PDF_SEPARATORS = [
    "\n",  # Newline
    "\r",  # Carriage return
    "\t",  # Tab
    " ",  # Space
    "\f",  # Form feed
    "\v",  # Vertical tab
    "\0",  # Null character
    "\x0c",  # Form feed (alternative)
    "\x1c",  # File separator
    "\x1d",  # Group separator
    "\x1e",  # Record separator
    "\x1f",  # Unit separator
    "\x7f",  # Delete
    "\ufffd",  # Replacement character
    "\u2028",  # Line separator
    "\u2029",  # Paragraph separator
    ]
    
    # If use RecursiveCharacterTextSplitter.from_huggingface_tokenizer, then the chunking will happen based on tokens.
    from transformers import AutoTokenizer
    tokenizer_name = "HuggingFaceH4/zephyr-7b-beta"
    

    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        AutoTokenizer.from_pretrained(tokenizer_name),
        chunk_size=512,  # The maximum number of characters in a chunk: we selected this value arbitrarily
        chunk_overlap=51,  # The number of characters to overlap between chunks
        add_start_index=True,  # If `True`, includes chunk's start index in metadata
        strip_whitespace=True,  # If `True`, strips whitespace from the start and end of every document
        separators=PDF_SEPARATORS,
    )
    
    docs_processed = []
    for doc in RAW_KNOWLEDGE_BASE:
        docs_processed += text_splitter.split_documents([doc])
        
    # The chunked docs (inside docs_processed) is a list. Converting that to a dataframe        
    df_db = pd.DataFrame(docs_processed, columns=["idx", "metadata", "content", "type"])    
    df_db= df_db[["content"]]
        
    def remove_unwanted_chars(text_tuple): 
        text = text_tuple[1] # the texts in individual rows are tuple format. It has 'page content' and the desired texts.
        return text
    
    df_db["filtered_content"] = df_db["content"].apply(remove_unwanted_chars)    
    
    
    api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
    headers = {"Authorization": f"Bearer {hf_token}"}
    
    def query_embed(texts):
        response = requests.post(api_url, headers=headers, json={"inputs": texts, "options":{"wait_for_model":True}})
        return response.json()
    
    # Embedding the document chunks with embedding model
    
    output = query_embed(df_db["filtered_content"].tolist())
    
    df_db['embedding'] = df_db.apply(lambda x: output[x.name], axis=1)
    
    df_temp = df_db 
    
    # Embed a user query in the same space
    
    query_vector = query_embed(question)
    
    import numpy as np
    def similarity(doc_embedding):
      return np.dot(doc_embedding, query_vector)    
        
    
    df_temp["similarity"] = df_temp["embedding"].apply(similarity)    
    df_temp.sort_values("similarity", ascending=False, inplace=True)   
    context = "\n".join(df_temp["filtered_content"][0:top_relevant_chunks])
    return context




# Sidebar with Hugging Face token, model selector, and slider
st.sidebar.title("Settings")
hf_token = st.sidebar.text_input("Enter Hugging Face Token (Required)", type="password")
model_id = st.sidebar.selectbox("Select Embedding Model (Optional)", ["sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L12-v2", "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"])
chunk_selection = st.sidebar.slider("Select the max number of document chunks for context (Optional)", min_value=1, max_value=100, value=10)

# Main panel with PDF upload, question input, and answer display
st.title("PDF Retrieval Augmented Generation (RAG) - No Paid API key required")

# File uploader for PDF
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

# Text box for question
question = st.text_input("Enter your question")

# Button to get answer
if st.button("Get Answer"):
    if not hf_token:
        st.warning("Please enter your Hugging Face token in the sidebar.")
    elif not uploaded_file:
        st.error("Please upload a PDF file.")
    elif not question:
        st.error("Please enter a question.")
    else:
        # Read and process the PDF
        pdf_reader = PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        # Load the QA pipeline
        context = pdf_rag_context(pdf_reader, model_id, question, chunk_selection)

        #### Tesxt Generation to answer the question
        llm = HuggingFaceHub(
            repo_id="HuggingFaceH4/zephyr-7b-beta", 
            huggingfacehub_api_token=hf_token,  # Pass the API token directly
            task="text-generation",  
            model_kwargs={"temperature": 0.7, "max_new_tokens":5000}
        )


        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")


        prompt_in_chat_format = [
            {
                "role": "system",
                "content": """Using the information contained in the context,
        give a comprehensive answer to the question.
        Respond only to the question asked, response should be concise and relevant to the question.
        If the answer cannot be deduced from the context, do not give an answer.""",
            },
            {
                "role": "user",
                "content": """Context:
        {context}
        ---
        Now here is the question you need to answer.

        Question: {question}""",
            },
        ]
        RAG_PROMPT_TEMPLATE = tokenizer.apply_chat_template(
            prompt_in_chat_format, tokenize=False, add_generation_prompt=True
        )


        prompt = PromptTemplate(
            input_variables=["question", "context"],
            template=RAG_PROMPT_TEMPLATE,
        )





                
        llm_chain = LLMChain(prompt=prompt, llm=llm)
        response = llm_chain.run({"question": question, "context": context})
        answer_start = response.find("<|assistant|>") + len("<|assistant|>")
        cleaned_response = response[answer_start:].strip()
        print(cleaned_response)



        # Display the answer
        st.write("Answer:",  cleaned_response)
