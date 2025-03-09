import streamlit as st
from tqdm import tqdm 
import pandas as pd
from pypdf import PdfReader
from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
import requests
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import time


st.sidebar.title("Settings")
chunk_selection = st.sidebar.slider("Select the max number of document chunks for context (Optional)", min_value=1, max_value=20, value=10)

if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="history",
        return_messages=True,
        input_key="question",
    )


# Initialize Session State for Chat History and Memory
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
    
if 'chat_history1' not in st.session_state:
    st.session_state.chat_history1 = []
    
# Clear Chat History Button
if st.sidebar.button("Clear Chat History"):
    st.session_state.chat_history = []
    st.success("Chat history cleared.")
    
    
    
    
    
    
# Add a spacer using st.sidebar.empty()
for _ in range(20):  # Adjust the range for spacing
    st.sidebar.write("")

# Bottom content
st.sidebar.markdown("---")  # Optional: Separator
st.sidebar.markdown("This application is developed using Large Language Models (LLMs) accessed via the Hugging Face API, with ChromaDB serving as the vector database for efficient data storage and retrieval.")
st.sidebar.info("📌 **Any Question?** [Please contact](https://www.linkedin.com/in/priyabrata-karmakar-phd-0806b3a8/)")

    
    

model_id = "sentence-transformers/all-MiniLM-L6-v2"
hf_token = "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" #(promotions.social1)"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"(pkarcreative)
api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
headers = {"Authorization": f"Bearer {hf_token}"}

def query_embed(texts):
    response = requests.post(api_url, headers=headers, json={"inputs": texts, "options":{"wait_for_model":True}})
    return response.json()

def pdf_rag_embedding(reader):
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
    
    

    
    # Embedding the document chunks with embedding model
    
    output = query_embed(df_db["filtered_content"].tolist())
    
    df_db['embedding'] = df_db.apply(lambda x: output[x.name], axis=1)
    
    return df_db







def pdf_rag_context(df_db, question, top_relevant_chunks):
    
    
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

#hf_token = st.sidebar.text_input("Enter Hugging Face Token (Required)", type="password")
hf_token = "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"


# Main panel with PDF upload, question input, and answer display
st.title("PDF Retrieval-Augmented Generation (RAG): Your Personal Q&A Assistant")

# File uploader for PDF
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
if uploaded_file is not None:
    st.text("File uploaded successfully")
    message_placeholder = st.empty()
    pdf_reader = PdfReader(uploaded_file)
    if 'embedding' not in st.session_state:
        message_placeholder.info("Your file is being processed to create embeddings. Please wait.")
        df_db = pdf_rag_embedding(pdf_reader)
        st.session_state.embedding = df_db
        message_placeholder.success("Embeddings created, you can ask questions now.")
        
        
        
# Custom avatars
USER_AVATAR = "👤"  # User avatar (Emoji or image path)
ASSISTANT_AVATAR = "🤖"  # Assistant avatar (Emoji or image path)

for message in st.session_state.chat_history:
    avatar = USER_AVATAR if message["role"] == "user" else ASSISTANT_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["message"])

# Text box for question
if uploaded_file is not None:

    if question := st.chat_input("Ask a question:", key="user_input"):
        user_message = {"role": "user", "message": question}
        st.session_state.chat_history.append(user_message)
        
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(question) #this markdown will be overwritten on the next iteration
            

            
        # Load the QA pipeline
        context = pdf_rag_context(st.session_state.embedding, question, chunk_selection)
        
        with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
            message_placeholder1 = st.empty()
            message_placeholder1.info("Please wait, answer to your question is being generated")
        
        #### Tesxt Generation to answer the question
        llm = HuggingFaceHub(
            repo_id="HuggingFaceH4/zephyr-7b-beta", 
            huggingfacehub_api_token=hf_token,  # Pass the API token directly
            task="text-generation",  
            model_kwargs={"temperature": 0.7, "max_new_tokens":5000}
        )
        
        conversation_history = "\n".join(
            [f"User: {chat['message']}\nAssistant: {chat.get('response', '')}" for chat in st.session_state.chat_history]
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
        {context}, Chat history: {history}
        ---
        Now here is the question you need to answer.
    
        Question: {question}""",
            },
        ]
        RAG_PROMPT_TEMPLATE = tokenizer.apply_chat_template(
            prompt_in_chat_format, tokenize=False, add_generation_prompt=True
        )
    
    
        prompt = PromptTemplate(
            input_variables=["question", "context", "history"],
            template=RAG_PROMPT_TEMPLATE,
        )
    
    
        llm_chain = LLMChain(prompt=prompt, llm=llm)
        response = llm_chain.run({"question": question, "context": context, "history": st.session_state.chat_history})
     
                
        
        
        llm_chain = LLMChain(prompt=prompt, llm=llm)
        response = llm_chain.run({"question": question, "context": context, "history": st.session_state.chat_history1})
        answer_start = response.find("<|assistant|>") + len("<|assistant|>")
        cleaned_response = response[answer_start:].strip()
        chatbot_message = {"role": "assistant", "message": cleaned_response}
        st.session_state.chat_history.append(chatbot_message)
        
        chatbot_message1 = {"question": question, f"answer to the - {question}": cleaned_response}
        st.session_state.chat_history1.append(chatbot_message1)
    
    
    
        # Display the answer

        #st.write("Answer:",  cleaned_response)
        # Display Assistant's Response with a Typing Effect
        

        message_placeholder1.markdown(cleaned_response)
        
