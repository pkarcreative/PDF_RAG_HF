import streamlit as st
from tqdm import tqdm 
import pandas as pd
from pypdf import PdfReader
from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
import requests
import time
import numpy as np
from openai import OpenAI

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
    
# OpenAI API Key input
user_openai_key = st.sidebar.text_input("Enter your OpenAI API Key (optional):", type="password")   
    
# Clear Chat History Button
if st.sidebar.button("Clear Chat History"):
    st.session_state.chat_history = []
    # Reset usage count for free tier users when clearing chat
    if not user_openai_key:
        st.session_state.usage_count = 0
    st.success("Chat history cleared.")



# Usage tracking for free tier
if 'usage_count' not in st.session_state:
    st.session_state.usage_count = 0

FREE_USAGE_LIMIT = 5  # Number of free questions allowed

if user_openai_key:
    st.sidebar.success("✅ Using your API key - Unlimited usage!")
else:
    remaining = FREE_USAGE_LIMIT - st.session_state.usage_count
    if remaining <= 0:
        st.sidebar.error("🚫 Free trial exhausted! Please add your OpenAI API key to continue.")
    elif remaining <= 3:
        st.sidebar.warning("⚠️ Almost out of free questions! Add your API key for unlimited usage.")
        st.sidebar.info(f"🆓 Free trial: {remaining}/{FREE_USAGE_LIMIT} questions remaining")
    else:
        st.sidebar.info(f"🆓 Free trial for 5 questions only")
    
# Add a spacer using st.sidebar.empty()
for _ in range(5):  # Adjust the range for spacing
    st.sidebar.write("")

# Bottom content
st.sidebar.markdown("---")  # Optional: Separator
st.sidebar.markdown("### 🎯 How to Get Your Own API Key")
st.sidebar.markdown("1. Visit [OpenAI API Keys](https://platform.openai.com/api-keys)")
st.sidebar.markdown("2. Create an account and generate a new API key")
st.sidebar.markdown("3. Paste it in the text box above for unlimited usage!")
st.sidebar.markdown("---")
st.sidebar.markdown("This application uses OpenAI's GPT-4o-mini model. The data remains only within your browser session and is not stored anywhere else. Once you refresh the page, all data will be permanently lost.")
st.sidebar.markdown("⚠️ **Free usage is limited.** Add your own API key for unlimited access.")
st.sidebar.info("📌 **Any Question?** [Please contact](https://www.linkedin.com/in/priyabrata-karmakar-phd-0806b3a8/)")

# Initialize OpenAI client
DEFAULT_OPENAI_KEY = "sk-your-default-openai-key-here"  # Add your default key here

def get_openai_client():
    # Priority: User key > Streamlit Secrets > Default key
    api_key = user_openai_key if user_openai_key else st.secrets.get("OPENAI_API_KEY", DEFAULT_OPENAI_KEY)
    
    if not api_key or api_key == "sk-your-default-openai-key-here":
        st.error("OpenAI API key is required. Please add it in the sidebar or configure it in Streamlit secrets.")
        st.stop()
    return OpenAI(api_key=api_key)

def get_embeddings(texts, client):
    """Get embeddings using OpenAI's text-embedding-3-small model"""
    try:
        if isinstance(texts, str):
            texts = [texts]
        
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )
        return [embedding.embedding for embedding in response.data]
    except Exception as e:
        st.error(f"Error getting embeddings: {str(e)}")
        return None

def pdf_rag_embedding(reader):
    client = get_openai_client()
    
    pdf_content = [page.extract_text() for page in reader.pages]    
    # Create the raw knowledge base
    RAW_KNOWLEDGE_BASE = [
        LangchainDocument(page_content=pdf_content[i]) for i in range(len(pdf_content)) ]
    
    # Text separators for PDF processing
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
    
    # Use character-based splitting instead of token-based
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Character-based chunking
        chunk_overlap=100,  # Character overlap
        add_start_index=True,
        strip_whitespace=True,
        separators=PDF_SEPARATORS,
    )
    
    docs_processed = []
    for doc in RAW_KNOWLEDGE_BASE:
        docs_processed += text_splitter.split_documents([doc])
        
    # Convert to dataframe
    df_db = pd.DataFrame(docs_processed, columns=["idx", "metadata", "content", "type"])    
    df_db = df_db[["content"]]
        
    def remove_unwanted_chars(text_tuple): 
        text = text_tuple[1] # Extract text from tuple
        return text
    
    df_db["filtered_content"] = df_db["content"].apply(remove_unwanted_chars)    
    
    # Get embeddings using OpenAI
    texts_to_embed = df_db["filtered_content"].tolist()
    embeddings = get_embeddings(texts_to_embed, client)
    
    if embeddings is None:
        st.error("Failed to create embeddings")
        return None
    
    df_db['embedding'] = embeddings
    
    return df_db

def pdf_rag_context(df_db, question, top_relevant_chunks):
    client = get_openai_client()
    
    df_temp = df_db.copy()
    
    # Get embedding for the user query
    query_embeddings = get_embeddings(question, client)
    if query_embeddings is None:
        st.error("Failed to embed query")
        return ""
    
    query_vector = query_embeddings[0]
    
    def similarity(doc_embedding):
        return np.dot(doc_embedding, query_vector)    
    
    df_temp["similarity"] = df_temp["embedding"].apply(similarity)    
    df_temp.sort_values("similarity", ascending=False, inplace=True)   
    context = "\n".join(df_temp["filtered_content"][0:top_relevant_chunks])
    return context

def generate_response(question, context, chat_history):
    """Generate response using GPT-4o-mini"""
    client = get_openai_client()
    
    # Create conversation history string
    history_string = ""
    for i, chat in enumerate(chat_history[-5:]):  # Only use last 5 exchanges
        if chat["role"] == "user":
            history_string += f"User: {chat['message']}\n"
        elif chat["role"] == "assistant":
            history_string += f"Assistant: {chat['message']}\n"
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system", 
                    "content": """Using the information contained in the context, give a comprehensive answer to the question. 
                    Respond only to the question asked, response should be concise and relevant to the question. 
                    If the answer cannot be deduced from the context, do not give an answer. 
                    Consider the chat history for context but focus on answering the current question."""
                },
                {
                    "role": "user", 
                    "content": f"""Context:
{context}

Chat History:
{history_string}

Question: {question}"""
                }
            ],
            temperature=0.7,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return "I apologize, but I encountered an error while generating the response."

# Custom avatars
USER_AVATAR = "👤"
ASSISTANT_AVATAR = "🤖"

# Main panel
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
        if df_db is not None:
            st.session_state.embedding = df_db
            message_placeholder.success("Embeddings created, you can ask questions now.")
        else:
            message_placeholder.error("Failed to create embeddings. Please check your API key.")

# Display chat history
for message in st.session_state.chat_history:
    avatar = USER_AVATAR if message["role"] == "user" else ASSISTANT_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["message"])

# Chat input
if uploaded_file is not None and 'embedding' in st.session_state:
    if question := st.chat_input("Ask a question:", key="user_input"):
        # Check usage limits for free tier users
        if not user_openai_key and st.session_state.usage_count >= FREE_USAGE_LIMIT:
            st.error("🚫 **Free trial exhausted!** Please add your OpenAI API key in the sidebar to continue.")
            st.stop()
        
        # Increment usage count for free tier users
        if not user_openai_key:
            st.session_state.usage_count += 1
        
        # Add user message to chat history
        user_message = {"role": "user", "message": question}
        st.session_state.chat_history.append(user_message)
        
        # Display user message
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(question)
        
        # Get context from RAG
        context = pdf_rag_context(st.session_state.embedding, question, chunk_selection)
        
        # Display assistant response
        with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
            message_placeholder1 = st.empty()
            message_placeholder1.info("Please wait, answer to your question is being generated...")
            
            # Generate response using GPT-4o-mini
            response = generate_response(question, context, st.session_state.chat_history)
            
            # Add assistant message to chat history
            chatbot_message = {"role": "assistant", "message": response}
            st.session_state.chat_history.append(chatbot_message)
            
            # Add to alternative chat history format
            chatbot_message1 = {"question": question, f"answer to the - {question}": response}
            st.session_state.chat_history1.append(chatbot_message1)
            
            # Display the response
            message_placeholder1.markdown(response)
            
            # Show remaining usage for free tier users
            if not user_openai_key:
                remaining = FREE_USAGE_LIMIT - st.session_state.usage_count
                if remaining > 0:
                    st.info(f"🆓 Free questions remaining: {remaining}")
                else:
                    st.warning("🔑 **Add your API key for unlimited usage!**")
