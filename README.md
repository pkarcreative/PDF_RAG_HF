# PDF Retrieval-Augmented Generation (RAG): Your Personal Q&A Assistant

A Streamlit-based application that allows you to upload PDF documents and ask questions about their content using OpenAI's GPT-4o-mini model with vector embeddings for intelligent document retrieval.

## Features

- **PDF Upload & Processing**: Upload any PDF document for analysis
- **Intelligent Q&A**: Ask questions about your PDF content with context-aware responses
- **Vector Embeddings**: Uses OpenAI's text-embedding-3-small model for semantic search
- **Chat History**: Maintains conversation context for better responses
- **Free Trial**: 5 free questions without API key requirement
- **Customizable Chunks**: Adjust the number of document chunks used for context (1-20)
- **Real-time Processing**: Live feedback during embedding creation and response generation

## Installation

1. **Clone the repository**:
```bash
git clone <your-repo-url>
cd pdf-rag-assistant
```

2. **Install required packages**:
```bash
pip install streamlit tqdm pandas pypdf langchain openai numpy requests
```

3. **Run the application**:
```bash
streamlit run app.py
```

## Usage

### Getting Started

1. **Launch the app** - Open your browser and navigate to the Streamlit URL (usually `http://localhost:8501`)
2. **Upload a PDF** - Use the file uploader to select your PDF document
3. **Wait for processing** - The app will create embeddings for your document (this may take a few moments)
4. **Start asking questions** - Once processing is complete, use the chat input to ask questions about your PDF

### API Key Setup (Recommended)

For unlimited usage, add your OpenAI API key:

1. Visit [OpenAI API Keys](https://platform.openai.com/api-keys)
2. Create an account and generate a new API key
3. Enter the key in the sidebar text input
4. Enjoy unlimited questions!

### Free Trial

- **5 free questions** without an API key
- Perfect for testing the application
- Add your API key for unlimited usage

## Configuration Options

### Sidebar Settings

- **Document Chunks**: Adjust the maximum number of document chunks (1-20) used for context
- **API Key Input**: Enter your OpenAI API key for unlimited usage
- **Clear Chat History**: Reset your conversation history

### Document Processing

- **Chunk Size**: 1000 characters per chunk
- **Chunk Overlap**: 100 characters overlap between chunks
- **Advanced Separators**: Handles various PDF text separators including whitespace, line breaks, and control characters

## How It Works

1. **PDF Processing**: Extracts text from each page of your PDF
2. **Text Chunking**: Splits the document into manageable chunks using RecursiveCharacterTextSplitter
3. **Embedding Creation**: Generates vector embeddings for each chunk using OpenAI's text-embedding-3-small
4. **Query Processing**: When you ask a question, the app:
   - Creates an embedding for your question
   - Finds the most relevant document chunks using cosine similarity
   - Sends the relevant context to GPT-4o-mini for response generation
5. **Response Generation**: GPT-4o-mini generates a comprehensive answer based on the retrieved context and chat history

## Technical Details

### Dependencies

- **streamlit**: Web interface framework
- **pypdf**: PDF text extraction
- **langchain**: Document processing and text splitting
- **openai**: GPT-4o-mini and embedding models
- **pandas**: Data manipulation
- **numpy**: Vector similarity calculations
- **tqdm**: Progress bars (for potential future enhancements)

### Models Used

- **Embeddings**: `text-embedding-3-small` (1536 dimensions)
- **Chat Completion**: `gpt-4o-mini`
- **Temperature**: 0.7 for balanced creativity and accuracy
- **Max Tokens**: 1000 per response

### Data Privacy

- **Local Processing**: All data remains within your browser session
- **No Storage**: Documents and conversations are not stored permanently
- **Session-Based**: Data is cleared when you refresh the page

## Limitations

- **PDF Only**: Currently supports PDF files only
- **Text-Based**: Works with text content; images and complex formatting may not be processed
- **Session Storage**: Chat history and embeddings are lost on page refresh
- **Free Trial Limit**: 5 questions without API key

## Troubleshooting

### Common Issues

1. **"OpenAI API key is required"**
   - Add your API key in the sidebar or configure Streamlit secrets

2. **"Failed to create embeddings"**
   - Check your internet connection
   - Verify your API key is valid
   - Ensure you have OpenAI API credits

3. **"Free trial exhausted"**
   - Add your OpenAI API key for unlimited usage
   - Or clear chat history to reset (if using shared demo)

4. **Poor answer quality**
   - Try increasing the number of document chunks in the sidebar
   - Ensure your question is specific and clear
   - Check that your PDF contains relevant information

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## Support

For questions or issues:
- **Technical Issues**: Create an issue in the repository
- **General Questions**: [Contact on LinkedIn](https://www.linkedin.com/in/priyabrata-karmakar-phd-0806b3a8/)

## License

This project is open source. Please check the license file for details.

## Acknowledgments

- OpenAI for providing the GPT-4o-mini and embedding models
- Streamlit for the excellent web framework
- LangChain for document processing utilities


This app will be continuously developed.
This app can be accessed here: https://pdf-rag-hf.streamlit.app/


