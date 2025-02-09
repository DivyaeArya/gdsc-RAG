# RAG App
## Hosted [Here](https://gdsc-rag.streamlit.app/)
https://gdsc-rag.streamlit.app/
## Overview
The RAG (Retrieval-Augmented Generation) App is a Streamlit-based application that integrates text retrieval, document processing, and text-to-speech functionalities. It allows users to interact with a chatbot powered by Groq LLM while enhancing responses using relevant context retrieved from uploaded text documents. Additionally, the app supports voice input transcription and text-to-speech synthesis.

## Features
- **Chat Interface**: A chatbot powered by Groq's language models with context-aware response generation.
- **Document Upload**: Users can upload `.txt` files, which are processed and stored in a vector database (ChromaDB) for retrieval.
- **Voice Input**: Users can record voice messages, which are transcribed into text using Groq's audio transcription API.
- **Text-to-Speech (TTS)**: Responses can be converted into speech using Deepgram's TTS service.
- **Persistence**: The app maintains a session history of messages exchanged.
- **Reset Functionality**: Users can clear uploaded documents and reset the database.

## Dependencies
Ensure you have the following Python libraries installed:
```sh
pip install streamlit requests base64 groq deepgram-sdk chromadb langchain
```

## API Keys
This application requires API keys for Groq and Deepgram. Replace the placeholders in the script with your actual API keys.

## Running the App
Run the Streamlit application with:
```sh
streamlit run app.py
```

## Known Issues
- If a document is uploaded after a message is sent, the app may resend the last prompt unintentionally. This can be fixed by resetting `text_input` properly.

## Future Improvements
- Enhance session handling to prevent unintended re-prompting after document uploads.
- Improve UI/UX for better user experience.
- Support additional file formats beyond `.txt`.

## License
This project is open-source and available for modification and redistribution under the MIT License.

