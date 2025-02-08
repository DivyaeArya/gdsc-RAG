import streamlit as st
import time
import base64
import requests
import tempfile
from groq import Groq
from deepgram import (
    DeepgramClient,
    SpeakOptions,
)
import chromadb
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

import os
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

GROQ_API_KEY = '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'
DEEPGRAM_API_KEY = 'dbbce00f3dccd72ab5dd503c2accfdb5f411a17e'

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

chroma_client = chromadb.PersistentClient(path="./chroma_db", settings=chromadb.Settings(allow_reset=True))  
vector_db = Chroma(client=chroma_client, embedding_function=embedding_model)
client = Groq(
    api_key= GROQ_API_KEY,
)

def process_documents(uploaded_files):
    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        text = uploaded_file.read().decode("utf-8")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=250)
        chunks = text_splitter.split_text(text)
        docs = [Document(page_content=chunk, metadata={"source": file_name}) for chunk in chunks]
        
        vector_db.add_documents(docs)
        st.sidebar.success(f"Added {len(chunks)} chunks from {file_name} to the database!")

filename = "output.mp3"
def tts(text):
    try:
        deepgram = DeepgramClient(api_key=DEEPGRAM_API_KEY)

        options = SpeakOptions(
            model="aura-asteria-en",
        )

        response = deepgram.speak.rest.v("1").save(filename, {"text":text}, options)
        print(response.to_json(indent=4))

    except Exception as e:
        print(f"Exception: {e}")


def transcribe_audio(audio_path):
    url = "https://api.groq.com/openai/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    files = {"file": open(audio_path, "rb")}
    data = {"model": "distil-whisper-large-v3-en", "language": "en"} 

    response = requests.post(url, headers=headers, files=files, data=data)

    if response.status_code == 200:
        return response.json().get("text", "Transcription failed")
    else:
        return f"Error: {response.text}"

def autoplay_audio(file_path, position):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio controls autoplay="true">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        position.markdown(
            md,
            unsafe_allow_html=True,
        )


def chat_completion(prompt):
    retrieved_docs = vector_db.similarity_search(prompt, k=3)
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    sources = ", ".join(set(doc.metadata.get("source", "Unknown") for doc in retrieved_docs))
    groq_output = client.chat.completions.create(messages=[{"role": "user", "content": f"Answer the following based on context (if None: proceed normally):\n{context}\nQuestion: {prompt}"}], 
                                                 model="llama-3.3-70b-versatile")
    answer = f"{groq_output.choices[0].message.content} \n \n Sources: {sources}"
    return answer



st.title('RAG App')


with st.sidebar:
    uploaded_files = st.file_uploader("Upload text document", type="txt", accept_multiple_files=True)
    if st.button("Reset (press after removing files)"):
        chroma_client.reset()
        st.rerun()
    voice_input = st.audio_input("Record your voice input here")
    st.divider()
    tts_toggle = st.toggle("TTS for responses", value = False)


if "messages" not in st.session_state:
    st.session_state.messages = []

prompt = ""

text_input = st.chat_input("Whats up?")

if uploaded_files:
    process_documents(uploaded_files)
    text_input = ""

if voice_input:
    audio_bytes = voice_input.read()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_bytes)
        transcription = transcribe_audio(temp_audio.name)
    prompt = transcription
    voice_input = None

if text_input:
    prompt = text_input
    text_input = "None"

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        a,b = st.columns(spec=[70,30])
        a.markdown(message["content"])
        if message["audio"]:
            # autoplay_audio(message["audio"], b)
            b.audio(message["audio"])


def typewr(response):
    for word in response.split():
        yield word + " "
        time.sleep(0.05)

if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt, "audio": None})

    response = chat_completion(prompt)
    with st.chat_message("assistant"):
        a,b = st.columns(spec=[70,30])
        a.write_stream(typewr(response))
        if tts_toggle:
            tts(response)
            autoplay_audio(filename, b)
            st.session_state.messages.append({"role": "assistant", "content": response, "audio": filename})
        else:
            st.session_state.messages.append({"role": "assistant", "content": response, "audio": None})
    prompt = ""

    
