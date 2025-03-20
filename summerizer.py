import nltk
import validators
import streamlit as st
from pytube import YouTube
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import UnstructuredURLLoader
import whisper
import tempfile
from moviepy.editor import AudioFileClip
from googletrans import Translator

# Load Whisper Model
model = whisper.load_model("medium")  # Options: "small", "medium", "large"

# Prompt Template for Summarization
prompt_temp = """
Please provide the summary of the following content in detail.
context = {text}
"""
prompt = PromptTemplate(template=prompt_temp, input_variables=['text'])

st.set_page_config(page_title="Summarize YT Videos & Websites", page_icon="🦜")
st.title("🦜 Summarize YT Videos & Websites")
st.subheader('Transcribe, Translate & Summarize')

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def download_audio(youtube_url):
    """Downloads YouTube video audio and converts it to WAV format."""
    try:
        yt = YouTube(youtube_url)
        stream = yt.streams.filter(only_audio=True).first()
        temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        stream.download(filename=temp_audio_file.name)
        
        # Convert to WAV format
        audio_clip = AudioFileClip(temp_audio_file.name)
        wav_file = temp_audio_file.name.replace(".mp4", ".wav")
        audio_clip.write_audiofile(wav_file, codec="pcm_s16le")
        audio_clip.close()
        
        return wav_file
    except Exception as e:
        return f"Error downloading audio: {e}"

def transcribe_audio(audio_path):
    """Transcribes audio using OpenAI Whisper."""
    try:
        result = model.transcribe(audio_path)
        return result["text"]
    except Exception as e:
        return f"Error transcribing audio: {e}"

def translate_to_english(text):
    """Translates text to English using Google Translate."""
    try:
        translator = Translator()
        translated_text = translator.translate(text, dest="en")
        return translated_text.text
    except Exception as e:
        return f"Error translating text: {e}"

def summarize_text(text, llm):
    """Summarizes text using LangChain and Gemini AI."""
    try:
        chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
        return chain.run(text)
    except Exception as e:
        return f"Error summarizing text: {e}"

def validate_url(url, llm):
    """Validates and processes YouTube or website URLs."""
    if not api_key.strip() or not url.strip():
        return "Please provide the API key and URL"
    elif not validators.url(url):
        return "Please provide a valid URL"
    
    try:
        with st.spinner("Processing..."):
            if "youtube.com" in url or "youtu.be" in url:
                # YouTube Processing
                audio_file = download_audio(url)
                if "Error" in audio_file:
                    return audio_file
                
                transcript = transcribe_audio(audio_file)
                if "Error" in transcript:
                    return transcript

                translated_text = translate_to_english(transcript)
                summary = summarize_text(translated_text, llm)
                
                return summary
            
            else:
                # Website Summarization
                loader = UnstructuredURLLoader(urls=[url], ssl_verify=False, headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"
                })
                data = loader.load()
                return summarize_text(data, llm)
    
    except Exception as e:
        return f"An unexpected error occurred: {e}"

api_key = st.sidebar.text_input("Enter the API Key", type="password")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=api_key)
url = st.text_input("Enter URL", label_visibility="collapsed")

if st.button("Summarize"):
    output = validate_url(url, llm)
    st.success(output)
