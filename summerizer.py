import nltk
import validators
import streamlit as st
from urllib.parse import urlparse
from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Download necessary NLTK data
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")

# Prompt template for summarization
prompt_temp = """
Please provide the summary of the following content in 500 words.
context = {text}
"""

prompt = PromptTemplate(template=prompt_temp, input_variables=["text"])

st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader("Summarize URL")

# Function to convert shortened YouTube URLs
def expand_youtube_url(url):
    parsed_url = urlparse(url)
    if "youtu.be" in parsed_url.netloc:
        video_id = parsed_url.path.lstrip("/")
        return f"https://www.youtube.com/watch?v={video_id}"
    return url

# Extract video ID from YouTube URL
def get_video_id(url):
    parsed_url = urlparse(url)
    if "youtube.com" in parsed_url.netloc:
        return parsed_url.query.split("v=")[-1].split("&")[0]
    elif "youtu.be" in parsed_url.netloc:
        return parsed_url.path.lstrip("/")
    return None

# Function to fetch YouTube transcript
def fetch_youtube_transcript(url):
    try:
        video_id = get_video_id(url)
        if not video_id:
            return "Invalid YouTube URL"
        
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        text = " ".join([entry["text"] for entry in transcript])
        return text
    except Exception as e:
        return f"Error fetching transcript: {e}"

# Function to validate and process the URL
def validate_url(url, api_key, llm):
    if not api_key.strip() or not url.strip():
        return "Please provide the information"
    elif not validators.url(url):
        return "Please provide a valid URL"
    else:
        try:
            with st.spinner("Processing..."):
                url = expand_youtube_url(url)  # Expand shortened YouTube URLs if necessary

                if "youtube.com" in url or "youtu.be" in url:
                    try:
                        transcript_text = fetch_youtube_transcript(url)
                        if "Error" in transcript_text:
                            return transcript_text

                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                        texts = text_splitter.split_text(transcript_text)

                        chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                        output = chain.run(texts)

                        return output
                    except Exception as e:
                        return f"Error processing YouTube URL: {e}"

                else:
                    try:
                        return "Currently, only YouTube summarization is supported."
                    except Exception as e:
                        return f"Error processing URL: {e}"

        except Exception as e:
            return f"An unexpected error occurred: {e}"

# Streamlit UI
api_key = st.sidebar.text_input("Enter the API Key", type="password")
url = st.text_input("Enter URL")

if st.button("Summarize"):
    if not api_key:
        st.error("API Key is required!")
    else:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=api_key)
        output = validate_url(url, api_key, llm)
        st.success(output)
