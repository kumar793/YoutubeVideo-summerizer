import nltk
import validators
import streamlit as st
from pytube import YouTube
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

prompt_temp = """
Please provide the summary of the following content in 500 words.
context = {text}
"""

prompt = PromptTemplate(template=prompt_temp, input_variables=['text'])

st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader('Summarize URL')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def validate_url(url, api_key):
    if not api_key.strip() or not url.strip():
        return "Please provide the information"
    elif not validators.url(url):
        return "Please provide a valid URL"
    else:
        try:
            with st.spinner("Processing..."):
                if "youtube.com" in url or "youtu.be" in url:
                    try:
                        loader = YoutubeLoader.from_youtube_url(url, add_video_info=True, language=['en', 'en-IN']) #added language to loader

                        documents = loader.load()

                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                        texts = text_splitter.split_documents(documents)

                        chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                        output = chain.run(texts)

                        return output
                    except Exception as e:
                        return f"Error processing YouTube URL: {e}"

                else:
                    try:
                        loader = UnstructuredURLLoader(urls=[url], ssl_verify=False, headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
                        documents = loader.load()
                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                        texts = text_splitter.split_documents(documents)

                        chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                        output = chain.run(texts)
                        return output
                    except Exception as e:
                        return f"Error processing URL: {e}"

        except Exception as e:
            return f"An unexpected error occurred: {e}"

api_key = st.sidebar.text_input("Enter the API Key", type="password")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=api_key)
url = st.text_input("URL", label_visibility="collapsed")

if st.button("Summarize"):
    output = validate_url(url, api_key)
    st.success(output)
