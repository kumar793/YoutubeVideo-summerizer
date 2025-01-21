import os
import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader,UnstructuredURLLoader
from dotenv import load_dotenv
load_dotenv()
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
prompt_temp = """
Please provide the summary of the following content in 500 words.
context = {text}
"""
prompt = PromptTemplate(template=prompt_temp,input_variables=['text'])

def validate_url(url):
    if not url.strip():
        return "Please provide the information"
    elif not validators.url(url):
        return "Please provide the valid url"
    else:
        try:
            with st.spinner("waiting"):
                if "youtube.com" in url:
                    loader = YoutubeLoader.from_youtube_url(url,add_video_info = True)
                else:
                    loader = UnstructuredURLLoader(urls=[url],ssl_verify = False)
                data = loader.load()

                chain = load_summarize_chain(llm,chain_type = "stuff",
                                            prompt = prompt)
                
                output = chain.invoke(data)

                return output
        except Exception as e:
            return e


api_key = st.sidebar.text_input("Enter the API Key" , type = "password")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",api_key = api_key)
url = st.text_input("URL",label_visibility="collapsed")
if st.button("summerize"):
    output = validate_url(url)
st.success(output)





