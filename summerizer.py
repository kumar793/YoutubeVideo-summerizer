import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader,UnstructuredURLLoader
prompt_temp = """
Please provide the summary of the following content in 500 words.
context = {text}
"""
prompt = PromptTemplate(template=prompt_temp,input_variables=['text'])

st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader('Summarize URL')

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
                    loader=UnstructuredURLLoader(urls=[generic_url],ssl_verify=False,
                                                 headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
                
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





