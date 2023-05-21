import streamlit as st
import pickle
import os
from langchain.chat_models import ChatOpenAI


print(os.getcwd())
from llama_index import LLMPredictor, PromptHelper, ServiceContext,SimpleDirectoryReader
from llama_index import download_loader, GPTVectorStoreIndex
from llamahub_modules.github_repo import GithubRepositoryReader, GithubClient

if 'response' not in st.session_state:
    st.session_state.response = ''

def initialize_session():
    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"))

    max_input_size = 4096
    num_output = 256
    max_chunk_overlap = 20
    prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    github_client = GithubClient(os.getenv(st.session_state.GITHUB_TOKEN))
    loader = GithubRepositoryReader(
        github_client,
        owner =  st.session_state.REPO_OWNER,
        repo =   st.session_state.REPO_NAME,
        # filter_directories =     ([st.session_state.FILTER_DIRECTORY], GithubRepositoryReader.FilterType.INCLUDE),
        filter_file_extensions = ([st.session_state.FILTER_FILE_EXTENSION], GithubRepositoryReader.FilterType.INCLUDE),
        verbose =                True,
        concurrent_requests =    10,
    )

    
def send_click():
    
    docs = None
    if os.path.exists("docs.pkl"):
        initialize_session()
        with open("docs.pkl", "rb") as f:
            docs = pickle.load(f)

    if docs is None:
        os.environ['OPENAI_API_KEY'] = st.session_state.OPENAI_API_KEY
        github_client = GithubClient(os.getenv(st.session_state.GITHUB_TOKEN))
        loader = GithubRepositoryReader(
            github_client,
            owner =  st.session_state.REPO_OWNER,
            repo =   st.session_state.REPO_NAME,
            # filter_directories =     ([st.session_state.FILTER_DIRECTORY], GithubRepositoryReader.FilterType.INCLUDE),
            filter_file_extensions = ([st.session_state.FILTER_FILE_EXTENSION], GithubRepositoryReader.FilterType.INCLUDE),
            verbose =                True,
            concurrent_requests =    10,
        )

        docs = loader.load_data(branch=st.session_state.REPO_BRANCH)

        with open("docs.pkl", "wb") as f:
            pickle.dump(docs, f)

    index = GPTVectorStoreIndex.from_documents(docs,  service_context=service_context)
    query_engine = index.as_query_engine()
    st.session_state.response  = query_engine.query(st.session_state.prompt)

st.title("GithubGPT")
sidebar_placeholder = st.sidebar.container()
sidebar_placeholder.header('Settings:')
sidebar_placeholder.text_input("OpenAI Key: ", key='OPENAI_API_KEY')
sidebar_placeholder.text_input("Github Token: ", key='GITHUB_TOKEN')
col1, col2 = st.columns(2)
col1.text_input("Repo Owner: ", key='REPO_OWNER')
col1.text_input("Filter Directory: ", key='FILTER_DIRECTORY')
col1.text_input("Branch: ", key='REPO_BRANCH')
col2.text_input("Repo Name: ", key='REPO_NAME')
col2.text_input("Filter File Extension: ", key='FILTER_FILE_EXTENSION')
st.text_input("Ask something: ", key='prompt')
st.button("Send", on_click=send_click)



if st.session_state.response:
        st.subheader("Response: ")
        st.success(st.session_state.response, icon= "🤖")



