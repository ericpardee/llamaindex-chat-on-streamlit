import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext
from llama_index.llms import OpenAI
import openai
from llama_index import SimpleDirectoryReader
import boto3
import paramiko
import git
import os
import pandas as pd
import shutil
import botocore.exceptions
import zipfile

st.set_page_config(page_title="Chat with the YOUR docs", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)
openai.api_key = st.secrets.openai_key
st.title("Chat with YOUR docs")
st.header("Powered by LlamaIndex 💬🦙 and GPT-4")

def get_user_inputs():
    data_source = st.selectbox('Select Data Source', ['zip', 's3', 'sftp', 'git'])

    credentials = {}
    if data_source == 's3':
        credentials['bucket_name'] = st.text_input('Bucket Name')
        credentials['key_id'] = st.text_input('Key ID')
        credentials['key_secret'] = st.text_input('Key Secret', type='password')

    elif data_source == 'sftp':
        credentials['hostname'] = st.text_input('Hostname')
        credentials['username'] = st.text_input('Username')
        credentials['password'] = st.text_input('Password', type='password')

    elif data_source == 'git':
        credentials['repo_url'] = st.text_input('Repository URL')
        credentials['access_token'] = st.text_input('Access Token', type='password')
        
    elif data_source == 'zip':
        uploaded_file = st.file_uploader("Choose a zip file containing .txt, .md, or .pdf files", type="zip")
        if uploaded_file is not None:
            st.session_state.uploaded_file = uploaded_file

    if data_source != 'zip':
        directory = st.text_input('Directory Path')
    else:
        directory = None  # Set directory to None if 'zip' is selected

    return data_source, credentials, directory

def download_data(source, credentials, directory):
    st.info(f"Starting download from {source}...")
    
    local_dir = './data'
    
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)

    try:
        if source == 's3':
            s3 = boto3.client(
                's3',
                aws_access_key_id=credentials['key_id'],
                aws_secret_access_key=credentials['key_secret']
            )
            s3_bucket = credentials['bucket_name']
            for obj in s3.list_objects_v2(Bucket=s3_bucket, Prefix=directory)['Contents']:
                file_name = obj['Key']
                local_file_path = os.path.join(local_dir, file_name.split('/')[-1])
                if obj['Size'] > 0:  # This will check if the object is a file (size > 0) or a directory (size == 0)
                    s3.download_file(s3_bucket, file_name, local_file_path)
                else:
                    os.makedirs(local_file_path, exist_ok=True)  # Create directory structure if the object is a directory
                
        elif source == 'sftp':
            transport = paramiko.Transport((credentials['hostname'], 22))
            transport.connect(username=credentials['username'], password=credentials['password'])
            sftp = transport.open_sftp()
            remote_path = directory
            files = sftp.listdir(remote_path)
            for file in files:
                remote_file_path = os.path.join(remote_path, file)
                local_file_path = os.path.join(local_dir, file)
                sftp.get(remote_file_path, local_file_path)
            sftp.close()
            
        elif source == 'git':
            repo = git.Repo.clone_from(credentials['repo_url'], to_path='./repo', branch='master')
            git_dir_path = os.path.join('./repo', directory)
            for root, dirs, files in os.walk(git_dir_path):
                for file in files:
                    source_file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(source_file_path, './repo')
                    local_file_path = os.path.join(local_dir, rel_path)
                    os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                    os.rename(source_file_path, local_file_path)
            shutil.rmtree('./repo')

        elif source == 'zip':
            if 'uploaded_file' in st.session_state:
                with st.spinner(text="Uploading and extracting zip file..."):
                    zip_path = os.path.join(local_dir, "uploaded.zip")
                    with open(zip_path, "wb") as f:
                        f.write(st.session_state.uploaded_file.getbuffer())
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(local_dir)
                    os.remove(zip_path)  # Optionally remove the zip file after extraction
                   
    except botocore.exceptions.NoCredentialsError as e:
        error_message = f"Credentials error: {str(e)}"
        st.error(error_message)
        raise Exception(error_message)
    except paramiko.AuthenticationException as e:
        error_message = f"Authentication error: {str(e)}"
        st.error(error_message)
        raise Exception(error_message)
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        st.error(error_message)
        raise Exception(error_message)

    st.info("Download completed.")

@st.cache_resource(show_spinner=False)
def load_data(data_source, credentials, directory):
    download_data(data_source, credentials, directory)
    with st.spinner(text="Loading and indexing the Streamlit docs – hang tight! This should take 1-2 minutes."):
        reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        docs = reader.load_data()
        service_context = ServiceContext.from_defaults(
            llm=OpenAI(model="gpt-4", 
                       temperature=0,
                       system_prompt="You are an expert on the Streamlit Python library and your job is to answer technical \
                           questions. Assume that all questions are related to the Streamlit Python library. Keep your answers technical \
                               and based on facts – do not hallucinate features.")
            )
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        return index

if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about YOUR docs!"}
    ]

data_source, credentials, directory = get_user_inputs()

if st.button('Load Data'):
    index = load_data(data_source, credentials, directory)
    st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

if "chat_engine" in st.session_state:
    prompt = st.chat_input("Your question")
    if prompt:  # Check if a non-empty string is returned from st.chat_input
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.chat(prompt)
            st.session_state.messages.append({"role": "assistant", "content": response.response})

    for message in st.session_state.messages:
        with st.chat_message(f"{message['role']}"):
            st.write(message["content"])
else:
    st.write("Please load the data to begin chatting.")