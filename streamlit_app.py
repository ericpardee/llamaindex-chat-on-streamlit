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

# Constants
SFTP_PORT = 22
LOCAL_DIR = './data'
UPLOAD_ZIP_FILE_NAME = 'uploaded.zip'
CHAT_ENGINE = 'condense_question' # https://gpt-index.readthedocs.io/en/latest/core_modules/query_modules/chat_engines/root.html

st.set_page_config(page_title="Chat with the YOUR docs", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)
openai.api_key = st.secrets.openai_key
st.title("Chat with YOUR docs")
st.header("Powered by LlamaIndex 💬🦙 and OpenAI")

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
        credentials['branch'] = st.text_input('Branch', value='main')  # Default to 'main'
        
    elif data_source == 'zip':
        uploaded_file = st.file_uploader("Choose a zip file containing .txt, .md, or .pdf files", type="zip")
        if uploaded_file is not None:
            st.session_state.uploaded_file = uploaded_file

    if data_source != 'zip':
        directory = st.text_input('Directory Path')
    else:
        directory = None  # Set directory to None if 'zip' is selected

    # Advanced Settings
    with st.expander("Advanced Settings"):
        model = st.selectbox('Select Model', ['gpt-3.5-turbo', 'gpt-4'], index=1)  # Default to gpt-4
        temperature = st.slider('Temperature',
                                min_value=0.0,
                                max_value=1.0,
                                value=0.0,
                                step=0.1,
                                help="The temperature parameter controls randomness in boltzmann sampling. Lower temperature results in less random completions. As the temperature approaches zero, the model will become deterministic and repetitive. Higher temperature results in more random completions."
                                )
        system_prompt = st.text_area('System Prompt',
                                     value="",  # Default value
                                     placeholder="You are an expert in healthcare IT security and compliance and your job is to answer technical questions. Assume that all questions are related to the healthcare IT security and compliance. Keep your answers technical and based on facts – do not hallucinate features.",
                                     height=100,
                                     help="The system prompt is used to provide context to the model. It should be a short paragraph \
                                        to help the model understand the domain of the data.")

    return data_source, credentials, directory, model, temperature, system_prompt

def download_data(source, credentials, directory):
    st.info(f"Starting download from {source}...")
    
    if not os.path.exists(LOCAL_DIR):
        os.makedirs(LOCAL_DIR)

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
                local_file_path = os.path.join(LOCAL_DIR, file_name.split('/')[-1])
                if obj['Size'] > 0:  # This will check if the object is a file (size > 0) or a directory (size == 0)
                    s3.download_file(s3_bucket, file_name, local_file_path)
                else:
                    os.makedirs(local_file_path, exist_ok=True)  # Create directory structure if the object is a directory

        elif source == 'sftp':
            try:
                transport = paramiko.Transport((credentials['hostname'], SFTP_PORT))
                transport.connect(username=credentials['username'], password=credentials['password'])
                sftp = paramiko.SFTPClient.from_transport(transport)  # Correct way to create SFTP client
                remote_path = directory
                files = sftp.listdir(remote_path)
                for file in files:
                    remote_file_path = os.path.join(remote_path, file)
                    local_file_path = os.path.join(LOCAL_DIR, file)
                    sftp.get(remote_file_path, local_file_path)
                sftp.close()
            except Exception as e:
                error_message = f"An error occurred: {str(e)}"
                st.error(error_message)
                raise Exception(error_message)
            finally:
                transport.close()  # Ensure the transport is closed
            
        elif source == 'git':
            repo = git.Repo.clone_from(
                credentials['repo_url'],
                to_path='./repo',
                branch=credentials['branch']  # Use the user-specified branch
            )
            git_dir_path = os.path.join('./repo', directory)
            for root, dirs, files in os.walk(git_dir_path):
                for file in files:
                    source_file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(source_file_path, './repo')
                    local_file_path = os.path.join(LOCAL_DIR, rel_path)
                    os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                    os.rename(source_file_path, local_file_path)
            shutil.rmtree('./repo')

        elif source == 'zip':
            if 'uploaded_file' in st.session_state:
                with st.spinner(text="Uploading and extracting zip file..."):
                    zip_path = os.path.join(LOCAL_DIR, UPLOAD_ZIP_FILE_NAME)
                    with open(zip_path, "wb") as f:
                        f.write(st.session_state.uploaded_file.getbuffer())
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(LOCAL_DIR)
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
def load_and_index_data(data_source, credentials, directory, model, temperature, system_prompt):
    download_data(data_source, credentials, directory)
    with st.spinner(text="Loading and indexing YOUR docs – hang tight! This should take 1-2 minutes."):
        reader = SimpleDirectoryReader(input_dir=LOCAL_DIR, recursive=True)
        docs = reader.load_data()
        service_context = ServiceContext.from_defaults(
            llm=OpenAI(model=model,
                       temperature=temperature,
                       system_prompt=system_prompt)
            )
        index = VectorStoreIndex.from_documents(docs, service_context=service_context) # https://docs.llamaindex.ai/en/stable/core_modules/data_modules/index/vector_store_guide.html
        return index

if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about YOUR docs!"}
    ]

data_source, credentials, directory, model, temperature, system_prompt = get_user_inputs()

if st.button('Load Data'):
    index = load_and_index_data(data_source, credentials, directory, model, temperature, system_prompt)  # Pass the additional arguments here
    st.session_state.chat_engine = index.as_chat_engine(chat_mode=CHAT_ENGINE, verbose=True)

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