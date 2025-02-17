# 🦙📚 LlamaIndex - Chat with YOUR docs

This chatbot is powered by LlamaIndex, and it augments OpenAI GPT 3.5 or 4 with your own data.

## Overview of the App

<img src="app.png" width="75%">

- Takes user queries via Streamlit's `st.chat_input` and displays both user queries and model responses with `st.chat_message`
- Uses LlamaIndex to load and index data and create a chat engine that will retrieve context from that data to respond to each user query

## Demo App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://lllamaindex-chat-with-docs.streamlit.app/)

## Get an OpenAI API key

You can get your own OpenAI API key by following these instructions:

1. Go to <https://platform.openai.com/account/api-keys>
2. Click on the `+ Create new secret key` button
3. Next, enter an identifier name (optional) and click on the `Create secret key` button

## Try out the app

Once the app is loaded, select and load your data, wait for the index to build, then enter your question about the content of the docs you provided and wait for a response.
