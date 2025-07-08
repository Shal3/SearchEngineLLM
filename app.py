import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun,DuckDuckGoSearchRun
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain.agents import initialize_agent,AgentType
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from duckduckgo_search.exceptions import DuckDuckGoSearchException
from langchain.utilities import SerpAPIWrapper
from langchain_core.exceptions import OutputParserException
## We can user SerpApi instead of DuckDuckGo to avoid rate limit issue

import os


from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

## Sidebar "settings" for API key during runtime
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your GROQ API Key:", type="password")

## Creating Arxiv Tool using ArxivQueryRun and binding with ArxivAPIWrapper
arxiv_wrapper = ArxivAPIWrapper(top_k_results = 2, doc_content_chars_max=1000)
arxiv = ArxivQueryRun(api_wrapper = arxiv_wrapper)

## Creating wiki tool using WikiQueryRun and wikiAPIWrapper to bind it 
wiki_wrapper = WikipediaAPIWrapper(top_k_results= 2,doc_content_chars_max=2000)
wiki = WikipediaQueryRun(api_wrapper= wiki_wrapper)

## Creating search tool using DuckDuckGo tool
search = DuckDuckGoSearchRun(name="search")

## Title of App
st.title("Langchain -- Chat with Search Engine")
""" 
In this example ,we are using "StreamlitCallbackHandler" to display the thoughts and actions of an agent
in an interactive Streamlit App.
"""

## Creating messages list that will have the default role of agent and message that gets displayed to teh user on start screen

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content" : "I am a Chatbot that can answer your query searching online. How can I help you Today?"}
    ]

##To store every role and content pair as per converstaion in session_state[messages] going forward
## As here agent will take diff roles like user or assistant 

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.chat_message(msg["role"]).write(msg["content"])

## Creating a user prompt and storing in st.session_state.messages with role,content pair
## role for user query will be ,role:user, content : user query /prompt
# this is our default prompt with some default user query
import time

if prompt := st.chat_input(placeholder="What is Machine Learning?"):
    st.session_state.messages.append({"role": "user", "content" : prompt})
    st.chat_message("user").write(prompt)

    llm = ChatGroq(groq_api_key = groq_api_key, model = "Llama3-8b-8192", streaming= True)

    tools = [arxiv,wiki,search]

    # Zero_shot_react_desc -- It doesn't use/rely on chat_history,it atcs based on current input only
    # Chat_zero_shot_react_desc -- It uses the chat_history to act and expects a certain structure to it or throws error
    search_agent = initialize_agent(tools=tools,llm=llm,agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                                    handle_parsing_errors= True,)
    
    ## Calling Agent for "assistant" role of it to perform search
    ## Using StreamlitCallbackHandler() here to know the thoughts and actions of agent to search a user query
    ## st.container() allows multiple elements to be used without any order

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)

        try:
            time.sleep(1.5)
            response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
        except DuckDuckGoSearchException as e:
            response = "Rate limited by DuckDuckGo. Please wait a moment and try again"
            print("Rate limited by DuckDuckGo:",e)
        except OutputParserException as e:
            response = "⚠️ There was a problem understanding the agent's output. Please refine your question."
            print("Parsing Error:", e)

        except Exception as e:
            response = f"⚠️ An unexpected error occurred: {str(e)}"

            print("Unexpected Error:", e)

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)





