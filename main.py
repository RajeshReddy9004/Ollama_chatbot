from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st 
import os

from dotenv import load_dotenv

load_dotenv()

# langsmith tracking
os.environ['LANGCHAIN_API_KEY']=os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGCHAIN_TRACING_V2']="true"
os.environ["LANGCHAIN_PROJECT"]="Simple Q&A Chatbot with OLLAMA"

# prompt template
# prompt=ChatPromptTemplate.from_messages(
#      ("system","you are a helpful assisstant. Please response to the user queries"),
#         ("user","Question:{question}")
# )

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("You are a helpful assistant. Please respond to the user queries."),
    HumanMessagePromptTemplate.from_template("Question: {question}")
])


def generate_response(question,engine,temperature,max_tokens):
    # openai.api_key=api_key
    # llm=ChatOpenAI(model=llm)
    llm = Ollama(model=engine)
    output_parser=StrOutputParser()
    chain=prompt|llm|output_parser
    answer=chain.invoke({'question':question})
    return answer


llm = st.sidebar.selectbox("Select OpenAI Model:", ["mistral"])


temperature=st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=0.7)
max_tokens=st.sidebar.slider("Max tokens",min_value=50,max_value=300,value=150)

    
st.write("Go ahead and ask the question")
user_input=st.text_input("You:")

if user_input:
    response=generate_response(user_input,llm,temperature,max_tokens)
    st.write(response)
else:
    st.write("Please provide the information")