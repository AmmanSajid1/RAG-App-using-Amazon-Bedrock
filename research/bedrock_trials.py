from langchain_aws import ChatBedrock  
from langchain.prompts import PromptTemplate
import boto3
import streamlit as st

# Initialize AWS Bedrock client
bedrock_client = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")

# Define model ID
model_id = "arn:aws:bedrock:us-east-1:464209272334:inference-profile/us.meta.llama3-1-8b-instruct-v1:0"

# Instantiate Bedrock LLM
llm = ChatBedrock(
    model_id=model_id,
    client=bedrock_client,
    provider="meta",
    model_kwargs={"temperature": 0.9}
)

# Define chatbot function
def my_chatbot(language, user_text):
    # Create prompt template
    prompt_template = PromptTemplate(
        input_variables=["language", "user_text"],
        template="You are a chatbot. You will respond to users in {language}.\n\n{user_text}"
    )
    
    chain = prompt_template | llm
    response = chain.invoke({"language": language, "user_text": user_text})  

    return response.content

# Streamlit UI
st.title("Bedrock Demo")

# Sidebar input options
language = st.sidebar.selectbox("Language", ["english", "spanish", "urdu", "french"])
user_text = st.sidebar.text_area(label="What is your query?", max_chars=100)
button = st.sidebar.button("Submit")

# Handle button click
if user_text and button:
    response = my_chatbot(language, user_text)
    st.write(response)  
else:
    st.write("Please enter a query")
