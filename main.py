import boto3
import streamlit as st
from langchain_aws import ChatBedrock, BedrockEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

prompt_template = """

Human: Use the following pieces of context to provide a concise 
answer to the question. Summarize with 250 words with detailed explanations.
If you don't know the answer, just say you don't know, making sure to say it in a polite tone and manner.
Don't make up information and don't use your own knowledge to answer queries, make sure to use the context.
Remember: if the context is insufficient to answer the user query then say you don't know the answer. DO NOT make up information.\n\n
<context>
{context}
</context>

Question: {input}\n\n
Assistant:"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Initialize AWS Bedrock client
bedrock_client = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")

# Get the embedding model
embed_model_id="amazon.titan-embed-text-v2:0"
bedrock_embeddings = BedrockEmbeddings(model_id=embed_model_id, client=bedrock_client)

def get_documents():
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500,
                                                   chunk_overlap=150)
    
    docs = text_splitter.split_documents(documents)
    return docs

def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(
        docs,
        bedrock_embeddings
    )
    vectorstore_faiss.save_local("faiss_index")

def get_llm(model_id, bedrock_client, provider: str, temperature: float = 0.3, max_gen_len: int = 512):
    # Instantiate Bedrock LLM
    llm = ChatBedrock(
    model_id=model_id,
    client=bedrock_client,
    provider=provider,
    model_kwargs={"temperature": temperature, "max_gen_len": max_gen_len})

    return llm


def get_llm_respose(llm, vectorstore_faiss, prompt, user_question):
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(vectorstore_faiss.as_retriever(search_type='similarity', search_kwargs={"k":3}), combine_docs_chain)
    response = rag_chain.invoke({"input": user_question})

    return response['answer']

def main():
    st.set_page_config("RAG Demo")
    st.header("RAG PDF Q&A App using Amazon Bedrock")
    user_question = st.text_input("Ask me a question about your PDFs")

    with st.sidebar:
        st.title("Update or Create Vector Store")

        if st.button("Store Vectors"):
            with st.spinner("Processing..."):
                docs = get_documents()
                get_vector_store(docs)
                st.success("Done")

    if st.button("Send Query"):
        with st.spinner("Thinking..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
            llm = get_llm(model_id="arn:aws:bedrock:us-east-1:464209272334:inference-profile/us.meta.llama3-1-8b-instruct-v1:0",
                            bedrock_client=bedrock_client, provider="meta")
            
            st.write(get_llm_respose(llm, faiss_index, PROMPT, user_question))

if __name__=="__main__":
    main()

        
                








