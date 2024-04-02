import streamlit as st
import openai

                
import pinecone
import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

# Check if environment variables are present. If not, throw an error
if os.getenv('PINECONE_API_KEY') is None:
    st.error("PINECONE_API_KEY not set. Please set this environment variable and restart the app.")
if os.getenv('PINECONE_REGION') is None:
    st.error("PINECONE_REGION not set. Please set this environment variable and restart the app.")
if os.getenv('PINECONE_INDEX_NAME') is None:
    st.error("PINECONE_INDEX_NAME not set. Please set this environment variable and restart the app.")
if os.getenv('OPENAI_API_KEY') is None:
    st.error("OPENAI_API_KEY not set. Please set this environment variable and restart the app.")
if os.getenv('OPENAI_BASE_URI') is None:
    st.error("OPENAI_BASE_URI not set. Please set this environment variable and restart the app.")
if os.getenv('OPENAI_EMBEDDINGS_MODEL_NAME') is None:
    st.error("OPENAI_EMBEDDINGS_MODEL_NAME not set. Please set this environment variable and restart the app.")
if os.getenv('OPENAI_MODEL_NAME') is None:
    st.error("OPENAI_MODEL_NAME not set. Please set this environment variable and restart the app.")



st.title("교육행정 질의응답 시스템 🔎")
query = st.text_input("무엇을 도와드릴까요??")

if st.button("질문하기"):
   
    # # get Pinecone API environment variables
    pinecone_api = os.getenv('PINECONE_API_KEY')
    pinecone_env = os.getenv('PINECONE_REGION')
    pinecone_index = os.getenv('PINECONE_INDEX_NAME')

    
    # # get Azure OpenAI environment variables
    openai.api_key = os.getenv('OPENAI_API_KEY')
    openai.api_base = os.getenv('OPENAI_BASE_URI')
    embeddings_model_name = os.getenv('OPENAI_EMBEDDINGS_MODEL_NAME')
    model_name = os.getenv('OPENAI_MODEL_NAME')
    

  # Pinecone 클라이언트 인스턴스 생성
    index = pinecone.Index(api_key='91c3f1fb-7746-47b6-8060-4f454ef29401', host='https://edudata2-yhu9kdz.svc.apw5-4e34-81fa.pinecone.io')

# 인덱스에 벡터 삽입, 조회, 업데이트 등의 작업 수행

 
    # Convert your query into a vector using Azure OpenAI
    try:
        query_vector = openai.Embedding.create(input=query, engine="text-embedding-ada-002")["data"][0]["embedding"]
    except Exception as e:
        st.error(f"Error calling OpenAI Embedding API: {e}")
        st.stop()
 
    # Search for the most similar vectors in Pinecone
    search_response = index.query(
        top_k=3,
        vector=query_vector,
        include_metadata=True)

    chunks = [item["metadata"]['text'] for item in search_response['matches']]
 
    # Combine texts into a single chunk to insert in the prompt
    joined_chunks = "\n".join(chunks)

    # Write the selected chunks into the UI
    with st.expander("Chunks"):
        for i, t in enumerate(chunks):
            t = t.replace("\n", " ")
            st.write("Chunk ", i, " - ", t)
    
    with st.spinner("Summarizing..."):
        try:
            # Build the prompt
            prompt = f"""
            Answer the following question based on the context below.
            If you don't know the answer, just say that you don't know. Don't try to make up an answer. Do not answer beyond this context.
            ---
            QUESTION: {query}                                            
            ---
            CONTEXT:
            {joined_chunks}
            """
 
            # Run chat completion using GPT-4
            response = openai.ChatCompletion.create(
                engine=model_name,
                messages=[
                    { "role": "system", "content":  "You are a Q&A assistant." },
                    { "role": "user", "content": prompt }
                ],
                temperature=0,
                max_tokens=1000
            )
 
            # Write query answer
            st.markdown("### Answer:")
            st.write(response.choices[0]['message']['content'])
   
   
        except Exception as e:
            st.error(f"Error with OpenAI Chat Completion: {e}")


