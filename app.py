from flask import Flask, request, jsonify
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from HTMLLoader import html_splitter
from langchain_together import Together
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
import os
from dotenv import load_dotenv
from flask_cors import CORS
import threading

app = Flask(__name__)
CORS(app)
load_dotenv()

# Initialize global variables to store preloaded content
preloaded_content = None
vectorstore = None

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

prompt = PromptTemplate(
    input_variables=['context', 'question'],
    template="Your name is Kanha. You are an assistant for question answering tasks. Use the following pieces of retrieved context and your own knowledge to answer the question. If you don't know the answer, just say that you don't know. Keep answers descriptive with bullet points and mention the process.\nQuestion: {question} \nContext: {context} \nAnswer:"
)

@app.route('/')
def index():
    return "Your chat guide Kanha is here"

@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.route('/preload', methods=['POST'])
def preload_content():
    global preloaded_content, vectorstore
    data = request.json
    url = data['url']

    # Dynamically update the URLs list
    urls = [url]

    # Split and store the page content
    splits = html_splitter(urls)
    model_name = "BAAI/bge-base-en-v1.5"
    encode_kwargs = {"normalize_embeddings": False}
    embedding_function = HuggingFaceBgeEmbeddings(model_name=model_name, encode_kwargs=encode_kwargs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_function, persist_directory="../chroma_db")
    
    preloaded_content = vectorstore.as_retriever(search_kwargs={"k": 2})
    
    return jsonify({"message": "Content preloaded successfully"})

@app.route('/query', methods=['POST'])
def handle_query():
    global preloaded_content
    data = request.json
    query = data['query']

    if preloaded_content is None:
        return jsonify({"error": "Content not preloaded"}), 400

    response = Together(
        model="mistralai/Mistral-7B-Instruct-v0.2",
        together_api_key=os.getenv("TOGETHER_API_KEY"),
        temperature=0.3,
        max_tokens=512
    )

    rag_chain = (
        {"context": preloaded_content | format_docs, "question": RunnablePassthrough()}
        | prompt
        | response
    )
    answer = rag_chain.invoke(query)
    return jsonify({"answer": answer})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
