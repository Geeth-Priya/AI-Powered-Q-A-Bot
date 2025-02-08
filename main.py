from flask import Flask, request, jsonify
from flask_restful import Api, Resource
from langchain_community.document_loaders import WebBaseLoader
import bs4
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_ollama.llms import OllamaLLM
import re

app = Flask(__name__)
api = Api(app)

@app.route('/')
def home():
    return "Welcome to the Flask API! You can ask questions at /ask."

#content upload
bs4_strainer = bs4.SoupStrainer(class_="courses-content")
loader = WebBaseLoader(
    web_paths=("https://brainlox.com/courses/category/technical",),
    bs_kwargs={"parse_only": bs4_strainer},
)
docs = loader.load()

#content Split
rts = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, add_start_index=True)
all_splits = rts.split_documents(docs)

# embeddings and storing in vector database
local_embeddings = OllamaEmbeddings(model="all-minilm")
vectorscore = Chroma.from_documents(documents=all_splits, embedding=local_embeddings)

pattern = r"\d+ LessonsView Details"

# Question Answering
class QuestionAnswer(Resource):
    def post(self):
        # Get the question from the request
        data = request.get_json()
        question = data.get("question", "")

        if not question:
            return {"error": "No question provided"}, 400

        # Retrieve relevant documents using similarity search
        retriever = vectorscore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        retrieved_docs = retriever.invoke(question)

        cleaned_docs = [re.sub(pattern, "", doc.page_content) for doc in retrieved_docs]
        context = " ".join(cleaned_docs)

        # Generating answer
        llm = OllamaLLM(model="llama3.2:1b")
        response = llm.invoke(f"""
            Answer the question briefly based on the context below. Focus on providing a concise and informative answer:
            Question: {question}
            Context: {context}
        """)

        refined_answer=re.sub(r"^Courses mentioned:", "", response).strip()

        return jsonify({"answer": response})

# Add the Q&A resource to the API
api.add_resource(QuestionAnswer, '/ask')

if __name__ == '__main__':
    app.run(debug=True)
