import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate 
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_core.runnables import RunnablePassthrough

from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
print(api_key)

llm = GoogleGenerativeAI(google_api_key=api_key, model='gemini-1.5-flash', temperature=0.9)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


instructor_embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001", task_type="retrieval_query"
)


vectordb_file_path = "faiss_index"


def create_vector_db():
    
    loader = CSVLoader(file_path="project.csv", source_column="States + UTs")
    data = loader.load()

    
    vectordb = FAISS.from_documents(documents=data, embedding=instructor_embeddings)
    vectordb.save_local(vectordb_file_path)


def get_qa_chain():
    vectordb = FAISS.load_local(vectordb_file_path, instructor_embeddings, allow_dangerous_deserialization=True)

    
    retriever = vectordb.as_retriever(score_threshold=0.7, top_k=5)

    prompt_template = """
   You are an expert assistant in tourism and travel planning. Based on the following context retrieved from a document, generate a complete and detailed answer to the user's question.

   Make sure the response is well-structured, includes practical recommendations, and explains the content clearly. Avoid using phrases such as "the document says" and aim to provide a natural response.

   CONTEXT: {context}

  QUESTION: {question}

   Answer as thoroughly as possible, combining information from the context and offering additional suggestions if relevant. If the answer is not found in the context, say "I don't know" and avoid creating false information.
   If information not found but a keyword is found create information regarding it. DO NOT mention not found in document or the text does not have information or text doesn't specify the exact or any such message in the output text.
   Answer in as detail as possible prefer points.
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
    )

    return chain
