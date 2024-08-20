from uuid import uuid4
import os
from dotenv import load_dotenv
from langchain_community.document_loaders.pdf import PyMuPDFLoader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from embeddingmodels import TransformerEmbedder 
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import chromadb
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma


qa_chain = None

load_dotenv(dotenv_path=f'{os.getcwd()}/helpers/.env')

#Read pdf data
def read_pdf(path):
    pdf_load = PyMuPDFLoader(path)
    list_pdf_data = pdf_load.load()
    text_data = ""
    for data in list_pdf_data:
        text_data+=data.page_content                                                                                                                                                                                                                                                                                                                                                               
    return text_data

# chunk data
def create_chunks(chunksize, chunkoverlap, text_data):
    text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size = chunksize,
                            chunk_overlap = chunkoverlap
                    )
    chunk_text_data = text_splitter.split_text(text_data)
    return chunk_text_data

# text_data = read_pdf("C:\\Users\\AIFA USER 03.LAPTOP-G1Q1AF36\\Downloads\\Mayo Oshin & Nuno Campos - Learning LangChain (for Raymond Rhine)-O'Reilly Media (2024).pdf")

def chroma_embeddings(pdf_data):
    # generate embeddings
    collection_name = str(uuid4())
    chunks_text = create_chunks(1000, 50, pdf_data)
    model = TransformerEmbedder('all-MiniLM-L6-v2')
    vectors_embeddings = model.generate_embeddings(chunks_text)
    client = chromadb.Client()
    collection = client.create_collection(collection_name)
    collection.add(
        documents=chunks_text,
        embeddings=vectors_embeddings.tolist(),
        ids=[str(i)for i in range(len(vectors_embeddings))]
    )
    return collection_name

def retrival_data(coll_name):
    db_model = SentenceTransformerEmbeddings(model_name = "all-MiniLM-L6-v2")
    retriver = Chroma(collection_name=coll_name, embedding_function=db_model).as_retriever()
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro",api_key=os.getenv('GOOGLE_API_KEY'),top_k=6)
    qa_chain = RetrievalQA.from_chain_type(llm = llm, retriever = retriver, chain_type = 'stuff')
    return qa_chain
 
def upload_file(file):
    pdf_data = read_pdf(file.name)
    collection_name = chroma_embeddings(pdf_data)
    global qa_chain
    qa_chain = retrival_data(collection_name)

def user_input(user): 
    response = qa_chain({"query":user})
    print(response['result'])
    return response['result']
