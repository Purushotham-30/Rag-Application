import os
from dotenv import load_dotenv
from langchain_community.document_loaders.pdf import PyMuPDFLoader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from helpers.embeddingmodels import TransformerEmbedder 
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from pinecone import Pinecone,ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from helpers.chroma_db import chroma_database
from helpers.pinecone_db import pinecone_database

qa_chain = None
chunk_size = 700
chunk_over_lap = 50
list_pdf_text = None
db = None
embed_model = None

load_dotenv(dotenv_path=f'{os.getcwd()}/helpers/.env')

def read_pdf(path):
    '''
    This function would read the pdf
    '''
    pdf_load = PyMuPDFLoader(path)
    list_pdf_data = pdf_load.load()
    global list_pdf_text
    list_pdf_text = list_pdf_data
    text_data = ""
    for data in list_pdf_data:
        text_data+=data.page_content                                                                                                                                                                                                                                                                                                                                                               
    return text_data

def create_chunks(text_data):
    '''
    This function would create chunks
    '''
    text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size = chunk_size,
                            chunk_overlap = chunk_over_lap
                    )
    chunk_text_data = text_splitter.split_text(text_data)
    return chunk_text_data


def retrival_data(coll_name):
    db_model = SentenceTransformerEmbeddings(model_name = embed_model)
    retriver = None
    if db == 'pinecone':
        retriver = PineconeVectorStore.from_existing_index(index_name=coll_name, embedding=db_model).as_retriever()
    else:
        retriver = Chroma(collection_name=coll_name, embedding_function=db_model).as_retriever()
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro",api_key=os.getenv('GOOGLE_API_KEY'),top_k=6)
    qa_chain = RetrievalQA.from_chain_type(llm = llm, retriever = retriver, chain_type = 'stuff')
    return qa_chain
 
# gemini-1.5-pro
# gemini-1.5-flash-001
def upload_file(file):
    pdf_data = read_pdf(file.name)
    chunks_text = create_chunks(pdf_data)
    collection_name = ''
    if db == 'pinecone':
        collection_name = pinecone_database(embed_model, chunks_text)
    else:
        collection_name = chroma_database(embed_model, chunks_text)
    global qa_chain
    qa_chain = retrival_data(collection_name)

def user_input(user):
    response = qa_chain({"query":user})
    return response['result']

def customisation(file, chunksize, chunk_overlap, database, embedding_model):
    global chunk_size, chunk_over_lap, db, embed_model
    chunk_size = chunksize
    chunk_over_lap = chunk_overlap
    db = database
    embed_model = embedding_model
    upload_file(file)
