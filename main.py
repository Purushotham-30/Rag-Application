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
from pinecone import Pinecone,ServerlessSpec
from langchain_pinecone import PineconeVectorStore

qa_chain = None
chunk_size = 700
chunk_over_lap = 50
list_pdf_text = None
db = None
embed_model = None

load_dotenv(dotenv_path=f'{os.getcwd()}/hlepers/.env')

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

def chroma_embeddings(pdf_data):
    '''
    This function would generate embeddings
    '''
    collection_name = 'pdf_data'
    chunks_text = create_chunks(pdf_data)
    print(embed_model)
    model = TransformerEmbedder(embed_model)
    vectors_embeddings = model.generate_embeddings(chunks_text)
    client = chromadb.Client()
    collection = client.get_or_create_collection(collection_name)
    collection.upsert(
        documents=chunks_text,
        embeddings=vectors_embeddings.tolist(),
        ids=[str(i)for i in range(len(vectors_embeddings))]
    )
    return collection_name

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
    collection_name = ''
    if db == 'pinecone':
        collection_name = pinecone_db(pdf_data)
    else:
        collection_name = chroma_embeddings(pdf_data)
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

def pinecone_db(pdf_data):
    '''
    This would store the vector embeddings in Pinecone database
    '''
    index_name = "docs-rag-testchatbot"
    chunks_text = create_chunks(pdf_data)
    print(embed_model)
    model = TransformerEmbedder(model)
    vectors_embeddings = model.generate_embeddings(chunks_text)
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    # pinecone.init(api_key=os.getenv('PINECONE_API_KEY'), environment='us-west1-gcp')
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=384, 
            metric="cosine", 
            spec=ServerlessSpec(
                cloud="aws", 
                region="us-east-1"
            ) 
        )
    index = pc.Index(index_name)
    ids = [f'id_{i}' for i in range(len(chunks_text))]
    metadata = [{'text' : data} for data in chunks_text]
    upsert_data = list(zip(ids, vectors_embeddings, metadata))
    index.upsert(vectors=upsert_data)
    return index_name