import os
from pinecone import Pinecone,ServerlessSpec
from helpers.embeddingmodels import TransformerEmbedder
from dotenv import load_dotenv

load_dotenv(dotenv_path=f'{os.getcwd()}/helpers/.env')

def pinecone_database(embed_model, chunks_text):
    '''
    This would store the vector embeddings in Pinecone database
    '''
    index_name = "docs-rag-testchatbot"
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