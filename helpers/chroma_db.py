from helpers.embeddingmodels import TransformerEmbedder
import chromadb

def chroma_database(embed_model, chunks_text):
    collection_name = 'pdf_data'
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