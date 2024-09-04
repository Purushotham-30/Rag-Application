from sentence_transformers import SentenceTransformer

class TransformerEmbedder:
    def __init__(self, model : str = 'all-MiniLM-L6-v2'):
        self.model = model
    
    def generate_embeddings(self, chunk_data):
        model = SentenceTransformer(self.model)
        vector_embeddings = model.encode(chunk_data)
        return vector_embeddings


