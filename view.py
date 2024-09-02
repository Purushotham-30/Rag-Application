import gradio as gr

def generate_output(prompt, history):
    from main import user_input
    return user_input(prompt)
    
def total_customisation(file, chunk_range, chunk_over_lap, select_db, select_embedding_model):
    from main import customisation
    return customisation(file, chunk_range, chunk_over_lap, select_db, select_embedding_model)

with gr.Blocks() as allign:
    with gr.Row():
        with gr.Column(scale=1): 
            gr.Markdown('# Upload File')
            pdf_file_input = gr.File(label='upload pdf')
            gr.Markdown('## select chunk size')
            chunk_range = gr.Slider(minimum=100, maximum=4000, step=1, label='chunk size')
            gr.Markdown('## chunk overlap')
            chunk_over_lap = gr.Slider(minimum=20, maximum=400, step=1, label='chunk over lap')
            gr.Markdown('## Data Base')
            select_db = gr.Radio(['pinecone', 'chromadb'], label='DataBases', info='Our services')
            gr.Markdown('## EmbeddingModel')
            select_embedding_model = gr.Dropdown(['all-MiniLM-L6-v2', 
                                                  'paraphrase-multilingual-MiniLM-L12-v2', 
                                                  'all-mpnet-base-v2', 'all-MiniLM-L12-v2'],
                                                  label='embedding models')
            submit_button = gr.Button('Submit')
            submit_button.click(
                fn=total_customisation,
                inputs=[pdf_file_input, chunk_range, chunk_over_lap, select_db, select_embedding_model]
            )

        with gr.Column(scale=4):
            iface = gr.ChatInterface(
                fn=generate_output,
                chatbot=gr.Chatbot(),
                title="ChatWithYourBooks",
                description="Discover, Discuss, and Delight in Every Page"
            )

allign.launch()