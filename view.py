import gradio as gr
from main import user_input, upload_file

def generate_output(prompt, history):
    return user_input(prompt)

def get_file_path(file):
    return file.name

with gr.Blocks() as allign:
    with gr.Row():
        with gr.Column(scale=1): 
            gr.Markdown('# Upload File')
            pdf_file_input = gr.File(label='upload pdf')
            pdf_file_input.change(
                fn=upload_file,
                inputs=pdf_file_input
            )
        with gr.Column(scale=4):
            iface = gr.ChatInterface(
                fn=generate_output,
                chatbot=gr.Chatbot(
                    placeholder='<h1 style="color : blue;font-size: 77px;">Welcome<h1>'
                ),
                title="ChatWithYourBooks",
                description="Discover, Discuss, and Delight in Every Page"
            )

allign.launch()