import gradio as gr

from .emails.download import demo as download_demo
from .emails.label import demo as email_demo

demo = gr.TabbedInterface([download_demo, email_demo], ["Download", "Label"])

if __name__ == "__main__":
    demo.launch()
