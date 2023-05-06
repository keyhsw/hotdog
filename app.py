import gradio as gr
import os
from transformers import pipeline

pipeline = pipeline(task="image-classification", model="julien-c/hotdog-not-hotdog")
def predict(image):
    predictions = pipeline(image)
    return {p["label"]: p["score"] for p in predictions}


title = """<h1 align="center">Demo of MiniGPT-4</h1>"""
description = """<h3>This is the demo of MiniGPT-4. Upload your images and start chatting!</h3>"""
article = """<div style='display:flex; gap: 0.25rem; '><a href='https://minigpt-4.github.io'><img src='https://img.shields.io/badge/Project-Page-Green'></a><a href='https://github.com/Vision-CAIR/MiniGPT-4'><img src='https://img.shields.io/badge/Github-Code-blue'></a><a href='https://github.com/TsuTikgiau/blip2-llm/blob/release_prepare/MiniGPT_4.pdf'><img src='https://img.shields.io/badge/Paper-PDF-red'></a></div>
"""

with gr.Blocks() as demo:
    gr.Markdown(title)
    gr.Markdown(SHARED_UI_WARNING)
    gr.Markdown(description)
    gr.Markdown(article)

demo.Interface(
    predict,
    inputs=gr.inputs.Image(label="Upload hot dog candidate", type="filepath"),
    outputs=gr.outputs.Label(num_top_classes=2),
#     title="Hot Dog? Or Not?"
).launch()
