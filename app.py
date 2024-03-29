import gradio as gr
import os
from transformers import pipeline

# from openxlab.model import download
# download(model_repo='thomas-yanxin/MindChat-InternLM-7B', model_name=['pytorch_model-00001-of-00002.bin'])

# from openxlab.model import download
# download(model_repo='thomas-yanxin/MindChat-InternLM-7B', 
# model_name='pytorch_model-00002-of-00002.bin')

pipeline = pipeline(task="image-classification", model="julien-c/hotdog-not-hotdog")
def predict(image):
    predictions = pipeline(image)
    return {p["label"]: p["score"] for p in predictions}

gr.Interface(
    predict,
    inputs=gr.inputs.Image(label="Upload hot dog candidate", type="filepath"),
    outputs=gr.outputs.Label(num_top_classes=2),
    title="Hot Dog? Or Not?",
).launch()
