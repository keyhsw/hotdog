import gradio as gr
import os
from transformers import pipeline

# os.system(1/0)

pipeline = pipeline(task="image-classification", model="julien-c/hotdog-not-hotdog")

def predict(image):
#     a = 1 / 0
    predictions = pipeline(image)
    return {p["label"]: p["score"] for p in predictions}

gr.Interface(
    predict,
    inputs=gr.inputs.Image(label="Upload hot dog candidate", type="filepath"),
    outputs=gr.outputs.Label(num_top_classes=2),
    title="Hot Dog? Or Not?",
).launch()
