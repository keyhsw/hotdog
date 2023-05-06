import gradio as gr
import os
from transformers import pipeline

pipeline = pipeline(task="image-classification", model="julien-c/hotdog-not-hotdog")
def predict(image):
    predictions = pipeline(image)
    return {p["label"]: p["score"] for p in predictions}


title = """<h1 align="center">Demo of MiniGPT-4</h1>"""
description = """<h3>This is the demo of MiniGPT-4. Upload your images and start chatting!</h3>"""
# article = """<div style='display:flex; gap: 0.25rem; '><a href='https://minigpt-4.github.io'><img src='https://img.shields.io/badge/Project-Page-Green'></a><a href='https://github.com/Vision-CAIR/MiniGPT-4'><img src='https://img.shields.io/badge/Github-Code-blue'></a><a href='https://github.com/TsuTikgiau/blip2-llm/blob/release_prepare/MiniGPT_4.pdf'><img src='https://img.shields.io/badge/Paper-PDF-red'></a></div>
# """
article = """<div align="center">
      <a href="https://openmmlab.medium.com/" style="text-decoration:none;">
        <img src="https://user-images.githubusercontent.com/25839884/219255827-67c1a27f-f8c5-46a9-811d-5e57448c61d1.png" width="3%" alt="" /></a>
      <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
      <a href="https://discord.com/channels/1037617289144569886/1046608014234370059" style="text-decoration:none;">
        <img src="https://user-images.githubusercontent.com/25839884/218347213-c080267f-cbb6-443e-8532-8e1ed9a58ea9.png" width="3%" alt="" /></a>
      <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
      <a href="https://twitter.com/OpenMMLab" style="text-decoration:none;">
        <img src="https://user-images.githubusercontent.com/25839884/218346637-d30c8a0f-3eba-4699-8131-512fb06d46db.png" width="3%" alt="" /></a>
      <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
      <a href="https://www.youtube.com/openmmlab" style="text-decoration:none;">
        <img src="https://user-images.githubusercontent.com/25839884/218346691-ceb2116a-465a-40af-8424-9f30d2348ca9.png" width="3%" alt="" /></a>
      <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
      <a href="https://space.bilibili.com/1293512903" style="text-decoration:none;">
        <img src="https://user-images.githubusercontent.com/25839884/219026751-d7d14cce-a7c9-4e82-9942-8375fca65b99.png" width="3%" alt="" /></a>
      <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
      <a href="https://www.zhihu.com/people/openmmlab" style="text-decoration:none;">
        <img src="https://user-images.githubusercontent.com/25839884/219026120-ba71e48b-6e94-4bd4-b4e9-b7d175b5e362.png" width="3%" alt="" /></a>
    </div>"""

# with gr.Blocks() as demo:
#     gr.Markdown(title)
# #     gr.Markdown(SHARED_UI_WARNING)
#     gr.Markdown(description)
#     gr.Markdown(article)


gr.Interface(
    predict,
    inputs=gr.inputs.Image(label="Upload hot dog candidate", type="filepath"),
    outputs=gr.outputs.Label(num_top_classes=2),
    title="Hot Dog? Or Not?",
    article = article,
).launch()
