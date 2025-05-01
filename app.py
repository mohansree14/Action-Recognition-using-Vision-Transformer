import streamlit as st
import torch
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import re
from PIL import Image
from transformers import AutoFeatureExtractor, AutoModelForVideoClassification

# Load model
@st.cache_resource
def load_model():
    model = AutoModelForVideoClassification.from_pretrained("facebook/timesformer-base-finetuned-k400")
    extractor = AutoFeatureExtractor.from_pretrained("facebook/timesformer-base-finetuned-k400")
    return model, extractor

model, extractor = load_model()
model.eval()

# Interface
st.title("Action Recognition from Image Frames")
st.success("App loaded successfully")
st.write("Paste the full path to a folder that contains 8 image frames (e.g., 0001.jpg to 0008.jpg).")

folder_path = st.text_input("Enter full path to frame folder:")

if folder_path:
    if not os.path.isdir(folder_path):
        st.error("Invalid folder path. Please check and try again.")
    else:
        image_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".jpg")])[:8]
        frames = []

        for img_name in image_files:
            img_path = os.path.join(folder_path, img_name)
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (224, 224))
            frames.append(image)

        if len(frames) < 8:
            st.warning("The folder must contain at least 8 .jpg image frames.")
        else:
            inputs = extractor([frames], return_tensors="pt")

            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                top5 = torch.topk(probs, k=5)

            st.subheader(" Top-5 Predicted Actions")
            for i, (score, idx) in enumerate(zip(top5.values[0], top5.indices[0])):
                label = model.config.id2label[idx.item()]
                st.write(f"{i+1}. **{label}** — {score.item()*100:.2f}%")

            # Plot bar chart
            labels = [model.config.id2label[idx.item()] for idx in top5.indices[0]]
            scores = [score.item() * 100 for score in top5.values[0]]
            st.subheader(" Top-5 Predictions (Bar Chart)")
            fig, ax = plt.subplots()
            ax.barh(labels[::-1], scores[::-1])
            ax.set_xlabel("Confidence (%)")
            ax.set_title("Top-5 Action Predictions")
            st.pyplot(fig)

else:
    st.info("Paste a valid folder path above to start prediction.")

# Confusion Matrix (optional image display)
st.subheader("Confusion Matrix")
try:
    conf_img = Image.open("images/confusion_matrix.png")
    st.image(conf_img, caption="Confusion Matrix", use_column_width=True)
except:
    st.warning("Confusion matrix image not found. Please place it in /images.")

# Training log viewer
st.subheader(" Training Log Output")
try:
    with open("logs/training.log", "r") as file:
        log_text = file.read()
    st.text_area("Training Log", log_text, height=300)
except:
    st.warning("Training log file not found.")

# Training loss curve
st.subheader(" Training Loss Curve")
try:
    losses = []
    with open("logs/training.log", "r") as file:
        for line in file:
            match = re.search(r"Loss: ([0-9.]+)", line)
            if match:
                losses.append(float(match.group(1)))
    if losses:
        st.line_chart(losses)
    else:
        st.warning("No loss values found to plot.")
except:
    st.warning("Could not read training log for plotting.")
