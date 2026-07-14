### FoodVision Big — Streamlit app ###
import os
from timeit import default_timer as timer
from typing import Dict, Tuple

import streamlit as st
import torch
from PIL import Image

from model import create_effnetb2_model

MODEL_WEIGHTS_PATH = "09_pretrained_effnetb2_feature_extractor_food101_20_percent.pth"

# ---------- Setup (cached so it only runs once) ----------

@st.cache_resource
def load_model_and_transforms():
    with open("class_names.txt", "r") as f:
        class_names = [food_name.strip() for food_name in f.readlines()]

    model, transforms = create_effnetb2_model(num_classes=len(class_names))

    # Load trained weights (map to CPU since Streamlit Cloud has no GPU)
    model.load_state_dict(
        torch.load(f=MODEL_WEIGHTS_PATH, map_location=torch.device("cpu"))
    )
    model.eval()

    return model, transforms, class_names


model, effnetb2_transforms, class_names = load_model_and_transforms()

# ---------- Predict function ----------

def predict(img: Image.Image) -> Tuple[Dict[str, float], float]:
    start_time = timer()

    img_transformed = effnetb2_transforms(img).unsqueeze(0)  # add batch dim

    with torch.inference_mode():
        pred_probs = torch.softmax(model(img_transformed), dim=1)

    pred_labels_and_probs = {
        class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))
    }

    pred_time = round(timer() - start_time, 4)
    return pred_labels_and_probs, pred_time


# ---------- UI ----------

st.set_page_config(page_title="FoodVision Big", page_icon="🍔")

st.title("FoodVision Big 🍔👁💪")
st.markdown(
    "An [EfficientNetB2 feature extractor]"
    "(https://pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b2.html) "
    "computer vision model to classify images into "
    "[101 classes of food from the Food101 dataset]"
    "(https://github.com/mrdbourke/pytorch-deep-learning/blob/main/extras/food101_class_names.txt)."
)

uploaded_file = st.file_uploader(
    "Upload a food image", type=["jpg", "jpeg", "png"]
)

# Optional: example images shown as clickable thumbnails
example_dir = "examples"
selected_example = None
if os.path.isdir(example_dir):
    example_files = os.listdir(example_dir)
    if example_files:
        st.write("Or try an example:")
        cols = st.columns(len(example_files))
        for col, example_file in zip(cols, example_files):
            example_path = os.path.join(example_dir, example_file)
            with col:
                st.image(example_path, width=100)
                if st.button("Use this", key=example_file):
                    selected_example = example_path

image_to_predict = None
if uploaded_file is not None:
    image_to_predict = Image.open(uploaded_file).convert("RGB")
elif selected_example is not None:
    image_to_predict = Image.open(selected_example).convert("RGB")

if image_to_predict is not None:
    st.image(image_to_predict, caption="Input image", use_container_width=True)

    with st.spinner("Predicting..."):
        pred_labels_and_probs, pred_time = predict(image_to_predict)

    # Sort and show top 5
    top_5 = dict(
        sorted(pred_labels_and_probs.items(), key=lambda x: x[1], reverse=True)[:5]
    )

    st.subheader("Predictions")
    for label, prob in top_5.items():
        st.write(f"**{label}**")
        st.progress(prob)
        st.caption(f"{prob:.2%}")

    st.info(f"Prediction time: {pred_time} seconds")

st.markdown(
    "---\n"
    "Created following [09. PyTorch Model Deployment]"
    "(https://www.learnpytorch.io/09_pytorch_model_deployment/)."
)
