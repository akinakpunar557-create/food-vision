import streamlit as st

st.set_page_config(
    page_title="Food Vision",
    page_icon="🍕",
    layout="centered",
)
import torch
from PIL import Image
from timeit import default_timer as timer
from typing import Tuple, Dict

from model import create_effnetb2_model

# ============================================================
# Setup
# ============================================================

with open("class_names.txt", "r") as f:
    class_names = [food_name.strip() for food_name in f.readlines()]

MODEL_WEIGHTS_PATH = "09_pretrained_effnetb2_feature_extractor_food101_20_percent.pth"

APP_TITLE = "🍽️ Food Vision"
APP_DESCRIPTION = (
    "Upload a food image and let an EfficientNetB2 model classify it!"
)

example_images = [
    "static/pizza.jpg",
]

# ============================================================
# Load model
# ============================================================

@st.cache_resource
def load_model():
    model, transforms = create_effnetb2_model(
        num_classes=len(class_names)
    )

    checkpoint = torch.load(
        MODEL_WEIGHTS_PATH,
        map_location="cpu",
        weights_only=False
    )

    model.load_state_dict(checkpoint)
    model.eval()

    return model, transforms


effnetb2, effnetb2_transforms = load_model()

# ============================================================
# Prediction
# ============================================================

def predict(img: Image.Image) -> Tuple[Dict, float]:
    start_time = timer()

    img_tensor = effnetb2_transforms(img).unsqueeze(0)

    with torch.inference_mode():
        pred_probs = torch.softmax(effnetb2(img_tensor), dim=1)

    pred_labels_and_probs = {
        class_names[i]: float(pred_probs[0][i])
        for i in range(len(class_names))
    }

    pred_time = round(timer() - start_time, 4)

    return pred_labels_and_probs, pred_time


# ============================================================
# Streamlit UI
# ============================================================



st.title(APP_TITLE)
st.write(APP_DESCRIPTION)

uploaded_file = st.file_uploader(
    "Upload a food image",
    type=["jpg", "jpeg", "png"]
)

if "selected_example" not in st.session_state:
    st.session_state.selected_example = None

if example_images:
    st.write("### Or try an example")

    cols = st.columns(len(example_images))

    for col, example_path in zip(cols, example_images):
        with col:
            st.image(example_path, use_column_width=True)

            if st.button("Use Image", key=example_path):
                st.session_state.selected_example = example_path

img = None
caption = None

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    caption = "Uploaded Image"
    st.session_state.selected_example = None

elif st.session_state.selected_example is not None:
    img = Image.open(st.session_state.selected_example).convert("RGB")
    caption = "Example Image"

if img is not None:

    st.image(img, caption=caption, use_column_width=True)

    with st.spinner("Predicting..."):
        pred_labels_and_probs, pred_time = predict(img)

    # Sort predictions
    top5 = sorted(
        pred_labels_and_probs.items(),
        key=lambda x: x[1],
        reverse=True
    )[:5]

    best_label, best_prob = top5[0]

    st.success(
        f"### 🍽️ Prediction: {best_label.replace('_',' ').title()}"
    )

    st.write(f"**Confidence:** {best_prob:.2%}")

    st.divider()

    st.subheader("Top 5 Predictions")

    for label, prob in top5:
        st.write(f"**{label.replace('_',' ').title()}**")
        st.progress(prob)
        st.write(f"{prob:.2%}")

    st.caption(f"Inference time: {pred_time:.4f} seconds")

else:
    st.info("Upload an image or choose the example above.")