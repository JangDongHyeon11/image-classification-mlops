import os
import base64
import requests
import streamlit as st
from PIL import Image
from io import BytesIO

PREDICT_ENDPOINT = os.getenv("PREDICT_ENDPOINT", "http://dl_service:4242/predict/")

st.title('Classify Images with ANY models you put (really)')

cached_img_file = None
cached_encoded_img = None

def reset():
    global cached_img_file
    global cached_encoded_img
    cached_img_file = None
    cached_encoded_img = None

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    cached_img_file = uploaded_file.read()
    cached_encoded_img = base64.b64encode(cached_img_file).decode('utf-8')

    st.image(Image.open(BytesIO(cached_img_file)), caption='Uploaded Image', use_column_width=True)

    if st.button('Make a prediction'):
        res = requests.post(PREDICT_ENDPOINT, files={'file':cached_img_file})
        res = res.json()
        pred_result = res['prediction']
        sorted_pred = list(reversed(sorted(pred_result.items(), key=lambda x: x[1])))

        st.write(f"Prediction: **{sorted_pred[0][0]}**")
        st.write("Top 3 Predictions:")
        for class_name, prob in sorted_pred[:3]:
            st.write(f"{class_name} : {prob*100:.2f}%")

        # Display the original and GradCAM overlaid image
        raw_hm_img = base64.b64decode(res['overlaid_img'])
        st.image(Image.open(BytesIO(raw_hm_img)), caption='GradCAM Heatmap', use_column_width=True)

        # Add a slider for opacity
        opacity = st.slider('GradCAM Opacity', 0, 100, 50) / 100
        original_img = Image.open(BytesIO(cached_img_file))
        heatmap_img = Image.open(BytesIO(raw_hm_img)).convert("RGBA")
        heatmap_img = heatmap_img.resize(original_img.size)

        blended_img = Image.blend(original_img.convert("RGBA"), heatmap_img, opacity)
        st.image(blended_img, caption='Blended Image', use_column_width=True)

if st.button('Reset'):
    reset()
    st.experimental_rerun()
