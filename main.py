import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D
from util import classify, set_background


def custom_objects():
    # Newer Keras saves a 'groups' arg that older DepthwiseConv2D doesn't accept
    class FixedDepthwiseConv2D(DepthwiseConv2D):
        def __init__(self, **kwargs):
            kwargs.pop('groups', None)
            super().__init__(**kwargs)
    return {'DepthwiseConv2D': FixedDepthwiseConv2D}


@st.cache_resource
def load_classifier():
    return load_model(
        './models/pneumonia_classifier.h5',
        custom_objects=custom_objects(),
        compile=False,
    )


@st.cache_data
def load_labels():
    with open('./models/labels.txt', 'r') as f:
        # Handles lines like "0 NORMAL" and names with spaces, and a missing trailing newline
        return [line.strip().split(' ', 1)[1] for line in f if line.strip()]


set_background('./bgs/bg5.png')

st.title('Pneumonia Classification')
st.header('Please upload a chest X-ray image')

file = st.file_uploader('Choose an image', type=['png', 'jpg', 'jpeg'])

model = load_classifier()
class_names = load_labels()

if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_container_width=True)

    class_name, conf_score = classify(image, model, class_names)

    st.write(f"## {class_name}")
    st.write(f"### Score: {conf_score * 100:.1f}%")
