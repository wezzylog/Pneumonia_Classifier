import streamlit as st
from keras.models import load_model
from PIL import Image
from tensorflow.keras.layers import DepthwiseConv2D
from util import classify, set_background
def custom_objects():
    return {"DepthwiseConv2D": lambda **kwargs: DepthwiseConv2D(**{k: v for k, v in kwargs.items() if k != 'groups'})}
set_background('./bgs/bg5.png')

#set title
st.title('Pneumonia Classification')

#set header
st.header('Please Upload a chest X-ray image')

#Upload file
file = st.file_uploader('Choose an image', type=['png', 'jpg', 'jpeg'])

#Load classifier
model = load_model('./models/pneumonia_classifier.h5' ,custom_objects=custom_objects())

#Load Class names
with open('./model/labels.txt', 'r') as f:
    class_names = [a[:-1].split(' ')[1] for a in f.readlines()]
    f.close()
#Display Image
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)

    #classify images
    class_name,conf_score = classify(image , model, class_names)

    #write classification
    st.write("## {}".format(class_name))
    st.write("### score: {}%".format(int(conf_score * 1000) / 10))
