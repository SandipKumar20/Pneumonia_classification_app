import streamlit as st
from keras.models import load_model
from PIL import Image
from utils import classify, set_background

set_background('x-ray.png')

# set title
st.title('Pneumonia Classification')

# set header
st.header('Please upload a chest x-ray')

# upload a file
img_file = st.file_uploader('', type=['jpg', 'jpeg', 'png'])

# load classifier
SNet = load_model('keras_model.h5')

# load class names
with open('labels.txt', 'r') as file:
    class_names = [a[:-1].split(' ')[1] for a in file.readlines()]
    file.close()

#print(class_names)

# display image
if img_file is not None:
    image = Image.open(img_file).convert('RGB')
    st.image(image, use_column_width=True)
    # classify image
    class_name, conf_score = classify(image, SNet, class_names)

    # write classification
    st.write(f"{class_name}")
    st.write(f"score: {conf_score}")
