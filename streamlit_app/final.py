import streamlit as st
import numpy as np
import pandas as pd
import time
import cv2
from PIL import Image
import create_model as cm


st.title("Chest X-ray Report Generator")
st.text("by Ashish Thomas Chempolil")
model_tokenizer = cm.create_model()
start = time.process_time()
image_1 = st.file_uploader("X-ray 1",type=['png','jpg','jpeg'])
image_2 = st.file_uploader("X-ray 2 (optional)",type=['png','jpg','jpeg'])

if st.button("Predict"):
    if (image_1 is not None):
        start = time.process_time()  
        image_1 = Image.open(image_1).convert("RGB") #converting to 3 channels
        
        image_1 = np.array(image_1)/255
        if image_2 is None:
            image_2 = image_1
        else:
            image_2 = Image.open(image_2).convert("RGB") #converting to 3 channels
            image_2 = np.array(image_2)/255
        st.image([image_1,image_2],width=256)
        caption = cm.function1([image_1],[image_2],model_tokenizer)
        st.markdown(" ### **Impression:**")
        impression = st.empty()
        impression.write(caption[0])
        time_taken = "Time Taken for prediction: %i seconds"%(time.process_time()-start)
        st.write(time_taken)
        del image_1,image_2
    else:
        st.markdown("## Upload an Image")



