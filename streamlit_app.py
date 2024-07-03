import streamlit as st
import pandas as pd
import numpy as np
from model.model import query

st.title('🙀 I will help you find cute cat images')

title = st.text_input("Cat Breed", "Yamaha")

if st.button("Find cat"):
    st.write(f"Finding image for {title}...")
    lst_image = query(title)
    for image in lst_image:
        st.image(image, caption=title)