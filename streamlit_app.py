import os
os.system('pip install streamlit')
os.system('pip install faiss-cpu')
os.system('pip install transformers')
os.system('pip install kaggle')
os.environ['KAGGLE_CONFIG_DIR'] = '/Users/apple/Desktop/cat_breed/cat_search/asset/'

import streamlit as st
from transformers import CLIPProcessor, CLIPModel
import faiss
from kaggle.api.kaggle_api_extended import KaggleApi

def download_file_from_kaggle(dataset, file_path, download_dir):
    api = KaggleApi()
    api.authenticate()
    
    os.makedirs(download_dir, exist_ok=True)
    api.dataset_download_file(dataset, file_path, path=download_dir)
    
    zip_path = os.path.join(download_dir, file_path + '.zip')
    if os.path.exists(zip_path):
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(download_dir)
        os.remove(zip_path)
    return os.path.join(download_dir, file_path)


index = faiss.read_index('/Users/apple/Desktop/cat_breed/cat_search/asset/cats.index')

with open('/Users/apple/Desktop/cat_breed/cat_search/asset/paths.txt', 'r') as f:
       image_paths = [line.strip().split('/cat-breeds')[1] for line in f.readlines()]



model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

st.title('🙀 Find Your Cat')
st.write('Type anything about cat, we will show you related cat image!')

txt = st.text_input("", "e.x. Rainbow cat")
if st.button("Search"):
    inputs = processor(text=[txt], return_tensors="pt", padding=True, truncation=True)
    text_features = model.get_text_features(**inputs)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    text_features = text_features.cpu().detach().numpy()
    _, indices = index.search(text_features, 1)
    paths = image_paths[indices[0]]
    st.write(paths)

    downloaded_path = download_file_from_kaggle('plukiot/cat-breeds', paths, 'downloaded_images')
    st.image(downloaded_path, caption='Here is your cat!')


