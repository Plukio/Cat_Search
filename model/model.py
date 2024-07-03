import os
import torch
import faiss
import pandas as pd
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from oauth2client.client import GoogleCredentials


# Authenticate and create the PyDrive client
def authenticate_drive():
    gauth = GoogleAuth()
    gauth.LoadClientConfigFile("client_secrets.json")
    gauth.LoadCredentialsFile("credentials.json")
    if gauth.credentials is None:
        gauth.LocalWebserverAuth()
    elif gauth.access_token_expired:
        gauth.Refresh()
    else:
        gauth.Authorize()
    gauth.SaveCredentialsFile("credentials.json")
    drive = GoogleDrive(gauth)
    return drive

drive = authenticate_drive()

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


index = faiss.read_index('/cat_search/asset/cats.index')
with open('/cat_search/asset/paths.txt', 'r') as f:
       image_paths = [line.strip().split('/cat-breeds')[1] for line in f.readlines()]
im_id = pd.read_csv('/cat_search/asset/file_ids.csv')


device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def query(text_input):
    text_tokens = processor(text=text_input, return_tensors="pt", padding=True).input_ids.to(device)

    with torch.no_grad():
        text_features = model.get_text_features(text_tokens).cpu().numpy()

    lst_img = []
    _, indices = index.search(text_features, 1)
    path = image_paths[indices[0]].split('/')[-1]

    for i in range(1):
        file_row = im_id[im_id['Title'] == path]
        file_id = file_row['id'].values[0]
        destination_path = '/cat_search/asset/images' + path
        download_file_from_drive(file_id, destination_path)
        img = Image.open(destination_path)
        lst_img.append(img)
    
    return lst_img

def download_file_from_drive(file_id, destination):
    file = drive.CreateFile({'id': file_id})
    file.GetContentFile(destination)