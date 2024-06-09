import streamlit as st
from bs4 import BeautifulSoup
import numpy as np
import pickle
from transformers import BertTokenizer, BertModel
from sklearn.linear_model import LogisticRegression
import re


def feature_extract(file):
    text_data = []

    # Decode bytes to UTF-8 string
    html_content = file.read().decode('utf-8')

    # Parsing the HTML content using BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser') 
    
    # finding the table elements
    tables = soup.find_all('table')
    for table in tables:
        rows = table.find_all('tr')     #table rows

        for row in rows:
            cells = row.find_all(['th','td'])   #table header and table data
            cell_text = ' '.join(cell.get_text(strip=True) for cell in cells)

            #removing all the characters other than alphabets and space
            text = re.sub(r'[^a-zA-Z\s]',' ', cell_text) 
            text_data.append(text)

    feature = ' '.join(text_data)
    feature = (re.sub(r'\s+',' ',feature)).lower() #substituting all the extra spaces with one white space
    return feature

#using pretrained bert tokenizer and model

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model =BertModel.from_pretrained('bert-base-uncased')

def embedding_text(text):
    #pytorch tensor of vector size 512
    inputs = tokenizer(text, return_tensors = 'pt',  truncation=True, padding=True, max_length=512)
    outputs = model(**inputs) #unpacks the inputs dictionary into keyword arguments
    
    # last hidden states are aggregated into single vector and detach from computational graph
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()


#loading the trained logistic regression model
model_path = '/Users/velmurugan/Desktop/@/python_works/html finance classification model/Classifier.pkl'

with open(model_path,'rb') as f:
    classifier = pickle.load(f)

labels = {1:'Cash Flow', 2:'Income Statement', 0:'Balance Sheets', 4:'Others', 3:'Notes'}

#setting the title and background for the ui

st.set_page_config(page_title="Finance File Classification")
st.title("Finance file Classification")

st.image('/Users/velmurugan/Desktop/@/python_works/html finance classification model/dollar.jpeg')

def setting_bg():
    st.markdown(f""" <style>.stApp {{
                background: linear-gradient(to bottom, #483D8B, #ADD8E6);
            }}
           </style>""",
        unsafe_allow_html=True)

setting_bg()

#get html file from user
uploaded_file = st.file_uploader("Upload an HTML file",type='html')

if uploaded_file is not None:
    st.success("File uploaded successfully!")

    predict_button = st.button('Predict Category')

    if predict_button:
        text = feature_extract(uploaded_file)  #applying feature extraction on uploaded file
        text_embed = embedding_text(text)      #word embedding on extracted text
        X = np.vstack([text_embed])            #formating into 2d matrix 
        y_pred = classifier.predict(X)[0]      #predict the class
        prediction = labels[y_pred]            #map the numerical value to class name

        st.write(f"The classified label is: {prediction}")
    

