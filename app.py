from flask import Flask,render_template,request
from transformers import AutoTokenizer,TFAutoModelForSequenceClassification
import tensorflow as tf
import warnings
from nltk.corpus import stopwords
from deep_translator import GoogleTranslator
from nltk.stem import WordNetLemmatizer
import pandas as pd
import streamlit as st
import time

def fetch_data():
    saved_directory = 'model'
    tokenizer = AutoTokenizer.from_pretrained(saved_directory)
    model = TFAutoModelForSequenceClassification.from_pretrained(saved_directory)

    return tokenizer,model

def preprocess(text):
    stop_words = set(stopwords.words('english'))
    stop_words.remove('who')

     #Translate
    text = GoogleTranslator(source='auto', target='en').translate(text)

    #Lowercase the text
    text = text.lower()

    #Split the text into words
    words = text.split()

    # Perform stemming
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

def predict(input_text):
    labels = ['Full Name', 'Age', 'Birthplace', 'Programming Language','General information','Coding Project','Frameworks','University']
    input_text = preprocess(input_text)
    new_X = dict(tokenizer(input_text, padding=True, truncation=True, max_length=50, return_tensors='tf'))
    predictions = model.predict(new_X)
    predictions = labels[tf.argmax(predictions['logits'][0].tolist())]

    text = dfLabel.text[dfLabel['label'] == predictions].reset_index(drop=True)[0]
    return text

dfLabel = pd.read_csv('Data/labelData.csv')

st.markdown("<h1 style='text-align: center;'>Chatty</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Ask Any Question</h1>", unsafe_allow_html=True)



tokenizer,model = fetch_data()



text = ''

st.text("")
st.text("")
st.text("")


with st.form("my_form"):
    input_text = st.text_input('Example: What is your name')
    submitted = st.form_submit_button("Submit")
    

my_bar = st.progress(0)

if submitted:
        for percent_complete in range(100):
            time.sleep(0.1)
            my_bar.progress(percent_complete + 1)

        text = input_text
        text = predict(input_text)

st.text("")
st.text("Answer")

st.info(text)
