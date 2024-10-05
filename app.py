import streamlit as st
import numpy as np
import pandas as pd
import pickle
import re
import sklearn
import string
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer
image_path = 'innomatics-logo-img.png'
st.image(image_path)
file_path = r'C:\Users\Administrator\MachineLearning\fake_news\Fake_news_algorithm1.pkl'
#new
with open(file_path , 'rb') as f:
    lr = pickle.load(f)

#new
model= pickle.load(open('countVectorizer.pkl','rb'))


def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]','',text)
    text = re.sub("\\W"," ",text)
    text = re.sub('https?://\S+|www\.\S+','',text)
    text = re.sub('<.*?>+',b'',text)
    text = re.sub('[%s]' % re.escape(string.punctuation),'',text)
    text = re.sub('\w*\d\w*','',text)
    return text



def output_lable(n):
        if n==0:
            return "It is a Fake"
        elif n==1:
            return "It is Not a Fake"

def manual_testing(news):
        testing_news = {"text":[news]}
        new_def_test = pd.DataFrame(testing_news)
        new_def_test['text'] = new_def_test["text"].apply(wordopt)
        new_x_test = new_def_test["text"]
        new_xv_test = model.transform(new_x_test)
        pred_rfc = lr.predict(new_xv_test)
        result1=output_lable(pred_rfc[0])

        return result1


def main():
    st.title("Fake News Classifier")

    news_input = st.text_area("Enter the news text:")

    if st.button("Classify"):
        if news_input:
            result = manual_testing(news_input)
            st.write(result)
            if result== "It is a Fake":
                 st.image('Fake_img.jpg')
            elif result == "It is Not a Fake":
                 st.image("nofake.jpg")
        else:
            st.warning("Please enter the news text.")

if __name__ == "__main__":
    main()



    


