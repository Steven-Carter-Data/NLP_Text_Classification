
##############
# Author: Steven Carter
# NLP -- Text Classification #

# Reference: https://www.kaggle.com/faressayah/20-news-groups-classification-prediction-cnns#%F0%9F%A4%96-Machine-Learning
# Reference: https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html

# Import Required Packages
# Dataset included in sklearn datasets

import pandas
import sklearn
from sklearn.datasets import fetch_20newsgroups
import streamlit as st

st.set_page_config(
     page_title='NLP Text Classification',
     layout="wide"
    )

# Create Page Title

st.title("NLP Text Classification")
st.sidebar.title("Machine Learning Classifiers")

st.write('This dataset is a collection newsgroup documents.')
st.write('The 20 newsgroups collection has become a popular data set for experiments in text applications of machine learning techniques, such as text classification and text clustering.')

# Create Categories

categories = ['alt.atheism', 'soc.religion.christian',
               'comp.graphics', 'sci.med']

# Load the files matching those categories

twenty_train = fetch_20newsgroups(subset='train',
     categories=categories, shuffle=True, random_state=42)


def homepage():
    with st.beta_expander("NLP Objectives"):
    # Add the objectives for the app
        st.write('1. Document Classification')
        st.write('2. Text Pre-Processing')
        st.write('3. Feature Extraction')
        st.write('4. Vocabulary Creation')
        