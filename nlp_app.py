
##############
# Author: Steven Carter
# NLP -- Reports & News Classification #

# Reference: https://www.kaggle.com/vbmokin/er-envprobl-nlp-bag-of-words-tf-idf-glove
# Reference: https://www.kaggle.com/faressayah/20-news-groups-classification-prediction-cnns#%F0%9F%A4%96-Machine-Learning
# Reference: https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html

# Import Required Packages
# Dataset included in sklearn datasets

from os import write
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import torch
# Transformers documentation: https://huggingface.co/transformers/
import transformers as ppb

import warnings
warnings.filterwarnings('ignore')

import streamlit as st

st.set_page_config(
     page_title='NLP: Reports & News Classification',
     layout="wide"
    )

# Create Page Title

st.title("NLP: Reports & News Classification")
st.sidebar.title("Machine Learning Classifiers")

# Load in the dataset

@st.cache(allow_output_mutation=True) # turns warning off 
# User-defined function to: 
#    Read in data, fill NA with 0, & convert the TYPES of column data 

def data_set():
    """Loads once then cached for subsequent runs"""
    df = pd.read_csv("C:/Users/612721/1_Projects/NLP_Text_Classification/NLP_Text_Classification/water_problem_nlp_en_for_Kaggle_100.csv"
    ,delimiter = ';', header=0)
    df = df.fillna(0)
    convert_dict = {'text' : str, 'env_problems' : int, 'pollution' : int, 
    'treatment' : int, 'climate' : int, 'biomonitoring' : int}
    df = df.astype(convert_dict)
    return df

# COPY THE DATASET so any changes are not applied to the original cached version

df = data_set().copy()

# Display the dataset

st.write('### Purpose: The goal for this application is to walk through a text classifer utilizing several NLP techniques.')
st.write('Check out the imported dataset (after minor cleaning):')
st.write(data_set())
st.write('**NOTE:** The "1" or "0" indicates whether the news includes the labels. 1 = YES | 0 = NO ')

# Focusing on 'env_problems ONLY

st.subheader('Environmental Problems')
st.write('In this app we only focusing on "Environmental Problems" only.') 
st.write('Here we can see the modified dataframe (env_problems renamed to target): ')
target_name = 'env_problems'
name_for_plot = 'Environmental Problems'
data = df[['text', target_name]]
data.columns = ['text', 'target']
st.write(data)

# Create train and test sets

st.write('The TASK for this dataset requires that the test set contain at least 40% of the data.')

train, test = train_test_split(data, 
    test_size = 0.4, # indicates 40% will be test data
    shuffle = True, # data will be shuffled prior to splitting
    random_state = 0) # controls the shuffling applied to the data before the split
train_base = train.copy()
train_target = train_base['target']
test_base = test.copy()
test_target = test_base['target']

st.write('There are {} rows and {} columns in **train**'.format(data.shape[0], data.shape[1]))
st.write('There are {} rows and {} columns in **test**'.format(test.shape[0], test.shape[1]))

# EDA

st.subheader('Exploratory Data Analysis')

st.write('''There are any number of places to start with EDA. For our purposes we are going to look at 
the number of examples of each class through visualizations. We can see below the number of articles that
involve environmental problems that those that do not''')

# Plot the "Included" & "Not Included"

# extracting the number of examples in each class

# included = data[data['target'] == 1].shape[0]
# not_included = data[data['target'] == 0].shape[0]

fig = px.bar(train)

st.plotly_chart(fig) # can't get the chart to show up


