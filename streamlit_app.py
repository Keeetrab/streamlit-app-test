import altair as alt
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from data_processing import clean_data
from rfm import calculate_rfm

# Show the page title and description.
st.set_page_config(page_title="Retail Segmentation test", page_icon="🎬")
st.title("Retail Segmentation test")
st.write(
    """
    This app visualizes data from [The Movie Database (TMDB)](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata).
    It shows which movie genre performed best at the box office over the years. Just 
    click on the widgets below to explore!
    """
)


# Load the data from a CSV. We're caching this so it doesn't reload every time the app
# reruns (e.g. if the user interacts with the widgets).
@st.cache_data
def load_data():
    data = pd.read_csv("my_data/retail_data.csv")
    return data


data = load_data()

st.write("Data Preview")
st.write(data.head())
st.write(data.info())
st.write(data.describe())

#Preprocessing
data = clean_data(data)

st.write("Preprocessing")
st.write(data)


#RFM

rfm = calculate_rfm(data)
st.write("RFM")
st.write(rfm)

st.write('Clusters Visualization')

st.pyplot(plt.gcf())