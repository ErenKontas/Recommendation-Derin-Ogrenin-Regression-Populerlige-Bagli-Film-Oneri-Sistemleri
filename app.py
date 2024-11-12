import pandas as pd
import streamlit as st
import joblib

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer

# Load data and update column names
df = pd.read_csv('train.csv')

# Select dependent and independent variables
x = df.drop(['Unnamed: 0','userID','rating'], axis=1)
y = df[['rating']]

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['year','vote','runtime']),
        ('cat', OneHotEncoder(), ['title','kind','genre','country','language','cast','director','composer','writer']) 
    ]
)

# Streamlit application
def film_pred(title,year,kind,genre,vote,country,language,cast,director,composer,writer,runtime):
    input_data = pd.DataFrame({
        'title': [title],
        'year': [year],
        'kind': [kind],
        'genre': [genre],
        'vote': [vote],
        'country': [country],
        'language': [language],
        'cast': [cast],
        'director': [director],
        'composer': [composer],
        'writer': [writer],
        'runtime': [runtime]
    })
    input_data_transformed = preprocessor.fit_transform(input_data)

    model = joblib.load('Film.pkl')

    prediction = model.predict(input_data_transformed)
    
    # Convert prediction to a scalar if it's an array
    return float(prediction[0])

st.title("Film Prediction Model")
st.write("Enter Input Data")
    
title = st.text_input('Title')
year = st.slider('Year', int(df['year'].min()), int(df['year'].max()))
kind = st.selectbox('Kind', ['Movie', 'TV Show', 'Documentary'])
genre = st.selectbox('Genre', df['genre'].unique().tolist())
vote = st.slider('Vote', 0, 10)
country = st.selectbox('Country', df['country'].unique().tolist())
language = st.selectbox('Language', df['language'].unique().tolist())
cast = st.text_input('Cast')
director = st.selectbox('Director', df['director'].unique().tolist())
composer = st.selectbox('Composer', df['composer'].unique().tolist())
writer = st.selectbox('Writer', df['writer'].unique().tolist())
runtime = st.slider('Runtime', float(df['runtime'].min()), float(df['runtime'].max()))
    
if st.button('Predict'):
    film = film_pred(title,year,kind,genre,vote,country,language,cast,director,composer,writer,runtime)
    st.write(f'The predicted film is: {film:.2f}')