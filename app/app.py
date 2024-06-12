# Import libraries
import pickle
import streamlit as st
import requests
import numpy as np
from PIL import Image
from os import path
import os
import time

from model_pickle_creator import ModelCreator


# Set favicon, page title and layout
NETFLIX_SYMBOL = Image.open("./data/images/Netflix_Symbol_RGB.png")
st.set_page_config(page_title="Recommender System", page_icon=NETFLIX_SYMBOL, layout="wide")


# Disable button if results already shown
if 'run_button' in st.session_state and st.session_state.run_button == True:
    st.session_state.running = True
else:
    st.session_state.running = False

# Show funny video while waiting, will probably be removed in endproduct
if "show_vid" not in st.session_state:
    if path.exists(f"./data/similarity_bert.pkl") and path.exists(f"./data/similarity_tfidf.pkl"):
        st.session_state.show_vid = False
    else: st.session_state.show_vid = True    

streamlit_style = """
			<style>
            /* Hide the scrollbar but keep scrolling functionality */
            ::-webkit-scrollbar {
                width: 0px;
                background: transparent; /* Make the scrollbar transparent */
            }
   
			@import url('https://fonts.googleapis.com/css2?family=Montserrat&display=swap');

			html, body, [class*="css"]  {
			font-family: 'Monserrat', sans-serif;
			}
   
            .info {
                font-size:24px !important;
            }
            .movie-title {
                font-size:30px !important;
                font-weight: bold;
            }
			</style>
			"""
st.markdown(streamlit_style, unsafe_allow_html=True)

# load tfidf model
def load_tfidf():
    if path.exists(f"./data/similarity_tfidf.pkl"):
        return pickle.load(open(f'./data/similarity_tfidf.pkl', 'rb'))
    else:
        ModelCreator().join_pkl("./data/model/", f"./data/similarity_tfidf.pkl", read_size=50000000, PARTS=[f"tfidf_model{i}" for i in range(1,28)])
        
# load bert model
def load_bert():
    if path.exists(f"./data/similarity_bert.pkl"):
        return pickle.load(open(f'./data/similarity_bert.pkl', 'rb'))
    else:
        ModelCreator().join_pkl("./data/model/", f"./data/similarity_bert.pkl", read_size=50000000, PARTS=[f"bert_model{i}" for i in range(1,28)])

# load movie list
def load_movies():
    return pickle.load(open(r'./data/movie_list.pkl', 'rb'))   

# show netflix logo and team number in sidebar
NETFLIX_LOGO = Image.open("./data/images/Netflix_Logo_RGB.png")
st.sidebar.image(NETFLIX_LOGO)
st.sidebar.title('Team 5')
if st.session_state.show_vid and False:
    st.sidebar.write("You can watch this video while the models are training")
    st.sidebar.video("https://www.youtube.com/watch?v=UcRtFYAz2Yo")

# Initialize models if needed
load_tfidf()
load_bert()

# cache models to prevent reloading on every run
@st.cache_data(show_spinner=True)
def cache_tfidf():
    return load_tfidf()

# cache models to prevent reloading on every run
@st.cache_data(show_spinner=True)
def cache_bert():
    return load_bert()

# cache movie list to prevent reloading on every run
@st.cache_data(show_spinner=True)
def cache_movies():
    return load_movies()

# Load data using st.cache_data to prevent reloading on every run
def load_data():
    return {
        'movies': cache_movies(),
        'similarity_tfidf': cache_tfidf(),
        'similarity_bert': cache_bert(),       
    }

# Load data
data = load_data()
movies = data['movies']

# Give option for watch history to get recommendations based on mutliple movies
if 'watched_movies' not in st.session_state:
    st.session_state.watched_movies = []
if 'summed_matrix_histories' not in st.session_state:
    st.session_state.summed_matrix_histories = np.zeros(movies.shape[0])

# recommend movies based on user input
def recommend(movie, use_history):
    if embed_type == 'TF-IDF':
        similarity = data['similarity_tfidf']
    else:
        similarity = data['similarity_bert']

    index = movies[movies['title'] == movie].index[0]

    if use_history:
        st.session_state.watched_movies.append(index)
        st.session_state.summed_matrix_histories = st.session_state.summed_matrix_histories + similarity[index]
        final_matrix = st.session_state.summed_matrix_histories
    else:
        final_matrix = similarity[index]

    distances = sorted(list(enumerate(final_matrix)), reverse=True, key=lambda x: x[1])
    recommended_movie_ids = []

    count = 0
    for index, item in distances[1:]:
        if index not in st.session_state.watched_movies:
            recommended_movie_ids.append(index)
            count = count + 1
            if count == 5:
                break
            
    return recommended_movie_ids
        
# display recommendation page
def display_selection_page():
    st.title('Movie Recommender System - Data Exploration')

    global embed_type
    embed_type = st.sidebar.selectbox(
        'Embedding type:',
        ['TF-IDF', 'BERT']
    )
    with st.container():
        col1,col2 = st.columns([2,1])
        with col1:
            with st.container(border=True):
                use_history = st.checkbox("Use multiple histories (save watched movies and recommend considering all movies watched)")

                movie_list = movies['title'].values
                selected_movie = st.selectbox(
                    "Type or select a movie from the dropdown",
                    movie_list 
                )
                generate_recommendation_btn = st.button('Show Recommendation', disabled=st.session_state.running, key='run_button')
            if selected_movie in movies["title"].values:
                movie = movies[movies["title"] == selected_movie]
                director = ', '.join(movie.iloc[0]['director']) if movie.iloc[0]['director'] else "-"
                cast = ', '.join(movie.iloc[0]['cast']) if movie.iloc[0]['cast'] else "-"
                genre = ', '.join(movie.iloc[0]['listed_in']) if movie.iloc[0]['listed_in'] else "-"
                country = ', '.join(movie.iloc[0]['country']) if movie.iloc[0]['country'] else "-"
                release = movie.iloc[0]['release_year'] if movie.iloc[0]['release_year'] else "-"
                platform = movie.iloc[0]['platform'] if movie.iloc[0]['platform'] else "-"
                with st.container(border=True):
                    st.markdown(f'<p class="info">Chosen Movie or Show :</p><br>', unsafe_allow_html=True)
                    st.header(capitalize_sentence(selected_movie))
                    st.markdown(f'<p class="info">Director: {capitalize_sentence(director)}</p>', unsafe_allow_html=True)  
                    st.markdown(f'<p class="info">Country: {capitalize_sentence(country)}</p>', unsafe_allow_html=True)    
                    st.markdown(f'<p class="info">Genre: {capitalize_sentence(genre)}</p>', unsafe_allow_html=True)
                    st.markdown(f'<p class="info">Platform: {capitalize_sentence(platform)}</p>', unsafe_allow_html=True)
            with st.container(border=True):
                if path.exists(f"./data/similarity_bert.pkl") and path.exists(f"./data/similarity_tfidf.pkl"):
                    st.title("🔻 Recommendation will be shown below 🔻")
                    st.session_state.running = True
                    st.session_state.show_vid = False
                else: 
                    st.warning("Models nopt trained yet. Please wait...") 
                    st.session_state.running = False   
        with col2:
            if selected_movie in movies["title"].values:
                st.image(get_image_from_tmdb(selected_movie), use_column_width=True)
        
    if generate_recommendation_btn:
        recommended_movie_ids = recommend(selected_movie, use_history)
        display_recommendations(recommended_movie_ids)
    
    if st.session_state.watched_movies:
        display_watched_movies()

# Display watched movies and reset button
def display_watched_movies():
    st.sidebar.write("Watched movies:")
    for i in st.session_state.watched_movies:
        st.sidebar.write(movies['title'][i])

    if st.sidebar.button("Reset"):
        st.session_state.watched_movies = []
        st.session_state.summed_matrix_histories = np.zeros(movies.shape[0])

# Display movie recommendations
def display_recommendations(recommended_movie_ids):
    
    titles = st.columns(5)
    columns = st.columns(5)
    info = st.columns(5)

    for i, index in enumerate(recommended_movie_ids):
        movie = movies.iloc[index]
        if not movie.empty:
            title = movie['title']
            director = ', '.join(movie['director']) if movie['director'] else "-"
            cast = ', '.join(movie['cast']) if movie['cast'] else "-"
            genre = ', '.join(movie['listed_in']) if movie['listed_in'] else "-"
            country = ', '.join(movie['country']) if movie['country'] else "-"
            release = movie['release_year'] if movie['release_year'] else "-"
            platform = movie["platform"] if movie["platform"] else "-"
            with titles[i]: st.title(capitalize_sentence(title))
            # Display each movie in a separate column
            with columns[i]:
                try:
                    st.image(get_image_from_tmdb(title), use_column_width=True)
                except Exception as e:
                    print("Failed to load image")
                    print(f"Error: {e}")    
                    st.image('./data/images/empty.jpg', use_column_width=True)
            with info[i]:      
                st.markdown(f'<p class="info">Director: {capitalize_sentence(director)}</p>', unsafe_allow_html=True)  
                st.markdown(f'<p class="info">Country: {capitalize_sentence(country)}</p>', unsafe_allow_html=True)    
                st.markdown(f'<p class="info">Genre: {capitalize_sentence(genre)}</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="info">Platform: {capitalize_sentence(platform)}</p>', unsafe_allow_html=True)
        else:
            st.write(f"Movie '{index}' not found in the dataset.")

# get image from TMDB API
def get_image_from_tmdb(movie_name):
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiIyNjVmNWJlMTg0OWIwYTRhZWMyZTE2ZWVkOWE5OWI0YiIsInN1YiI6IjY2NTViYjUwMGMzZDA0NDAyMGEzNmE0NyIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.tRC-sEvBZFYg7c-Jqj5isV0o9wqRSbEbDQTbunHUS9Y"
    }
    url = f"https://api.themoviedb.org/3/search/movie?query={movie_name}"
    response = requests.get(url, headers=headers)
    #print(response.json())
    try:
        return f'https://image.tmdb.org/t/p/w185{response.json()["results"][0]["poster_path"]}'
    except Exception as e:
        print("Error inside get_image_from_tmdb func: " + e)
        return None
        

# Capitalize the first letter of each word in a sentence
def capitalize_sentence(string):
    # Split the string into sentences
    sentences = string.split(' ')

    # Capitalize the first letter of each sentence
    capitalized_sentences = [sentence.capitalize() for sentence in sentences]

    # Join the capitalized sentences back into a single string
    return ' '.join(capitalized_sentences)

# Generate recommendation using GPT model and display embedding
def generate_recommendation(movie_prompt):    
        response = requests.post("http://localhost:5000/embed", json={"prompt": movie_prompt})
        recommended_movies = response.json()["recommended_movie_ids"]
        display_recommendations(recommended_movies)

# Main function to display selected page
def main():
    global embed_type
    display_selection_page()

# Run the app
if __name__ == "__main__":
    main()
