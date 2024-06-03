import pandas as pd
import numpy as np
import nltk
import string
import ast
import pickle
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords')
def get_tfidf_model():
    # Read the CSV files
    history_df = pd.read_csv('./data/netflix_history_preprocessed.csv')
    titles_df = pd.read_csv('./data/netflix_titles_preprocessed.csv')
    weights_df = pd.read_csv('./data/weights.csv')

    # Convert string representation of list to actual list
    titles_df['director'] = titles_df['director'].apply(ast.literal_eval)
    titles_df['cast'] = titles_df['cast'].apply(ast.literal_eval)
    titles_df['country'] = titles_df['country'].apply(ast.literal_eval)
    titles_df['listed_in'] = titles_df['listed_in'].apply(ast.literal_eval)

    # Keep only the first occurrence of each title
    titles_df = titles_df.drop_duplicates(subset=['title'], keep='first').reset_index(drop=True)

    history_titles_set = set(history_df['Title'])
    titles_set = set(titles_df['title'])

    overlaps = history_titles_set.intersection(titles_set)
    en_history_df = history_df[history_df['Title'].isin(overlaps)]
    watch_history = en_history_df['Title'].to_list()

    def preprocess_text(text):
        # Tokenization
        tokens = nltk.tokenize.word_tokenize(text.lower())

        # Remove punctuation
        tokens = [token for token in tokens if token not in string.punctuation]

        # Remove stop words
        stop_words = set(nltk.corpus.stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]

        # Stemming
        stemmer = nltk.stem.PorterStemmer()
        tokens = [stemmer.stem(token) for token in tokens]

        return ' '.join(tokens)

    titles_df['description'] = titles_df['description'].apply(preprocess_text)

    def preprocess_name(name_list):
        # Remove spaces between each name
        return [name.replace(' ', '') for name in name_list]

    titles_df['director'] = titles_df['director'].apply(preprocess_name)
    titles_df['cast'] = titles_df['cast'].apply(preprocess_name)

    # Flatten the list of actor names
    actor_names = [name for sublist in titles_df['cast'] for name in sublist]

    # Count the occurrences of each actor name
    name_counts = Counter(actor_names)

    def keep_top_three_actors(actor_list):
        if len(actor_list) == 0:
            return []
        # Keep only the top k most frequent actors
        actor_list.sort(key=lambda x: name_counts[x], reverse=True)
        return actor_list[:3]

    titles_df['cast'] = titles_df['cast'].apply(keep_top_three_actors)

    # Calculate TF-IDF vectors for processed titles and descriptions
    tfidf_vectorizer = TfidfVectorizer()
    titles_tfidf = tfidf_vectorizer.fit_transform(titles_df['description'])

    # Calculate cosine similarity
    similarity_scores = cosine_similarity(titles_tfidf, titles_tfidf)

    # Function to check if two lists have overlapping elements
    def have_overlap(list1, list2):
        return bool(set(list1) & set(list2))

    def create_overlap_matrix(column_name):
        matrix_size = len(titles_df)
        overlap_matrix = np.zeros((matrix_size, matrix_size), dtype=int)

        column = titles_df[column_name].to_list()
        for i in range(matrix_size):
            for j in range(matrix_size):
                if have_overlap(column[i], column[j]):
                    overlap_matrix[i, j] = 1

        return overlap_matrix

    overlap_director = create_overlap_matrix('director')
    overlap_cast = create_overlap_matrix('cast')
    overlap_country = create_overlap_matrix('country')
    overlap_genre = create_overlap_matrix('listed_in')

    description_weight, director_weight, cast_weight, country_weight, genre_weight = [weights_df[col] for col in "description,director,cast,country,genre".split(",")]
    # Combine similarity scores, director overlap, cast overlap, country overlap, and genre overlap
    combined_scores = description_weight * similarity_scores + director_weight * overlap_director + cast_weight * overlap_cast + country_weight * overlap_country + genre_weight * overlap_genre
    combined_scores = np.array(combined_scores, dtype=np.float32)

    pickle.dump(combined_scores, open('./data/similarity_tfidf.pkl', 'wb'))
    return pickle.load(open('./data/similarity_tfidf.pkl', 'rb'))
    
if __name__ == '__main__':
    get_tfidf_model()