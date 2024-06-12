import pandas as pd
import numpy as np
import nltk
import string
import ast
import pickle
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ModelTfidf:
    def __init__(self):
        # Read the CSV files
        self.history_df = pd.read_csv('./data/netflix_history_preprocessed.csv')
        self.titles_df = pd.read_csv('./data/netflix_titles_preprocessed.csv')
        self.combined_scores = None
        self.name_counts = None
    
    def download_nltk_data(self):
        nltk.download('punkt')
        nltk.download('stopwords')
    
    
    def preprocess_data(self):
        # Convert string representation of list to actual list
        self.titles_df['director'] = self.titles_df['director'].apply(ast.literal_eval)
        self.titles_df['cast'] = self.titles_df['cast'].apply(ast.literal_eval)
        self.titles_df['country'] = self.titles_df['country'].apply(ast.literal_eval)
        self.titles_df['listed_in'] = self.titles_df['listed_in'].apply(ast.literal_eval)

        # Keep only the first occurrence of each title
        self.titles_df = self.titles_df.drop_duplicates(subset=['title'], keep='first').reset_index(drop=True)

        history_titles_set = set(self.history_df['Title'])
        self.titles_set = set(self.titles_df['title'])

        overlaps = history_titles_set.intersection(self.titles_set)
        en_history_df = self.history_df[self.history_df['Title'].isin(overlaps)]
        watch_history = en_history_df['Title'].to_list()

        self.titles_df['description'] = self.titles_df['description'].apply(self.preprocess_text)
        
        self.titles_df['director'] = self.titles_df['director'].apply(self.preprocess_name)
        self.titles_df['cast'] = self.titles_df['cast'].apply(self.preprocess_name)

        # Flatten the list of actor names
        actor_names = [name for sublist in self.titles_df['cast'] for name in sublist]

        # Count the occurrences of each actor name
        self.name_counts = Counter(actor_names)
        
        self.titles_df['cast'] = self.titles_df['cast'].apply(self.keep_top_three_actors)
    
    
    def calc_cosine_similarity(self):
        # Calculate TF-IDF vectors for processed titles and descriptions
        tfidf_vectorizer = TfidfVectorizer()
        self.titles_tfidf = tfidf_vectorizer.fit_transform(self.titles_df['description'])

        # Calculate cosine similarity
        similarity_scores = cosine_similarity(self.titles_tfidf, self.titles_tfidf)
        
        overlap_director = self.create_overlap_matrix('director')
        overlap_cast = self.create_overlap_matrix('cast')
        overlap_country = self.create_overlap_matrix('country')
        overlap_genre = self.create_overlap_matrix('listed_in')
        
        self.combined_scores = 50 * similarity_scores + 1 * overlap_director + 2 * overlap_cast + 0.5 * overlap_country + 2 * overlap_genre
        self.combined_scores = np.array(self.combined_scores, dtype=np.float32)
        return self.combined_scores
    
    
    def dump_pickle_file(self):
        pickle.dump(self.combined_scores, open('./data/model/similarity_tfidf.pkl', 'wb'))
    
    def get_model(self):
        self.download_nltk_data()
        self.preprocess_data()
        self.calc_cosine_similarity()
        self.dump_pickle_file()
        
        
    def preprocess_text(self, text):
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

        

    def preprocess_name(self, name_list):
        # Remove spaces between each name
        return [name.replace(' ', '') for name in name_list]

        

    def keep_top_three_actors(self, actor_list):
        if len(actor_list) == 0:
            return []
        # Keep only the top k most frequent actors
        actor_list.sort(key=lambda x: self.name_counts[x], reverse=True)
        return actor_list[:3]

        

        # Function to check if two lists have overlapping elements
    def have_overlap(self, list1, list2):
        return bool(set(list1) & set(list2))

    def create_overlap_matrix(self, column_name):
        matrix_size = len(self.titles_df)
        overlap_matrix = np.zeros((matrix_size, matrix_size), dtype=int)

        column = self.titles_df[column_name].to_list()
        for i in range(matrix_size):
            for j in range(matrix_size):
                if self.have_overlap(column[i], column[j]):
                    overlap_matrix[i, j] = 1

        return overlap_matrix

        # Evaluation of the recommendation
    def evaluate(self, x1, x2, x3, x4, x5, consider_history=False):
        target_ranks = []
        combined_scores = x1 * self.similarity_scores + x2 * self.overlap_director + x3 * self.overlap_cast + x4 * self.overlap_country + x5 * self.overlap_genre
        scores = np.zeros(combined_scores.shape[0])

        for i in range(1, len(self.watch_history)):
            target_title = self.watch_history[i]
            target_row_index = self.titles_df.index[self.titles_df['title'] == target_title].tolist()[0]
            prev_title = self.watch_history[i - 1]
            prev_row_index = self.titles_df.index[self.titles_df['title'] == prev_title].tolist()[0]

            # Get recommendation based on the similarity
            if consider_history:
                scores += combined_scores[prev_row_index]
            else:
                scores = combined_scores[prev_row_index]
            recommendation_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            target_rank = recommendation_indices.index(target_row_index)
            target_ranks.append(target_rank)

        print('Average rank:', np.mean(target_ranks))
        print('Successful recommendations:', np.sum(np.array(target_ranks) <= 5))
            

        

if __name__ == "__main__":
    ModelTfidf().get_model()    