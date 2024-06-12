import pandas as pd
import numpy as np
import ast
import pickle
import torch
from collections import Counter
from sentence_transformers import SentenceTransformer
import streamlit as st

class ModelBert:
    def __init__(self):
        # Read the CSV files
        self.history_df = pd.read_csv('./data/netflix_history_preprocessed.csv')
        self.titles_df = pd.read_csv('./data/netflix_titles_preprocessed.csv')
        self.watch_history = None
        self.combined_scores = None
        self.model = None
        self.metadata_embeddings = None
        self.metadata_similarity_scores = None
        self.descriptions_embeddings = None
        self.descriptions_similarity_scores = None
    
    def preprocess_text(self):
        # Convert string representation of list to actual list
        self.titles_df['director'] = self.titles_df['director'].apply(ast.literal_eval)
        self.titles_df['cast'] = self.titles_df['cast'].apply(ast.literal_eval)
        self.titles_df['country'] = self.titles_df['country'].apply(ast.literal_eval)
        self.titles_df['listed_in'] = self.titles_df['listed_in'].apply(ast.literal_eval)

        # Keep only the first occurrence of each title
        self.titles_df = self.titles_df.drop_duplicates(subset=['title'], keep='first').reset_index(drop=True)

        history_titles_set = set(self.history_df['Title'])
        titles_set = set(self.titles_df['title'])
        overlaps = history_titles_set.intersection(titles_set)
        self.history_df = self.history_df[self.history_df['Title'].isin(overlaps)]
        self.watch_history = self.history_df['Title'].to_list()

        # Flatten the list of actor names
        actor_names = [name for sublist in self.titles_df['cast'] for name in sublist]

        # Count the occurrences of each actor name
        name_counts = Counter(actor_names)
        
    
    def get_similarity_scores(self):
        self.titles_df['cast'] = self.titles_df['cast'].apply(self.keep_top_three_actors)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)
        descriptions = self.titles_df['description'].to_list()
        self.descriptions_embeddings = self.model.encode(descriptions, convert_to_tensor=True)
        self.descriptions_similarity_scores = torch.matmul(self.descriptions_embeddings, self.descriptions_embeddings.T).cpu().numpy()
        
    
    def generate_metadata(self):
        metadata = []

        for index, row in self.titles_df.iterrows():
            text = ''
            if row['director']:
                text += f"the director is {','.join(row['director'])}. "
            if row['cast']:
                text += f"the leading actors are {','.join(row['cast'])}. "
            if row['country']:
                text += f"the movie is from {','.join(row['country'])}. "
            if row['release_year']:
                text += f"the movie is released in {row['release_year']}. "
            if row['listed_in']:
                text += f"the movie falls within the genre of {','.join(row['listed_in'])}. "
            
            metadata.append(text)
        self.metadata_embeddings = self.model.encode(metadata, convert_to_tensor=True)
        self.metadata_similarity_scores = torch.matmul(self.metadata_embeddings, self.metadata_embeddings.T).cpu().numpy()
    
    def dump_pickle_files(self):
        pickle.dump(self.titles_df, open('./data/movie_list.pkl', 'wb'))
        pickle.dump(self.descriptions_similarity_scores + self.metadata_similarity_scores, open('./data/similarity_bert.pkl', 'wb'))
        pickle.dump(self.descriptions_embeddings.cpu().numpy(), open('./data/descriptions_embeddings.pkl', 'wb'))
        pickle.dump(self.metadata_embeddings.cpu().numpy(), open('./data/metadata_embeddings.pkl', 'wb'))
        
    def get_model(self):
        pass
        

    def keep_top_three_actors(self, actor_list):
        if len(actor_list) == 0:
            return []
        # Keep only the top k most frequent actors
        actor_list.sort(key=lambda x: self.name_counts[x], reverse=True)
        return actor_list[:3]

    
    # Evaluation of the recommendation

    def evaluate(self, similarity_scores, consider_history=False):
        target_ranks = []
        scores = np.zeros(similarity_scores.shape[0])
        
        for i in range(1, len(self.watch_history)):
            target_title = self.watch_history[i]
            target_row_index = self.titles_df.index[self.titles_df['title'] == target_title].tolist()[0]
            prev_title = self.watch_history[i - 1]
            prev_row_index = self.titles_df.index[self.titles_df['title'] == prev_title].tolist()[0]
        
            # Get recommendation based on the similarity
            if consider_history:
                scores = 1 / 2 * scores + 1 / 2 * similarity_scores[prev_row_index]
            else:
                scores = similarity_scores[prev_row_index]
            recommendation_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            target_rank = recommendation_indices.index(target_row_index)
            target_ranks.append(target_rank)
        
        print('Average rank:', np.mean(target_ranks))
        print('Successful recommendations:', np.sum(np.array(target_ranks) <= 5))

        
    
if __name__ == "__main__":
    ModelBert.get_model()