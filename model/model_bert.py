import pandas as pd
import numpy as np
import os
import ast
import pickle
import torch
from collections import Counter
from sentence_transformers import SentenceTransformer

def get_bert_model():
    # Read the CSV files
    history_df = pd.read_csv('./data/netflix_history_preprocessed.csv')
    titles_df = pd.read_csv('./data/netflix_titles_preprocessed.csv')

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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(device)
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)
    descriptions = titles_df['description'].to_list()
    descriptions_embeddings = model.encode(descriptions, convert_to_tensor=True)
    descriptions_similarity_scores = torch.matmul(descriptions_embeddings, descriptions_embeddings.T).cpu().numpy()

    metadata = []

    for index, row in titles_df.iterrows():
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
        
    metadata_embeddings = model.encode(metadata, convert_to_tensor=True)
    metadata_similarity_scores = torch.matmul(metadata_embeddings, metadata_embeddings.T).cpu().numpy()

    pickle.dump(titles_df, open('./data/movie_list.pkl', 'wb'))
    pickle.dump(descriptions_similarity_scores + metadata_similarity_scores, open('./data/similarity_bert.pkl', 'wb'))
    pickle.dump(descriptions_embeddings.cpu().numpy(), open('./data/descriptions_embeddings.pkl', 'wb'))
    pickle.dump(metadata_embeddings.cpu().numpy(), open('./data/metadata_embeddings.pkl', 'wb'))
    print(os.system("ls"))
    return pickle.load(open('./data/similarity_bert.pkl', 'rb'))
if __name__ == '__main__':
    get_bert_model()