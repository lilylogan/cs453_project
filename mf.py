#!/usr/bin/env python
# coding: utf-8

# Matrix Factorization

"""This python script exisits so that other ipynb files can access the information in mf.ipynb"""

# In[4]:


import pandas as pd
import numpy as np
import random
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split



# In[ ]:


def mf_recommender(user_songs, num_pred=5, data_path="cleaned_data/csr_df.csv", num_components=200, num_iter=10):
    """
    Recommends songs using Matrix Factorization (SVD).
    
    Parameters:
        user_songs (list): List of song names liked by the user.
        data_path (str): Path to the song interaction dataset.
        num_components (int): Number of components for SVD.
        num_iter (int): Number of iterations for SVD.

    Returns:
        list: Top recommended songs.
    """
    # Load dataset
    df_songs = pd.read_csv(data_path)

    # Convert to NumPy array
    songs = df_songs.to_numpy()

    # Initialize new user row (all 0s)
    new_user_row = np.zeros(songs.shape[1])
    
    # Mark songs the user likes
    for song in user_songs:
        if song in df_songs.columns:
            new_user_row[df_songs.columns.get_loc(song)] = 1

    list_of_songs = []

    for _ in range(5):  # Run multiple times for better recommendations
        rand = int(random.random() * 100)  # Random seed

        # Prepare training data
        x, y = songs.reshape((songs.shape[0], songs.shape[1])), range(songs.shape[0])
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.24, random_state=42)

        # Apply SVD for dimensionality reduction
        svd = TruncatedSVD(n_components=num_components, n_iter=num_iter, random_state=rand)
        svd.fit(x)

        # Transform training data
        train_mat = svd.transform(x_train)

        # Reconstruct an approximate version of original data
        approx_matrix = np.dot(train_mat, svd.components_)

        # Project new user onto reduced feature space
        new_user_mat = np.dot(new_user_row, svd.components_.T)

        # Reconstruct predictions in original space
        new_predictions = np.dot(new_user_mat, svd.components_)

        # Sort indices of predicted ratings in descending order
        recommendations = np.argsort(-new_predictions)[:num_pred]

        # Convert indices to song names
        recommended_songs = [df_songs.columns[i] for i in recommendations]
        list_of_songs.extend(recommended_songs)

    # Sort songs by frequency of occurrence in recommendations
    sorted_songs = sorted(set(list_of_songs), key=lambda song: list_of_songs.count(song))
    
    # Select the top num_pred songs (returns a list of recommended track keys)
    return sorted_songs[-num_pred:]


# In[ ]:


# from matplotlib_venn import venn3

# # Exploring how similar the outputs from the three methods are (how many recommended songs overlap)

# recommended_songs_euc = [song[0] for song in closest_songs_euc]

# # Create a Venn diagram
# venn3([set(top_songs), set(recommended_songs_euc), set(recommended_songs_cos)], set_labels=('SVD', 'Euclidean', 'Cosine'))
# plt.show()

# print([set(top_songs)])
# print([set(recommended_songs_euc)])
# print([set(recommended_songs_cos)])

