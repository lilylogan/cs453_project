#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from scipy.sparse import csr_matrix


# #### Load dataset

# In[3]:


# def train_or_test(test):
#     if test:
#         songs_df = pd.read_csv("subset_data/test_songs.csv")
#     else:
#         songs_df = pd.read_csv("subset_data/train_songs.csv")
    
#     return songs_df

# songs_df = pd.read_csv("subset_data/train_songs.csv")
songs_df = pd.read_csv("cleaned_data/songs_cleaned.csv")


# In[4]:


X = songs_df.sample(frac=1, random_state=42).reset_index(drop=True)


# #### Determine which attributes are numerical and which are categorical (will be used for standardizing)

# In[5]:


# Get numerical columns
numerical_cols = songs_df.select_dtypes(include=['number']).columns.tolist()


# In[6]:


categorical_cols = songs_df.select_dtypes(exclude=['number']).columns.tolist()


# #### Using a ColumnTransformer, create a preprocessor that which standardize both the categorical and numerical attributes

# In[7]:


preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=True), categorical_cols),  
        ('num', StandardScaler(), numerical_cols)      
    ], sparse_threshold=0.3)  # Keeps output sparse when >30% sparse


# In[8]:


# fit the knn
X_preprocessed = preprocessor.fit_transform(X)


# In[9]:


# Get the full list of feature names after the transformation
full_feature_names = preprocessor.get_feature_names_out()

# Find the indices of numerical features (those starting with 'num')
num_indices = [i for i, name in enumerate(full_feature_names) if name.startswith('num')]

# Find the indices of categorical features (those starting with 'cat')
cat_indices = [i for i, name in enumerate(full_feature_names) if name.startswith('cat')]


# #### 1. KNN

# In[10]:


knn = NearestNeighbors(n_neighbors=1, metric='euclidean')


# In[11]:


knn.fit(X_preprocessed)


# In[ ]:


import numpy as np

def knn_recommender(playlist_keys, r=5):
    """Finds r nearest songs based on the aggregated features of a playlist."""

    # Ensure input is a list
    if not isinstance(playlist_keys, list):
        playlist_keys = [playlist_keys]

    # Check if all playlist songs exist in the dataset
    missing_keys = [key for key in playlist_keys if key not in songs_df["track_key"].values]
    if missing_keys:
        return f"These keys are not in the database: {missing_keys}"

    # Extract and preprocess features for all songs in the playlist
    playlist_songs = songs_df[songs_df["track_key"].isin(playlist_keys)]
    processed_songs = preprocessor.transform(playlist_songs)

    # Aggregate the playlist features (e.g., by averaging)
    playlist_vector = csr_matrix(np.mean(processed_songs, axis=0).reshape(1, -1))

    # Get KNN recommendations
    distances, indices = knn.kneighbors(playlist_vector, n_neighbors=r*2)

    recommendations = []

    print("\nRecommended Songs:")
    recs = 0
    for j in range(len(distances[0])):
        song_idx = indices[0][j]
        song_row = X_preprocessed[song_idx]  # Get the transformed features

        # 🔹 **Convert sparse row to dense**
        song_dense = song_row.toarray()[0]

        # 🔹 **Manually Reverse Preprocessing**
        # Extract transformers from ColumnTransformer
        onehot_encoder = preprocessor.named_transformers_['cat']
        scaler = preprocessor.named_transformers_['num']


        # Apply inverse transforms
        original_categorical = onehot_encoder.inverse_transform(song_dense[cat_indices].reshape(1, -1))
        original_numerical = scaler.inverse_transform(song_dense[num_indices].reshape(1, -1))

        # Combine back to original format
        original_features = np.concatenate((original_categorical, original_numerical), axis=1)

        # 🔹 **Find Original Song Info**
        original_song = songs_df[songs_df["track_key"] == original_features[0][6]]  # Retrieve song details


        if original_song["track_key"].iloc[0] not in playlist_keys:

            # print(f"  Distance: {distances[0][j]}")
            print(original_song["track_name"].iloc[0])

            # recommendations.append(original_song)
            recommendations.append(original_song["track_key"].iloc[0])

            recs+=1

            if recs == r:
                break

    return recommendations


# In[19]:


test_1 = knn_recommender(["samurai - tiesto remix _ r3hab"], 3)


# In[14]:


test_2 = knn_recommender(["clarity _ zedd"], 3)


# In[15]:


test_3 = knn_recommender(["heroes (we could be) _ alesso"], 3)


# In[16]:


test_4 = knn_recommender(["clarity _ zedd", "heroes (we could be) _ alesso", "samurai - tiesto remix _ r3hab"], 3)

