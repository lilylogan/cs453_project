import numpy as np
import pandas as pd
import os

# set a seed so that we can replicate this study
np.random.seed(42)

# get the song dataset
song_df = pd.read_csv("cleaned_data/songs_cleaned.csv")

# create a directory called "subset_data"
directory = "subset_data"
os.makedirs(directory, exist_ok=True)

# get a subset of 100 songs from the dataset
test_songs = song_df.sample(n=100, random_state=42)
train_songs = song_df.drop(test_songs.index)

# save the two subsets
test_songs.to_csv("subset_data/test_songs.csv", index=False)
train_songs.to_csv("subset_data/train_songs.csv", index=False)

print("saved songs to subset_data directory! :)")