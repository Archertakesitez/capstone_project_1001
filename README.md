# DS-GA 1001 Capstone Project: Spotify Songs Data Analysis
Capstone Project for NYU's Graduate course DS-GA 1001

## Table of Contents

- [Introduction](#introduction)
- [Data Description](#data-description)
- [Authors](#authors)

## Introduction
We are performing a very comprehensive data analysis on both 'spotify52kData.csv' and 'starRatings.csv' files. The datasets are included in the [data](/data) directory; since the starRating file has exceeded the GitHub file size limit, we have uploaded its .zip version, so you have to unzip it before running our codes. We carefully followed the guidelines of [IDS Capstone project spec sheet.pdf](/IDS_Capstone_project_spec_sheet.pdf) and applied different techniques, including multiple linear regression, lasso/ridge regression, significance tests, PCA, K-means clustering, logistic regression, SVM, Random Forest, MLP neural network, recommendation system, and so forth, to solve each of the given questions. 
Our solutions are included in [capstone_DS14.pdf](/capstone_DS14.pdf) and their assocaited codes are included in both [capstone_ds14.ipynb](/capstone_ds14.ipynb) and [DS14_code.py](/DS14_code.py). You can choose either the .ipynb or the .py file to review and run our codes. 

## Data Description
This “spotify52kData.csv” dataset consists of data on 52,000 songs that were randomly picked from a variety of genres sorted in alphabetic order (a as in “acoustic” to h as in “hiphop”). For the purposes of this analysis, we assume that the data for one song are independent for data from other songs. The first row of this dataset is the column headers, and the remaining rows (from 2 to 52001) are specific individual songs.
### features:
- **songNumber**: the track ID of the song, from 0 to 51999.
- Column 2: artist(s) – the artist(s) who are credited with creating the song.
- Column 3: album_name – the name of the album
- **track_name**: the title of the specific track corresponding to the track ID
- **popularity**: this is an important metric provided by spotify, an integer from 0 to 100, where a higher number corresponds to a higher number of plays on spotify.
- **duration**: this is the duration of the song in ms. A ms is a millisecond. There are a thousand milliseconds in a second and 60 seconds in a minute.
- **explicit**: this is a binary (Boolean) categorical variable. If it is true, the lyrics of the track contain explicit language, e.g. foul language, swear words or content that some consider indecent. Column 8: danceability – this is an audio feature provided by the Spotify API. It tries to quantify how easy it is to dance to the song (presumably capturing tempo and beat), and varies from 0 to 1.
- **energy**: this is an audio feature provided by the Spotify API. It tries to quantify how “hard” a song goes. Intense songs have more energy, softer/melodic songs lower energy, it varies from 0 to 1. Column 10: key – what is the key of the song, from A to G# (mapped to categories 0 to 11).
- **loudness**: average loudness of a track in dB (decibels)
- **mode**: this is a binary categorical variable. 1 = song is in major, 0 – song is in minor
- **speechiness**: quantifies how much of the song is spoken, varying from 0 (fully instrumental songs) to 1 (songs that consist entirely of spoken words).
- **acousticness**: varies from 0 (song contains exclusively synthesized sounds) to 1 (song features exclusively acoustic instruments like acoustic guitars, pianos or orchestral instruments).
- **instrumentalness**: basically the inverse of speechiness, varying from 1 (for songs without any vocals) to 0.
- **liveness**: this is an audio feature provided by the Spotify API. It tries to quantify how likely the recording was live in front of an audience (values close to 1) vs. how likely it was recorded in a studio without a live audience (values close to 0).
- **valence**: this is an audio feature provided by the Spotify API. It tries to quantify how uplifting a song is. Songs with a positive mood =close to 1 and songs with a negative mood =close to 0 Column 18: tempo – speed of the song in beats per minute (BPM)
- **time_signature**: how many beats there are in a measure (usually 4 or 3)
- **track_genre**: genre assigned by spotify, e.g. “blues” or “classical”. 1k songs per genre.
In addition, there is a file (“starRatings.csv”) that contains explicit feedback, specifically star ratings from 10k users on 5k songs they listened to, on a scale from 0 (lowest) to 4 (highest). In this file, there are no headers. Each row corresponds to a user and each column to a song, specifically to the first 5k rows (songs) in the spotify52kData.csv dataset, in the same order. Missing data is represented as nans.

## Authors
- **[Erchi Zhang](https://github.com/Archertakesitez)**
- **[Joz Zhou](https://github.com/jozzhou99)**
