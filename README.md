# Emotify
A Music Recommendation System using Emotion Detection

## Project Overview
This project detects a user’s facial emotion using a **Swin Transformer** and recommends music based on that emotion.  
It integrates:
- Facial emotion detection
- Spotify API for music recommendations
- Content-based recommendation system with user feedback

## Pipeline

1. **User Input** – Upload or capture a facial image.
2. **Emotion Detection** – Fine-tuned Swin Transformer predicts the user’s emotion.
3. **Recommendation API** – Calls Spotify API and recommends songs from Spotify.
4. **Database** – Stores user preferences (like, dislike).
5. **Content-Based Filtering** – Adjusts recommendations based on user feedback.

