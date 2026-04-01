import random
import sqlite3
import numpy as np
import spotipy
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from spotipy.oauth2 import SpotifyClientCredentials

CLIENT_ID = YOUR_CLIENT_ID
CLIENT_SECRET = YOUR_CLIENT_SECRET

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET
))

def init_db():
    conn = sqlite3.connect("feedback.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            song_id TEXT,
            song_name TEXT,
            emotion TEXT,
            artist TEXT,
            feedback INTEGER
        )
    """)
    conn.commit()
    conn.close()

init_db()

def get_query_from_emotion(emotion):
    emotion = emotion.lower()
    mood_queries = {
        'happy':   ['happy upbeat pop', 'feel good songs', 'party hits'],
        'sad':     ['sad songs', 'heartbreak songs', 'slow acoustic', 'melancholy music'],
        'angry':   ['angry rock', 'metal rage', 'intense workout', 'hard rock songs'],
        'neutral': ['chill songs', 'indie chill']
    }
    return mood_queries.get(emotion, ['chill songs'])

def get_batch_audio_features(track_ids):
    if not track_ids:
        return []
    try:
        features_list = sp.audio_features(track_ids)
        parsed = []
        for f in features_list:
            if f is None:
                parsed.append([0, 0, 0, 0, 0])
            else:
                parsed.append([
                    f["valence"],
                    f["energy"],
                    f["danceability"],
                    min(f["tempo"] / 200.0, 1.0),
                    f["acousticness"]
                ])
        return parsed
    except Exception as e:
        print(f"Error fetching audio features: {e}")
        return [[0, 0, 0, 0, 0]] * len(track_ids)

def get_songs(emotion, limit=20):
    queries = get_query_from_emotion(emotion)
    songs = []

    for query in queries:
        try:
            offset = random.randint(0, 20)
            results = sp.search(q=query, type='track', limit=10, offset=offset)
            tracks = results['tracks']['items']

            for track in tracks:
                if not track or not track.get("id"):
                    continue
                songs.append({
                    'track_id': track['id'],
                    'title': track['name'],
                    'artist': track['artists'][0]['name'],
                    'url': track['external_urls']['spotify'],
                    'album_cover': track['album']['images'][0]['url'] if track['album']['images'] else None,
                    'emotion': emotion
                })
        except Exception as e:
            print(f"Error with query '{query}': {e}")

    if not songs:
        try:
            results = sp.search(q="top hits", type='track', limit=10)
            for track in results['tracks']['items']:
                if track and track.get('id'):
                    songs.append({
                        'track_id': track['id'],
                        'title': track['name'],
                        'artist': track['artists'][0]['name'],
                        'url': track['external_urls']['spotify'],
                        'album_cover': track['album']['images'][0]['url'] if track['album']['images'] else None,
                        'emotion': 'neutral'
                    })
        except Exception as e:
            print(f"Final fallback failed: {e}")
            return []

    track_ids = [s['track_id'] for s in songs]
    features  = get_batch_audio_features(track_ids)

    for song, feat in zip(songs, features):
        song['audio_features'] = feat

    random.shuffle(songs)
    return songs[:limit]

def build_user_profile():
    conn = sqlite3.connect("feedback.db")
    c = conn.cursor()

    c.execute("""
        SELECT feedback, emotion, artist
        FROM feedback
    """)
    rows = c.fetchall()
    conn.close()

    if len(rows) < 3:
        return None

    feedbacks = np.array([r[0] for r in rows])
    emotions  = [r[1] for r in rows]
    artists   = [r[2] for r in rows]


    weighted_profile = np.zeros(5)

    liked_mask   = feedbacks == 1
    emotion_pref = Counter(np.array(emotions)[liked_mask])
    artist_pref  = Counter(np.array(artists)[liked_mask])

    return {
        "audio_profile": weighted_profile, 
        "emotion_pref": emotion_pref,
        "artist_pref": artist_pref
    }

def rank_songs(songs, user_profile):
    if user_profile is None or not songs:
        return songs

    audio_profile = user_profile["audio_profile"]
    emotion_pref  = user_profile["emotion_pref"]
    artist_pref   = user_profile["artist_pref"]
    user_vector   = np.array(audio_profile).reshape(1, -1)

    for song in songs:
        feat  = song.get("audio_features", [0, 0, 0, 0, 0])
        f_vec = np.array(feat).reshape(1, -1)

        audio_score   = cosine_similarity(user_vector, f_vec)[0][0] if np.any(f_vec) else 0
        emotion_score = emotion_pref.get(song["emotion"], 0)
        artist_score  = artist_pref.get(song["artist"], 0)

        song["score"] = (5 * audio_score) + (2 * emotion_score) + (3 * artist_score)

    return sorted(songs, key=lambda x: x.get("score", 0), reverse=True)

def save_feedback(song, emotion, feedback):
    conn = sqlite3.connect("feedback.db")
    c = conn.cursor()

    c.execute("""
        INSERT INTO feedback (song_id, song_name, emotion, artist, feedback)
        VALUES (?, ?, ?, ?, ?)
    """, (
        song["track_id"],
        song["title"],
        emotion,
        song["artist"],
        feedback
    ))

    conn.commit()
    conn.close()
