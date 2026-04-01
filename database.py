import sqlite3

DB_FILE = "feedback.db"

def init_db():
    """Initialize the feedback database and table."""
    conn = sqlite3.connect(DB_FILE)
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


def save_feedback(track, emotion, feedback_value):
    """
    Save user's like/dislike feedback for a track.

    :param track: dict with keys 'track_id', 'title', 'artist'
    :param emotion: str
    :param feedback_value: int (1=like, -1=dislike)
    """
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        INSERT INTO feedback (song_id, song_name, emotion, artist, feedback)
        VALUES (?, ?, ?, ?, ?)
    """, (
        track['track_id'],
        track['title'],
        emotion,
        track['artist'],
        feedback_value
    ))
    conn.commit()
    conn.close()