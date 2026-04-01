import streamlit as st
import cv2
import numpy as np
import time

from recommendspot import get_songs, build_user_profile, rank_songs
from database import save_feedback, init_db
from emotion_detect import detect_emotion_from_frame

st.set_page_config(page_title="Emotify", layout="wide")

st.markdown("""
<style>
.stApp {
    background-color: #fff0f3 !important;  /* very light pastel pink */
}

html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
    color: #4a2c2a;
}

/* Cards */
.card {
    background: #f8c8dc;
    border-radius: 20px;
    padding: 18px;
    box-shadow: 0 12px 30px rgba(74,44,42,0.15);
    margin-bottom: 16px;
}

/* Music Player */
.music-player {
    background: #f8c8dc;
    border-radius: 30px;
    padding: 20px;
    margin-top: 20px;
    text-align: center;
}

/* Buttons */
div.stButton > button {
    background: #fddde6;
    border: none;
    text-align: left;
    font-weight: 600;
    color: #4a2c2a;
    padding: 15px 20px;
    border-radius: 20px;
    margin-top: 15px;
    width: 100%;
}

div.stButton > button:hover {
    background: #fbcfe0;
}
</style>
""", unsafe_allow_html=True)
init_db()

st.title("🎧 EMOTIFY")
st.markdown("### Feel it • Detect it • Play it")

col1, col2 = st.columns(2)
with col1:
    st.markdown('<div class="card">📸 Live Emotion Detection</div>', unsafe_allow_html=True)
    FRAME_WINDOW = st.image([])

with col2:
    status_area = st.empty()
    music_area = st.container()

if "camera_stopped" not in st.session_state:
    st.session_state.camera_stopped = False
if "detected_emotion" not in st.session_state:
    st.session_state.detected_emotion = None
if "recommendations" not in st.session_state:
    st.session_state.recommendations = []
if "current_track" not in st.session_state:
    st.session_state.current_track = None
if "liked_songs" not in st.session_state:
    st.session_state.liked_songs = []
if "feedback_message" not in st.session_state:
    st.session_state.feedback_message = ""

if not st.session_state.camera_stopped:
    run = st.checkbox("🎥 Start Webcam")
else:
    run = False
    st.checkbox("🎥 Start Webcam", value=False, disabled=True)

if st.session_state.camera_stopped:
    if st.button("🔄 Restart Webcam"):
        st.session_state.camera_stopped = False
        st.session_state.detected_emotion = None
        st.session_state.recommendations = []
        st.session_state.current_track = None
        st.session_state.feedback_message = ""
        st.rerun()

if run:
    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        emotion, conf, triggered, box = detect_emotion_from_frame(frame)

        if emotion:
            status_area.info(f"Detecting: {emotion.upper()} ({conf*100:.1f}%)")

        if triggered:
            status_area.success(f"Emotion Detected: {emotion.upper()}")

            recs = get_songs(emotion)

            if recs:
                profile = build_user_profile()
                recs = rank_songs(recs, profile)
                st.session_state.detected_emotion = emotion
                st.session_state.recommendations = recs[:5]
                st.session_state.current_track = recs[0]
                st.session_state.camera_stopped = True
            else:
                status_area.error("No tracks found.")

            cap.release()
            break

        if box:
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (154, 165, 129), 3)
            cv2.putText(frame, emotion, (x, y-12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (56, 66, 43), 2)

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        time.sleep(0.01)

    if cap.isOpened():
        cap.release()

if st.session_state.camera_stopped and st.session_state.recommendations:

    main_track   = st.session_state.current_track
    other_tracks = [s for s in st.session_state.recommendations if s != main_track]

    with music_area:
        embed_url = f"https://open.spotify.com/embed/track/{main_track['track_id']}?autoplay=1"
        st.markdown(f"""
        <div class="music-player">
            <h3 style="color:white;">🎵 Now Playing</h3>
            <iframe src="{embed_url}"
                width="100%" height="152"
                frameBorder="0"
                allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture">
            </iframe>
        </div>
        """, unsafe_allow_html=True)

        def handle_feedback(feedback_value):
            if feedback_value == 1:
                if main_track not in st.session_state.liked_songs:
                    st.session_state.liked_songs.append(main_track)
                remaining = st.session_state.recommendations
                st.session_state.feedback_message = "Preference saved!"
            else:
                remaining = [t for t in st.session_state.recommendations if t != main_track]
                st.session_state.feedback_message = "Feedback saved!"

            save_feedback(main_track, st.session_state.detected_emotion, feedback_value)
            profile = build_user_profile()

            if profile and remaining:
                remaining = rank_songs(remaining, profile)

            st.session_state.recommendations = remaining
            st.session_state.current_track = remaining[0] if remaining else None
            st.rerun()

        col1, col2 = st.columns(2)
        with col1:
            if st.button("👍 Like", use_container_width=True):
                handle_feedback(1)
        with col2:
            if st.button("👎 Dislike", use_container_width=True):
                handle_feedback(-1)

        if st.session_state.feedback_message:
            st.markdown(
                f"<p style='text-align:center; font-weight:600;'>"
                f"{st.session_state.feedback_message}</p>",
                unsafe_allow_html=True
            )

        if other_tracks:
            st.markdown("#### Up Next")
            for i, song in enumerate(other_tracks):
                if st.button(f"{song['title']} by {song['artist']}", key=f"song_{i}", use_container_width=True):
                    st.session_state.current_track = song
                    st.rerun()

        if st.session_state.liked_songs:
            st.markdown("---")
            st.markdown("#### ❤️ Liked Songs")
            for song in st.session_state.liked_songs:
                st.markdown(f"""
                <div class="card">
                    🎵 <b>{song['title']}</b> — {song['artist']}<br>
                    <a href="{song['url']}" target="_blank" style="color:#38422b;">Open in Spotify ↗</a>
                </div>
                """, unsafe_allow_html=True)

elif not run and not st.session_state.camera_stopped:
    st.markdown("👆 **Enable the camera to start emotion-based music discovery.**")