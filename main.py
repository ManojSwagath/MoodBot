"""
=============================================================================
  Empathetic AI Companion  —  Facial Emotion Detection & Spoken Response
=============================================================================

  This program uses a webcam to detect a person's face, analyse their
  dominant emotion with a pretrained CNN (FER library), and respond with
  a spoken empathetic message via pyttsx3.

  Runs on:
    • Windows / macOS / Linux  (local PC testing)
    • Raspberry Pi 5 (4 GB)    (deployment target)

  Quick start:
    pip install -r requirements.txt
    python main.py

  Raspberry Pi prerequisites:
    sudo apt update
    sudo apt install espeak-ng libespeak1 libatlas-base-dev libhdf5-dev
    pip install -r requirements.txt

  Press 'q' in the webcam window to quit.
=============================================================================
"""

import sys
import threading
import time
from collections import deque

import cv2
import pyttsx3
import speech_recognition as sr
from fer.fer import FER

# ──────────────────────────────────────────────────────────────
# 1.  CONFIGURATION  — tweak these constants freely
# ──────────────────────────────────────────────────────────────

CAMERA_INDEX      = 0          # 0 = default webcam
FRAME_WIDTH       = 640        # Lower to 320 on Raspberry Pi for speed
FRAME_HEIGHT      = 480        # Lower to 240 on Raspberry Pi for speed
ANALYSE_EVERY_N   = 10         # Run emotion model every Nth frame (saves CPU)
STABLE_COUNT      = 3          # Require N consecutive same-emotion reads before speaking
TTS_RATE          = 160        # Words-per-minute for the speech engine
WINDOW_NAME       = "Empathetic AI Companion"
CONVERSATION_LIMIT = 5         # Max back-and-forth messages per emotion
LISTEN_TIMEOUT    = 5          # Seconds to wait for speech
PHRASE_TIME_LIMIT = 8          # Max seconds of a single phrase

# ──────────────────────────────────────────────────────────────
# 2.  EMOTION → RESPONSE TEMPLATES
# ──────────────────────────────────────────────────────────────

EMOTION_RESPONSES = {
    "happy":    "Hi! You look very happy today! Keep smiling!",
    "sad":      "Hello… You look a little sad. Is everything okay? I'm here for you.",
    "angry":    "Hey, try to relax. Take a deep breath. Everything will be alright.",
}

# Colour per emotion for the on-screen label  (BGR format)
EMOTION_COLOURS = {
    "happy":    (0, 255, 0),     # green
    "sad":      (255, 0, 0),     # blue
    "angry":    (0, 0, 255),     # red
}

# Only track these three emotions
TRACKED_EMOTIONS = set(EMOTION_RESPONSES.keys())

# Conversation follow-up responses based on emotion + keywords
CONVERSATION_RESPONSES = {
    "happy": {
        "default": [
            "That's wonderful! What made your day so great?",
            "I love hearing that! Tell me more!",
            "Your happiness is contagious! What's the good news?",
            "That sounds amazing! Keep that energy going!",
            "I'm so glad to hear that! You deserve it!",
        ],
    },
    "sad": {
        "default": [
            "I understand. It's okay to feel this way. Want to talk about it?",
            "I'm here for you. Sometimes talking helps. What happened?",
            "Take your time. I'm listening whenever you're ready.",
            "You're not alone in this. I care about how you feel.",
            "It's okay to have tough days. Tomorrow can be better.",
        ],
    },
    "angry": {
        "default": [
            "I hear you. What's bothering you? Let it out.",
            "That sounds frustrating. Take a deep breath with me.",
            "It's okay to feel angry. What happened?",
            "I understand your frustration. Let's work through it.",
            "You have every right to feel that way. I'm here to listen.",
        ],
    },
}


# ──────────────────────────────────────────────────────────────
# 3.  EMOTION DETECTION  (FER — pretrained Keras CNN)
# ──────────────────────────────────────────────────────────────

def create_detector() -> FER:
    """
    Initialise the FER emotion detector.

    Uses OpenCV's Haar-cascade face detector under the hood
    (mtcnn=False) which is much lighter and faster — ideal for
    real-time webcam and Raspberry Pi deployment.
    """
    print("[INFO] Loading FER emotion detection model …")
    detector = FER(mtcnn=False)   # mtcnn=True is more accurate but slower
    print("[INFO] Model loaded successfully.")
    return detector


def detect_emotion(frame, detector: FER) -> tuple:
    """
    Analyse a single BGR frame and return:
        (dominant_emotion: str | None,
         bounding_box:     tuple(x,y,w,h) | None,
         all_scores:       dict | None)

    Returns (None, None, None) when no face is detected.
    """
    try:
        results = detector.detect_emotions(frame)

        if not results:
            return None, None, None

        # Take the face with the highest detection confidence
        top = max(results, key=lambda r: max(r["emotions"].values()))
        box      = top["box"]                # (x, y, w, h)
        emotions = top["emotions"]           # {'happy': 0.92, 'sad': 0.01, …}
        dominant = max(emotions, key=emotions.get)

        return dominant, box, emotions

    except Exception as exc:
        print(f"[WARN] Emotion detection error: {exc}")
        return None, None, None


# ──────────────────────────────────────────────────────────────
# 4.  TEXT-TO-SPEECH  (pyttsx3 — offline, cross-platform)
# ──────────────────────────────────────────────────────────────

def create_tts_engine() -> pyttsx3.Engine:
    """
    Create and configure a pyttsx3 TTS engine.
    Uses SAPI5 on Windows, espeak-ng on Linux / Raspberry Pi.
    """
    engine = pyttsx3.init()
    engine.setProperty("rate", TTS_RATE)
    # Optionally pick a different voice:
    # voices = engine.getProperty("voices")
    # engine.setProperty("voice", voices[1].id)   # e.g. index 1 = female
    return engine


def speak_text(text: str, engine: pyttsx3.Engine):
    """Speak text synchronously (call from threads only)."""
    try:
        engine.say(text)
        engine.runAndWait()
    except Exception as exc:
        print(f"[WARN] TTS error: {exc}")


def speak_response(emotion: str, engine: pyttsx3.Engine, spoken_flag: dict):
    """
    Speak the empathetic response for *emotion* in a daemon thread
    so the camera loop is never blocked.

    *spoken_flag* is a mutable dict {"busy": bool} shared with the
    main loop to prevent overlapping speech.
    """
    text = EMOTION_RESPONSES.get(emotion, "Hello! How are you?")

    def _speak():
        spoken_flag["busy"] = True
        try:
            engine.say(text)
            engine.runAndWait()
        except Exception as exc:
            print(f"[WARN] TTS error: {exc}")
        finally:
            spoken_flag["busy"] = False

    thread = threading.Thread(target=_speak, daemon=True)
    thread.start()


# ──────────────────────────────────────────────────────────────
# 4b. SPEECH RECOGNITION  (Google free API via SpeechRecognition)
# ──────────────────────────────────────────────────────────────

def create_recognizer() -> sr.Recognizer:
    """Create and configure a speech recognizer."""
    recognizer = sr.Recognizer()
    recognizer.dynamic_energy_threshold = True
    return recognizer


def listen_for_speech(recognizer: sr.Recognizer) -> str | None:
    """
    Listen to the microphone and return the transcribed text.
    Returns None if nothing was heard or recognition failed.
    """
    try:
        with sr.Microphone() as source:
            print("[LISTEN] Listening … (speak now)")
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(
                source, timeout=LISTEN_TIMEOUT, phrase_time_limit=PHRASE_TIME_LIMIT
            )
        print("[LISTEN] Processing speech …")
        text = recognizer.recognize_google(audio)
        print(f"[HEARD]  \"{text}\"")
        return text
    except sr.WaitTimeoutError:
        print("[LISTEN] No speech detected (timeout).")
        return None
    except sr.UnknownValueError:
        print("[LISTEN] Could not understand the audio.")
        return None
    except sr.RequestError as exc:
        print(f"[WARN] Speech recognition service error: {exc}")
        return None


# ──────────────────────────────────────────────────────────────
# 4c. CONVERSATION ENGINE
# ──────────────────────────────────────────────────────────────

def run_conversation(emotion: str, engine: pyttsx3.Engine,
                     recognizer: sr.Recognizer, spoken_flag: dict):
    """
    Run a back-and-forth voice conversation in a background thread.
    Starts after the initial emotion greeting, up to CONVERSATION_LIMIT exchanges.
    """
    def _converse():
        spoken_flag["busy"] = True
        try:
            # Speak the initial emotion greeting
            greeting = EMOTION_RESPONSES.get(emotion, "Hello! How are you?")
            print(f'[SPEAK]   "{greeting}"')
            speak_text(greeting, engine)

            responses = CONVERSATION_RESPONSES.get(emotion, {}).get("default", [])
            msg_count = 0

            while msg_count < CONVERSATION_LIMIT:
                # Listen for user reply
                user_text = listen_for_speech(recognizer)

                if user_text is None:
                    # No speech detected — end conversation
                    farewell = "It seems like you're quiet. I'm here whenever you want to talk!"
                    print(f'[SPEAK]   "{farewell}"')
                    speak_text(farewell, engine)
                    break

                msg_count += 1

                if msg_count >= CONVERSATION_LIMIT:
                    closing = "It was really nice talking to you! Take care and stay strong!"
                    print(f'[SPEAK]   "{closing}"')
                    speak_text(closing, engine)
                    break

                # Pick a response from the pool
                reply = responses[msg_count % len(responses)]
                print(f'[SPEAK]   "{reply}"')
                speak_text(reply, engine)

        finally:
            spoken_flag["busy"] = False
            print("[CONVO] Conversation ended.")

    thread = threading.Thread(target=_converse, daemon=True)
    thread.start()


# ──────────────────────────────────────────────────────────────
# 5.  CAMERA HELPER  (cross-platform: Windows + Raspberry Pi)
# ──────────────────────────────────────────────────────────────

def open_camera(index: int = CAMERA_INDEX) -> cv2.VideoCapture:
    """
    Try multiple backends so the same code works on both
    Windows (DirectShow) and Raspberry Pi (V4L2).
    """
    backends = [
        ("DirectShow", cv2.CAP_DSHOW),
        ("V4L2",       cv2.CAP_V4L2),
        ("Any",        cv2.CAP_ANY),
    ]
    for name, backend in backends:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            cap.set(cv2.CAP_PROP_FOURCC,
                    cv2.VideoWriter_fourcc(*"MJPG"))
            print(f"[INFO] Camera opened with {name} backend.")
            return cap
    return None


# ──────────────────────────────────────────────────────────────
# 6.  DRAW OVERLAY  — bounding box + emotion label on frame
# ──────────────────────────────────────────────────────────────

def draw_overlay(frame, emotion: str, box: tuple, scores: dict):
    """
    Draw a rectangle around the detected face and label it with
    the dominant emotion and its confidence score.
    """
    x, y, w, h = box
    colour = EMOTION_COLOURS.get(emotion, (255, 255, 255))

    # Bounding box
    cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)

    # Label background
    confidence = scores.get(emotion, 0.0)
    label = f"{emotion.upper()} ({confidence:.0%})"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(frame, (x, y - th - 14), (x + tw + 6, y), colour, -1)

    # Label text
    cv2.putText(frame, label, (x + 3, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

    # Bottom-left instructions
    cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)


# ──────────────────────────────────────────────────────────────
# 7.  MAIN LOOP
# ──────────────────────────────────────────────────────────────

def main():
    """
    Core camera loop:
      1. Capture a frame from the webcam.
      2. Every N frames, run the FER emotion detector.
      3. Overlay the detected emotion on the video feed.
      4. If the emotion is stable for STABLE_COUNT consecutive
         reads *and* differs from the last spoken emotion,
         speak the empathetic response in a background thread.
      5. Repeat until the user presses 'q'.
    """

    # --- Initialise components ---
    cap = open_camera()
    if cap is None:
        print("[ERROR] Cannot open webcam. Check camera connection.")
        sys.exit(1)

    detector    = create_detector()
    tts_engine  = create_tts_engine()
    recognizer  = create_recognizer()

    # --- State variables ---
    frame_count    = 0              # Total frames captured
    current_emotion = None          # Latest detected emotion
    current_box     = None          # Latest bounding box
    current_scores  = None          # Latest emotion scores dict
    last_spoken     = None          # Last emotion that was spoken aloud
    emotion_buffer  = deque(maxlen=STABLE_COUNT)  # Rolling window for stability
    spoken_flag     = {"busy": False}             # Shared flag — is TTS speaking?

    print("\n" + "=" * 55)
    print("  EMPATHETIC AI COMPANION  —  Running")
    print("  Press 'q' in the video window to quit.")
    print("=" * 55 + "\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Failed to grab frame. Retrying …")
                time.sleep(0.1)
                continue

            frame_count += 1

            # --- Run emotion detection every N frames ---
            if frame_count % ANALYSE_EVERY_N == 0:
                emotion, box, scores = detect_emotion(frame, detector)

                if emotion is not None:
                    # Only react to tracked emotions
                    if emotion not in TRACKED_EMOTIONS:
                        continue
                    current_emotion = emotion
                    current_box     = box
                    current_scores  = scores
                    emotion_buffer.append(emotion)

                    # Console output
                    print(f"[EMOTION] {emotion.upper():>10s}  "
                          f"(confidence {scores[emotion]:.0%})")

                    # --- Speak & start conversation when emotion is STABLE and NEW ---
                    all_same = (len(emotion_buffer) == STABLE_COUNT
                                and len(set(emotion_buffer)) == 1)

                    if (all_same
                            and emotion != last_spoken
                            and not spoken_flag["busy"]):
                        run_conversation(emotion, tts_engine, recognizer, spoken_flag)
                        last_spoken = emotion

            # --- Draw overlay if we have a detection ---
            if current_emotion and current_box is not None:
                draw_overlay(frame, current_emotion, current_box, current_scores)

            # --- Show the video feed ---
            cv2.imshow(WINDOW_NAME, frame)

            # --- Quit on 'q' key ---
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n[INFO] Quitting …")
                break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")

    finally:
        # --- Clean up ---
        cap.release()
        cv2.destroyAllWindows()
        tts_engine.stop()
        print("[INFO] Camera released. Goodbye!")


# ──────────────────────────────────────────────────────────────
# 8.  ENTRY POINT
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
