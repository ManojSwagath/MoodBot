# MoodBot — Design & Architecture

> **Empathetic AI Companion** | Facial Emotion Detection, Spoken Response & Robotic Actuation

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [High-Level Architecture](#2-high-level-architecture)
3. [Hardware Stack](#3-hardware-stack)
4. [Software Stack](#4-software-stack)
5. [System Components](#5-system-components)
   - [Emotion Detection Engine](#51-emotion-detection-engine)
   - [Text-to-Speech (TTS)](#52-text-to-speech-tts)
   - [Speech Recognition](#53-speech-recognition)
   - [Conversation Engine](#54-conversation-engine)
   - [Visual Overlay](#55-visual-overlay)
   - [Camera Manager](#56-camera-manager)
6. [Data Flow](#6-data-flow)
7. [Threading Model](#7-threading-model)
8. [Wiring & Hardware Connections](#8-wiring--hardware-connections)
9. [Configuration Reference](#9-configuration-reference)
10. [Deployment Targets](#10-deployment-targets)
11. [Project File Structure](#11-project-file-structure)
12. [Future Roadmap](#12-future-roadmap)

---

## 1. Project Overview

MoodBot is an emotionally intelligent robotic companion that:

- **Sees** a person's face via webcam and detects their dominant emotion using a CNN.
- **Speaks** an empathetic, contextual response via offline TTS.
- **Listens** to the user's reply via microphone and holds a short conversation.
- **Moves** physically in response to detected emotions using motors controlled by an ESP32.

The system targets two deployment environments:

| Environment | Purpose |
|---|---|
| Windows PC | Development, testing, demo |
| Raspberry Pi 5 (4 GB) | Embedded deployment on the robot chassis |

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        MOODBOT SYSTEM                           │
│                                                                 │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │  Webcam  │───▶│   OpenCV     │───▶│  FER CNN Model       │  │
│  └──────────┘    │  Frame Grab  │    │  (Emotion Detection) │  │
│                  └──────────────┘    └──────────┬───────────┘  │
│                                                  │ emotion      │
│  ┌──────────┐    ┌──────────────┐    ┌──────────▼───────────┐  │
│  │   Mic    │───▶│   Speech     │◀───│  Conversation Engine │  │
│  └──────────┘    │  Recognition │    │  (State Machine)     │  │
│                  └──────┬───────┘    └──────────┬───────────┘  │
│                         │ text                  │ response text │
│                         │            ┌──────────▼───────────┐  │
│                         └───────────▶│   pyttsx3 TTS        │  │
│                                      │   (Spoken Response)  │  │
│                                      └──────────────────────┘  │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │     OpenCV Display Window  (Live Feed + Overlay)        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Raspberry Pi 5  ──USB Serial──▶  ESP32  ──▶  L298N     │  │
│  │                                           ──▶  4x Motors │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Hardware Stack

| Component | Model / Spec | Role |
|---|---|---|
| **Main Computer** | Raspberry Pi 5 — 4 GB RAM | Runs the full Python AI pipeline |
| **Microcontroller** | ESP32 Dev Module | Motor control via USB Serial commands |
| **Motor Driver** | L298N Dual H-Bridge | Drives 4 DC motors from ESP32 PWM signals |
| **Motors** | 4× DC motors (2.5V) | Physical movement of the robot chassis |
| **Camera** | Webcam / RPi Camera Module | Captures live video for emotion detection |
| **Microphone** | USB Microphone | Captures user speech for recognition |
| **Power** | Battery pack | Powers motors and RPi |

---

## 4. Software Stack

| Layer | Library / Tool | Version | Purpose |
|---|---|---|---|
| **Language** | Python | 3.11+ | Core application |
| **Computer Vision** | OpenCV (`cv2`) | ≥ 4.8.0 | Frame capture, display, face overlay |
| **Emotion Detection** | FER | ≥ 22.5.1 | Pretrained CNN — classifies 7 emotions |
| **Deep Learning Backend** | TensorFlow / Keras | (via FER) | Powers the CNN inference |
| **Text-to-Speech** | pyttsx3 | ≥ 2.98 | Offline TTS (SAPI5 on Windows, espeak-ng on Linux) |
| **Speech Recognition** | SpeechRecognition | ≥ 3.10 | Mic → text via Google free API |
| **Audio I/O** | PyAudio | ≥ 0.2.14 | Microphone access for SpeechRecognition |
| **Numerical** | NumPy | ≥ 1.24 | Array operations (used by OpenCV / TF) |
| **Firmware** | Arduino (C++) | — | ESP32 motor control sketch |

---

## 5. System Components

### 5.1 Emotion Detection Engine

**File:** `main.py` — `create_detector()`, `detect_emotion()`

- Uses the **FER** library with an OpenCV **Haar-cascade** face detector (`mtcnn=False`) for real-time performance.
- Processes one frame every `ANALYSE_EVERY_N` frames (default: 10) to reduce CPU load.
- Returns the **dominant emotion**, **bounding box** `(x, y, w, h)`, and a **scores dict** for all 7 emotions.
- A **stability buffer** (`deque` of size `STABLE_COUNT = 3`) ensures the same emotion must appear 3 consecutive readings before triggering a response — preventing jitter.

**Tracked Emotions:**

| Emotion | On-Screen Colour | Initial Response |
|---|---|---|
| `happy` | Green | "Hi! You look very happy today! Keep smiling!" |
| `sad` | Blue | "Hello… You look a little sad. Is everything okay?" |
| `angry` | Red | "Hey, try to relax. Take a deep breath." |

---

### 5.2 Text-to-Speech (TTS)

**File:** `main.py` — `create_tts_engine()`, `speak_text()`, `speak_response()`

- **pyttsx3** runs fully **offline** — no API key or internet required.
- Uses **SAPI5** on Windows and **espeak-ng** on Linux / Raspberry Pi.
- Configured at **160 words per minute** (`TTS_RATE`).
- Speech runs in a **daemon thread** so the camera loop is never blocked.
- A shared `spoken_flag = {"busy": bool}` mutex prevents overlapping speech.

---

### 5.3 Speech Recognition

**File:** `main.py` — `create_recognizer()`, `listen_for_speech()`

- Uses **Google Speech Recognition** (free tier, internet required).
- Adjusts for ambient noise automatically before each listen.
- Configured timeouts: `LISTEN_TIMEOUT = 5s`, `PHRASE_TIME_LIMIT = 8s`.
- Returns `None` gracefully on timeout, unintelligible audio, or API error.

---

### 5.4 Conversation Engine

**File:** `main.py` — `run_conversation()`, `CONVERSATION_RESPONSES`

- Triggered when an emotion is **stable and new** (not already spoken).
- Runs entirely in a **background thread** to keep the camera loop live.
- Flow:
  1. Speak the **initial emotion greeting**.
  2. Listen for a user reply (up to `LISTEN_TIMEOUT` seconds).
  3. Respond with a **context-aware reply** from the pool for that emotion.
  4. Repeat up to `CONVERSATION_LIMIT = 5` exchanges.
  5. Conclude with a farewell message.
- Each emotion has a pool of **5 empathetic follow-up responses** that cycle.

---

### 5.5 Visual Overlay

**File:** `main.py` — `draw_overlay()`

- Draws a **coloured bounding box** around the detected face.
- Renders an **emotion label** with confidence percentage (e.g., `HAPPY (87%)`).
- Label background is colour-coded per emotion for quick visual feedback.
- Displays `"Press 'q' to quit"` instruction at the bottom of the frame.

---

### 5.6 Camera Manager

**File:** `main.py` — `open_camera()`

- Tries multiple OpenCV backends in order: **DirectShow → V4L2 → Any**.
- This ensures the same code works seamlessly on both Windows and Raspberry Pi without changes.
- Configures capture resolution and MJPEG codec on open.

---

## 6. Data Flow

```
┌─────────────┐
│   Webcam    │
└──────┬──────┘
       │ BGR frame (640×480)
       ▼
┌─────────────────────┐     every Nth frame
│   Frame Capture     │ ──────────────────────▶ ┌─────────────────────┐
│   (OpenCV loop)     │                          │   FER CNN Detector  │
└─────────────────────┘                          └──────────┬──────────┘
       ▲                                                     │
       │  draw overlay                          (emotion, box, scores)
       │                                                     ▼
       │                                        ┌────────────────────────┐
       └────────────────────────────────────────│  Stability Buffer      │
                                                │  deque(maxlen=3)       │
                                                └──────────┬─────────────┘
                                                           │ stable & new emotion
                                                           ▼
                                                ┌──────────────────────────┐
                                                │   Conversation Engine    │
                                                │   (background thread)    │
                                                └────┬─────────────────────┘
                                                     │
                                     ┌───────────────┼────────────────────┐
                                     ▼               ▼                    ▼
                               ┌──────────┐   ┌──────────┐         ┌──────────┐
                               │  pyttsx3 │   │  Google  │         │  pyttsx3 │
                               │  Greet   │   │  Speech  │         │  Reply   │
                               └──────────┘   │  Recog.  │         └──────────┘
                                              └──────────┘
```

---

## 7. Threading Model

The application uses Python's `threading` module to prevent the camera loop from blocking on slow I/O operations.

```
Main Thread
│
├── camera loop (OpenCV imshow, waitKey) ◀─── always running at ~30 fps
│
└── [spawns] Conversation Thread (daemon=True)
       │
       ├── speak initial greeting   (pyttsx3 — blocking)
       ├── listen for user reply    (SpeechRecognition — blocking)
       ├── speak follow-up reply    (pyttsx3 — blocking)
       └── ... repeat up to CONVERSATION_LIMIT times
```

**Shared State:**

| Variable | Type | Purpose |
|---|---|---|
| `spoken_flag["busy"]` | `dict[bool]` | Prevents two conversations overlapping |
| `last_spoken` | `str \| None` | Prevents re-triggering the same emotion |
| `emotion_buffer` | `deque` | Rolling stability check window |

---

## 8. Wiring & Hardware Connections

### Raspberry Pi 5 → ESP32 (USB Serial)

```
Raspberry Pi 5                ESP32
──────────────                ─────
USB-A  ────────── USB cable ──── USB (auto COM port)
```

The RPi sends a plain-text command string (`"HAPPY\n"`, `"SAD\n"`, `"ANGRY\n"`, `"STOP\n"`) over the serial port. The ESP32 firmware parses it and drives the motors.

---

### ESP32 → L298N Motor Driver

| ESP32 Pin | L298N Pin | Function |
|---|---|---|
| GPIO 14 | ENA | PWM speed — Motor A |
| GPIO 26 | IN1 | Direction — Motor A |
| GPIO 27 | IN2 | Direction — Motor A |
| GPIO 12 | ENB | PWM speed — Motor B |
| GPIO 25 | IN3 | Direction — Motor B |
| GPIO 33 | IN4 | Direction — Motor B |

---

### Emotion → Motor Behaviour

| Detected Emotion | Motor Command | Robot Behaviour |
|---|---|---|
| `happy` | `HAPPY` → `moveForward()` | Robot approaches the user |
| `angry` | `ANGRY` → `moveBackward()` | Robot backs away |
| `sad` | `SAD` → `stopMotors()` | Robot stays present, holds position |

---

## 9. Configuration Reference

All tunable constants are at the top of `main.py`:

| Constant | Default | Description |
|---|---|---|
| `CAMERA_INDEX` | `0` | Webcam device index |
| `FRAME_WIDTH` | `640` | Capture width in pixels (use 320 on RPi) |
| `FRAME_HEIGHT` | `480` | Capture height in pixels (use 240 on RPi) |
| `ANALYSE_EVERY_N` | `10` | Run FER model every Nth frame |
| `STABLE_COUNT` | `3` | Consecutive reads before triggering response |
| `TTS_RATE` | `160` | TTS speech rate (words per minute) |
| `CONVERSATION_LIMIT` | `5` | Max exchanges in one conversation session |
| `LISTEN_TIMEOUT` | `5` | Seconds to wait for user to start speaking |
| `PHRASE_TIME_LIMIT` | `8` | Max length of a single phrase (seconds) |

**Raspberry Pi optimised settings:**
```python
FRAME_WIDTH     = 320
FRAME_HEIGHT    = 240
ANALYSE_EVERY_N = 20
```

---

## 10. Deployment Targets

### Windows (Development)

```
Webcam ──▶ Python (main.py) ──▶ OpenCV window
                           ──▶ pyttsx3 (SAPI5 voice)
                           ──▶ Google Speech API (internet)
```

### Raspberry Pi 5 (Robot Deployment)

```
RPi Camera / USB Webcam ──▶ Python (main.py) ──▶ HDMI display (optional)
USB Mic                 ──▶                  ──▶ espeak-ng TTS
                                             ──▶ USB Serial ──▶ ESP32 ──▶ L298N ──▶ Motors
```

Auto-start on boot via `/etc/rc.local`:
```bash
su pi -c "cd /home/pi/MoodBot && source .venv/bin/activate && python main.py &"
```

---

## 11. Project File Structure

```
MoodBot/
│
├── main.py               # Full application — all modules in one file
├── requirements.txt      # Python dependencies
├── CONNECTION.md         # Wiring, deployment, and RPi setup guide
├── Details.md            # Project description and hardware component list
├── ARCHITECTURE.md       # This file — design & architecture overview
└── .gitignore            # Excludes .venv/, __pycache__/, etc.
```

---

## 12. Future Roadmap

| Feature | Description |
|---|---|
| **ESP32 Serial Integration** | Add `pyserial` to `main.py` to send emotion commands over USB to the ESP32 |
| **MTCNN Face Detector** | Switch `FER(mtcnn=True)` for higher accuracy when processing power allows |
| **Ultrasonic Obstacle Avoidance** | Integrate HC-SR04 sensors on the ESP32 to prevent collisions during `moveForward()` |
| **Neutral / Surprise / Fear / Disgust** | Expand `TRACKED_EMOTIONS` and add response templates for all 7 FER emotion classes |
| **RNN / LSTM for Speech Emotion** | Add audio-based emotion detection alongside facial analysis for multi-modal fusion |
| **Offline Speech Recognition** | Replace Google API with `Whisper` (OpenAI) or `Vosk` for fully offline operation on RPi |
| **Web Dashboard** | Real-time emotion log and conversation history via Flask + WebSocket |
| **Battery Monitor** | Read voltage via ESP32 ADC and warn when battery is low |
