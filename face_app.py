"""Face recognition system: database + engine + fully spoken registration + live test, in one file.

Pipeline: Webcam -> Detect Face -> Generate Encoding -> Compare With Database -> Name/Relationship or UNKNOWN.
Known encodings are loaded from SQLite once at startup and kept in RAM, so recognition
never touches the database on a per-frame basis.

IMPORTANT: run this from a normal terminal (Command Prompt / PowerShell), not from
IDLE. IDLE's Shell is a separate console talking to the script over an internal
pipe, and mixing that with a live OpenCV window plus blocking microphone calls is a
known source of the script appearing to "hang" -- it isn't frozen, it's just that
keystrokes/audio can't reach whichever window doesn't have focus. A plain terminal
doesn't have this problem.

Usage:
    python face_system.py                                # open webcam, recognize live (default tolerance 0.6, voice on)
                                                           #   unknown faces trigger a spoken name/relationship conversation
                                                           #   automatically -- no keypress needed. ESC = quit,
                                                           #   M = open the entry-management menu (typed, for cleanup/editing)
    python face_system.py test [tolerance] [--no-voice]   # same as above, with options
    python face_system.py register "Rahul" "Son"          # register one person from webcam and exit (typed, no camera loop)
    python face_system.py register "Rahul" "Son" path/to/photo.jpg   # register from an image file and exit
    python face_system.py manage                          # entry-management menu on its own (no live video)
    python face_system.py directions <lat> <lon>          # speak walking directions home from a given GPS fix
                                                           #   (needs HOME_LAT, HOME_LON; ORS_API_KEY optional for
                                                           #   real routing, falls back to an offline estimate)
"""

from arduino_interface import ArduinoInterface
import math
import os
import pickle
import platform
import sqlite3
import sys
import time
from pathlib import Path

import cv2
import face_recognition
import numpy as np
import pyttsx3
import requests
import speech_recognition as sr

DB_PATH = Path(__file__).parent / "face_database.db"
DEFAULT_TOLERANCE = 0.6
IS_WINDOWS = platform.system() == "Windows"

# Home coordinates are read from the environment, never hardcoded here — this repo is
# public, and a real home address is the kind of thing that shouldn't sit in git history
# forever. ORS_API_KEY is optional (free signup at openrouteservice.org, no billing/card
# required) — without it, `directions` still works using an offline straight-line
# distance/direction estimate instead of real turn-by-turn walking directions.
#   set ORS_API_KEY=...      (Windows, optional)   /   export ORS_API_KEY="..."  (Mac/Linux)
#   set HOME_LAT=9.094020
#   set HOME_LON=76.491208
ORS_API_KEY_ENV = "ORS_API_KEY"
ORS_DIRECTIONS_URL = "https://api.openrouteservice.org/v2/directions/foot-walking"
HOME_LAT_ENV = "HOME_LAT"
HOME_LON_ENV = "HOME_LON"

# How long to wait before asking about the *same* still-unknown face again
# after a registration attempt fails or is skipped. Prevents re-asking every frame.
SKIP_COOLDOWN_SECONDS = 8
arduino = ArduinoInterface("COM7")
#arduino.test_buzzer()
print("Testing Arduino...")

message = arduino.read_message()

if message == "EMERGENCY_BUTTON:PRESSED":
    print("EMERGENCY BUTTON PRESSED!")


def open_camera(index=0):
    """Open the webcam.

    On Windows, the default backend (MSMF) is often slow to open, prints
    spurious warnings, or fails on some camera drivers. DirectShow
    (CAP_DSHOW) is the reliable choice there. Other platforms use the
    default backend as before.
    """
    if IS_WINDOWS:
        return cv2.VideoCapture(index, cv2.CAP_DSHOW)
    return cv2.VideoCapture(index)


# --------------------------------------------------------------------------
# Database
# --------------------------------------------------------------------------

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS people (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            relationship TEXT,
            encoding BLOB NOT NULL
        )
        """
    )
    return conn


def add_person(name, relationship, encoding):
    """Save a new person's face encoding. Returns the new row id."""
    conn = get_connection()
    try:
        cur = conn.execute(
            "INSERT INTO people (name, relationship, encoding) VALUES (?, ?, ?)",
            (name, relationship, pickle.dumps(encoding)),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def get_all_people():
    """Return list of dicts: {id, name, relationship, encoding (numpy array)}."""
    conn = get_connection()
    try:
        rows = conn.execute("SELECT id, name, relationship, encoding FROM people").fetchall()
    finally:
        conn.close()

    people = []
    for row_id, name, relationship, encoding_blob in rows:
        people.append(
            {
                "id": row_id,
                "name": name,
                "relationship": relationship,
                "encoding": pickle.loads(encoding_blob),
            }
        )
    return people


def delete_person(person_id):
    conn = get_connection()
    try:
        conn.execute("DELETE FROM people WHERE id = ?", (person_id,))
        conn.commit()
    finally:
        conn.close()


def get_person(person_id):
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT id, name, relationship, encoding FROM people WHERE id = ?", (person_id,)
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    row_id, name, relationship, encoding_blob = row
    return {"id": row_id, "name": name, "relationship": relationship, "encoding": pickle.loads(encoding_blob)}


def update_person(person_id, name=None, relationship=None, encoding=None):
    """Update only the fields given (None = leave unchanged). Returns False if no such person."""
    if get_person(person_id) is None:
        return False

    fields, values = [], []
    if name is not None:
        fields.append("name = ?")
        values.append(name)
    if relationship is not None:
        fields.append("relationship = ?")
        values.append(relationship)
    if encoding is not None:
        fields.append("encoding = ?")
        values.append(pickle.dumps(encoding))

    if not fields:
        return True

    values.append(person_id)
    conn = get_connection()
    try:
        conn.execute(f"UPDATE people SET {', '.join(fields)} WHERE id = ?", values)
        conn.commit()
    finally:
        conn.close()
    return True


# --------------------------------------------------------------------------
# Recognition engine
# --------------------------------------------------------------------------

class FaceEngine:
    def __init__(self, tolerance=DEFAULT_TOLERANCE):
        self.tolerance = tolerance
        self.known_ids = []
        self.known_names = []
        self.known_relationships = []
        self.known_encodings = []
        self.reload()

    def reload(self):
        """(Re)load all known face encodings from the database into RAM."""
        people = get_all_people()
        self.known_ids = [p["id"] for p in people]
        self.known_names = [p["name"] for p in people]
        self.known_relationships = [p["relationship"] for p in people]
        self.known_encodings = [p["encoding"] for p in people]

    @staticmethod
    def detect_and_encode(frame):
        """Return (face_locations, face_encodings) for a BGR frame (e.g. from cv2)."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb_frame)
        encodings = face_recognition.face_encodings(rgb_frame, locations)
        return locations, encodings

    def encode_single_face(self, frame):
        """Return the encoding for the first face found in a frame, or None."""
        _, encodings = self.detect_and_encode(frame)
        if not encodings:
            return None
        return encodings[0]

    def _nearest(self, encoding):
        """Return the closest known person regardless of tolerance, or None if the DB is empty:
        {"name", "relationship", "distance"} — distance is a raw face_distance value (lower = more alike).
        """
        if not self.known_encodings:
            return None
        distances = face_recognition.face_distance(self.known_encodings, encoding)
        best_index = int(np.argmin(distances))
        return {
            "name": self.known_names[best_index],
            "relationship": self.known_relationships[best_index],
            "distance": float(distances[best_index]),
        }

    def recognize_faces(self, frame):
        """Detect every face in the frame and return a list of results, one per face:
        {"location": (top, right, bottom, left), "name": str, "relationship": str|None,
         "distance": float|None, "nearest_name": str|None}
        Unmatched faces get name="UNKNOWN", relationship=None. "distance" and "nearest_name"
        are always the closest known person's distance/name (even above tolerance), which is
        useful for threshold tuning — they are None only when the database has no one registered.
        """
        locations, encodings = self.detect_and_encode(frame)
        results = []
        for location, encoding in zip(locations, encodings):
            nearest = self._nearest(encoding)
            if nearest and nearest["distance"] <= self.tolerance:
                results.append(
                    {
                        "location": location,
                        "name": nearest["name"],
                        "relationship": nearest["relationship"],
                        "distance": nearest["distance"],
                        "nearest_name": nearest["name"],
                        "encoding": encoding,
                    }
                )
            else:
                results.append(
                    {
                        "location": location,
                        "name": "UNKNOWN",
                        "relationship": None,
                        "distance": nearest["distance"] if nearest else None,
                        "nearest_name": nearest["name"] if nearest else None,
                        "encoding": encoding,
                    }
                )
        return results

    def recognize_face(self, frame):
        """Single-face convenience API for downstream integration (e.g. the voice module).

        Returns {"name": ..., "relationship": ...} for the first recognized face,
        or the string "UNKNOWN" if no face is found or it doesn't match anyone.
        """
        results = self.recognize_faces(frame)
        if not results or results[0]["name"] == "UNKNOWN":
            return "UNKNOWN"
        return {"name": results[0]["name"], "relationship": results[0]["relationship"]}

    def register_person(self, encoding, name, relationship):
        """Save a new person's encoding to the database and refresh the in-RAM cache."""
        person_id = add_person(name, relationship, encoding)
        self.reload()
        return person_id


# --------------------------------------------------------------------------
# Registration (CLI: register)
# --------------------------------------------------------------------------

CAMERA_WINDOW_TITLE = "Register Face - press SPACE to capture, ESC to cancel"


def capture_frame_from_webcam():
    """Open the webcam and let the user pick a frame with SPACE. Returns a BGR frame or None."""
    cap = open_camera()
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    captured = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Failed to read frame from webcam")

            cv2.imshow(CAMERA_WINDOW_TITLE, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 32:  # SPACE
                captured = frame
                break
            if key == 27:  # ESC
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return captured


def encoding_from_image_path(image_path):
    image = face_recognition.load_image_file(image_path)
    locations = face_recognition.face_locations(image)
    if not locations:
        return None
    encodings = face_recognition.face_encodings(image, locations)
    return encodings[0]


def register_person(name, relationship, image_path=None):
    """Compute a face encoding (from webcam or image file) and save the person. Returns True/False."""
    if image_path:
        encoding = encoding_from_image_path(image_path)
    else:
        frame = capture_frame_from_webcam()
        if frame is None:
            print("Capture cancelled.")
            return False
        encoding = FaceEngine.detect_and_encode(frame)[1]
        encoding = encoding[0] if encoding else None

    if encoding is None:
        print("No face detected. Registration failed.")
        return False

    engine = FaceEngine()
    engine.register_person(encoding, name, relationship)
    print("Saved Successfully")
    return True


# --------------------------------------------------------------------------
# Voice announcements + spoken registration (SAPI via pywin32 on Windows,
# pyttsx3 on macOS/Linux, for text-to-speech; SpeechRecognition for speech-to-text)
# --------------------------------------------------------------------------

if IS_WINDOWS:
    try:
        import win32com.client as _win32com_client
    except ImportError:
        _win32com_client = None
else:
    _win32com_client = None


class Voice:
    """Speaks announcements and listens for spoken answers.

    Text-to-speech:
      - Windows: talks directly to the native SAPI voice via pywin32
        (win32com.client), NOT pyttsx3. pyttsx3's Windows driver has a
        well-known bug where runAndWait() only actually produces audio
        the FIRST time it's called in a process -- every call after that
        runs with no error but goes silent. Calling .Speak() repeatedly on
        a raw SAPI.SpVoice COM object doesn't have this problem.
      - macOS/Linux: pyttsx3, one engine created at startup and reused for
        every speak() call.

    Speech-to-text: SpeechRecognition using Google's free web API (needs
    internet, which is fine for a demo with network available). Every mic/
    network call is wrapped so a hiccup degrades to a typed prompt instead
    of crashing or hanging the program.
    """

    def __init__(self, enabled=True):
        self.enabled = enabled
        self.engine = None
        self._sapi = None
        self.recognizer = sr.Recognizer()
        if not self.enabled:
            return

        if IS_WINDOWS and _win32com_client is not None:
            try:
                self._sapi = _win32com_client.Dispatch("SAPI.SpVoice")
            except Exception as exc:
                print(f"[voice unavailable: {exc}]")
                self.enabled = False
        else:
            try:
                self.engine = pyttsx3.init()
            except Exception as exc:
                print(f"[voice unavailable: {exc}]")
                self.enabled = False

    def speak(self, text):
        print(f"[voice] {text}")
        if not self.enabled:
            return
        try:
            if self._sapi is not None:
                self._sapi.Speak(text)
            elif self.engine is not None:
                self.engine.say(text)
                self.engine.runAndWait()
        except Exception as exc:
            print(f"[voice unavailable: {exc}]")
            self.enabled = False

    def listen(self, timeout=6, phrase_time_limit=6):
        """Listen on the default microphone for one phrase.

        Returns the recognized text, or None if nothing was heard, the mic
        isn't available, or the network/API call failed.
        """
        try:
            with sr.Microphone() as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                try:
                    audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
                except sr.WaitTimeoutError:
                    return None
        except OSError:
            print("[voice] No microphone detected.")
            return None

        try:
            return self.recognizer.recognize_google(audio).strip()
        except sr.UnknownValueError:
            return None  # heard something, couldn't understand it
        except sr.RequestError:
            print("[voice] Speech recognition service unavailable (check internet connection).")
            return None

    def ask(self, question, retries=2):
        """Speak `question`, listen for a spoken answer, retrying on silence/garble.

        Falls back to typed input if voice keeps failing, so the flow never
        hangs. Always returns a string (possibly empty).
        """
        self.speak(question)
        if not self.enabled:
            return input(f"  {question}\n  Type your answer: ").strip()

        for _ in range(retries):
            answer = self.listen()
            if answer:
                return answer
            self.speak("Sorry, I didn't catch that.")

        print(f"(Voice input unavailable) {question}")
        return input("  Type your answer instead: ").strip()

    def stop(self):
        if self.engine is not None:
            try:
                self.engine.stop()
            except Exception:
                pass


def announcement_for(name, relationship):
    if relationship:
        return f"This is {name}, your {relationship.lower()}"
    return f"This is {name}"


# --------------------------------------------------------------------------
# Navigation home (CLI: directions) — OpenRouteService (free) for real walking
# directions when available, falling back to an offline distance/direction
# estimate (pure math, no internet or API key needed) when it isn't.
# --------------------------------------------------------------------------

def _format_distance_meters(meters):
    if meters >= 1000:
        return f"{meters / 1000:.1f} km"
    return f"{meters:.0f} m"


def _format_duration_seconds(seconds):
    minutes = seconds / 60
    if minutes < 1:
        return "less than a minute"
    return f"{minutes:.0f} min{'s' if minutes >= 2 else ''}"


def _get_directions_via_ors(current_lat, current_lon, home_lat, home_lon):
    """Try OpenRouteService (free, no billing — https://openrouteservice.org).
    Returns {"distance", "duration", "steps", "source"}, or None if unavailable/failed.
    """
    api_key = os.environ.get(ORS_API_KEY_ENV)
    if not api_key:
        return None

    try:
        response = requests.get(
            ORS_DIRECTIONS_URL,
            params={
                "api_key": api_key,
                "start": f"{current_lon},{current_lat}",
                "end": f"{home_lon},{home_lat}",
            },
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        summary = data["routes"][0]["summary"]
        steps = [step["instruction"] for step in data["routes"][0]["segments"][0]["steps"]]
    except Exception as exc:
        print(f"[maps] OpenRouteService request failed: {exc}")
        return None

    return {
        "distance": _format_distance_meters(summary["distance"]),
        "duration": _format_duration_seconds(summary["duration"]),
        "steps": steps,
        "source": "openrouteservice",
    }


_COMPASS_DIRECTIONS = ["north", "northeast", "east", "southeast", "south", "southwest", "west", "northwest"]


def _haversine_distance_and_bearing(lat1, lon1, lat2, lon2):
    """Great-circle distance (meters) and initial compass bearing (as an 8-point
    direction string) between two points. Pure math — no internet, no API key.
    """
    earth_radius_m = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    distance_m = 2 * earth_radius_m * math.asin(math.sqrt(a))

    y = math.sin(d_lambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(d_lambda)
    bearing_deg = (math.degrees(math.atan2(y, x)) + 360) % 360
    direction = _COMPASS_DIRECTIONS[round(bearing_deg / 45) % 8]

    return distance_m, direction


def _get_directions_via_haversine(current_lat, current_lon, home_lat, home_lon):
    """Offline fallback when OpenRouteService is unset/unreachable: straight-line
    distance + rough compass direction only, no routed steps.
    """
    distance_m, direction = _haversine_distance_and_bearing(current_lat, current_lon, home_lat, home_lon)
    distance_str = _format_distance_meters(distance_m)
    return {
        "distance": distance_str,
        "duration": None,
        "steps": [f"Home is roughly {distance_str} to the {direction} in a straight line."],
        "source": "offline-estimate",
    }


def get_walking_directions(current_lat, current_lon, home_lat=None, home_lon=None):
    """Get directions home: tries OpenRouteService first (real routed walking
    directions), falls back to an offline straight-line estimate if that's
    unavailable. Returns {"distance": str, "duration": str|None, "steps": [str, ...],
    "source": str}, or None if home coordinates aren't configured at all.
    """
    if home_lat is None:
        home_lat = os.environ.get(HOME_LAT_ENV)
    if home_lon is None:
        home_lon = os.environ.get(HOME_LON_ENV)
    if home_lat is None or home_lon is None:
        print(f"[maps] Set the {HOME_LAT_ENV} and {HOME_LON_ENV} environment variables.")
        return None
    home_lat, home_lon = float(home_lat), float(home_lon)

    directions = _get_directions_via_ors(current_lat, current_lon, home_lat, home_lon)
    if directions is not None:
        return directions

    print("[maps] Falling back to an offline distance/direction estimate.")
    return _get_directions_via_haversine(current_lat, current_lon, home_lat, home_lon)


def announce_directions_home(current_lat, current_lon, voice_io=None):
    """Print (and speak, if a Voice instance is given) a summary of the way home."""
    directions = get_walking_directions(current_lat, current_lon)
    if directions is None:
        message = "Sorry, I can't get directions home right now."
    elif directions["duration"] is not None:
        first_step = directions["steps"][0] if directions["steps"] else ""
        message = f"Home is {directions['distance']} away, about {directions['duration']} on foot. {first_step}"
    else:
        message = directions["steps"][0]

    print(message)
    if voice_io is not None:
        voice_io.speak(message)
    return directions


# --------------------------------------------------------------------------
# Live webcam test (CLI: test)
# --------------------------------------------------------------------------

BOX_COLOR_KNOWN = (0, 200, 0)
BOX_COLOR_UNKNOWN = (0, 0, 255)


def label_for(result):
    distance_str = f' ({result["distance"]:.2f})' if result["distance"] is not None else ""
    if result["name"] == "UNKNOWN":
        nearest = f' [nearest: {result["nearest_name"]}]' if result["nearest_name"] else ""
        return f"UNKNOWN{distance_str}{nearest}"
    return f'{result["name"]} - {result["relationship"]}{distance_str}'


INSTRUCTION_BAR = "Unknown faces are asked about automatically   [M] Manage entries   [ESC] Quit"
WINDOW_TITLE = "Face Recognition"


def _register_unknown_by_voice(voice, engine, encoding):
    """Have a spoken conversation to register an UNKNOWN face's encoding.

    Returns True if someone was registered, False if skipped or if
    something went wrong (so the caller can just move on either way).
    """
    try:
        name = voice.ask("I don't recognize this person. What is their name?")
        if not name:
            voice.speak("Okay, skipping for now.")
            return False

        relationship = voice.ask(f"What is {name}'s relationship to you?")
        if not relationship:
            relationship = "Unknown"

        engine.register_person(encoding, name, relationship)
        voice.speak(f"Got it. I'll remember {name} as your {relationship}.")
        print(f"Saved Successfully: {name} - {relationship}")
        return True
    except Exception as exc:
        # Never let a voice/registration hiccup crash the live demo.
        print(f"[warning] Registration failed: {exc}")
        voice.speak("Something went wrong saving that. Let's continue.")
        return False


def run_live_test(tolerance=DEFAULT_TOLERANCE, voice=True):
    engine = FaceEngine(tolerance=tolerance)
    print(f"Loaded {len(engine.known_names)} known face(s) from database. Tolerance={tolerance}")
    print(INSTRUCTION_BAR)

    voice_io = Voice(enabled=voice)
    last_announced_name = None
    last_skip_time = 0.0

    cap = open_camera()
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            results = engine.recognize_faces(frame)

            known_this_frame = [r for r in results if r["name"] != "UNKNOWN"]
            if voice:
                if known_this_frame:
                    current_name = known_this_frame[0]["name"]
                    if current_name != last_announced_name:
                        voice_io.speak(announcement_for(current_name, known_this_frame[0]["relationship"]))
                        last_announced_name = current_name
                else:
                    last_announced_name = None

            for result in results:
                top, right, bottom, left = result["location"]
                color = BOX_COLOR_UNKNOWN if result["name"] == "UNKNOWN" else BOX_COLOR_KNOWN
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.rectangle(frame, (left, bottom - 25), (right, bottom), color, cv2.FILLED)
                cv2.putText(
                    frame,
                    label_for(result),
                    (left + 6, bottom - 6),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

            cv2.rectangle(frame, (0, 0), (frame.shape[1], 24), (30, 30, 30), cv2.FILLED)
            cv2.putText(frame, INSTRUCTION_BAR, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow(WINDOW_TITLE, frame)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                break
            elif key in (ord("m"), ord("M")):
                print("\n--- Entry management (recognition paused) ---")
                run_manage()
                engine.reload()
                last_announced_name = None
                print("--- Resuming live recognition ---\n")
                continue

            # Automatically start a spoken registration conversation the moment
            # an unknown face is seen (with a cooldown so a skip/failure doesn't
            # re-ask every single frame).
            now = time.time()
            unknown = next((r for r in results if r["name"] == "UNKNOWN"), None)
            if unknown is not None and (now - last_skip_time) > SKIP_COOLDOWN_SECONDS:
                cv2.imshow(WINDOW_TITLE, frame)
                cv2.waitKey(1)

                registered = _register_unknown_by_voice(voice_io, engine, unknown["encoding"])
                if registered:
                    last_announced_name = None
                else:
                    last_skip_time = time.time()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        voice_io.stop()


# --------------------------------------------------------------------------
# Interactive management menu (CLI: manage) — register many people in a loop,
# and list/edit/delete existing ones.
# --------------------------------------------------------------------------

def _prompt_new_encoding():
    """Ask the user for webcam or image-file capture, return an encoding or None."""
    source = input("Capture from (w)ebcam or (i)mage file? [w]: ").strip().lower() or "w"
    if source == "i":
        image_path = input("Path to image file: ").strip()
        encoding = encoding_from_image_path(image_path)
        if encoding is None:
            print("No face detected in that image.")
        return encoding

    frame = capture_frame_from_webcam()
    if frame is None:
        print("Capture cancelled.")
        return None
    encoding = FaceEngine.detect_and_encode(frame)[1]
    if not encoding:
        print("No face detected in the captured frame.")
        return None
    return encoding[0]


def _manage_register_loop():
    while True:
        name = input("Name: ").strip()
        relationship = input("Relationship: ").strip()
        encoding = _prompt_new_encoding()
        if encoding is not None:
            add_person(name, relationship, encoding)
            print(f"Saved Successfully: {name} - {relationship}")
        else:
            print("Skipped (no face captured).")

        again = input("Register another person? (y/n): ").strip().lower()
        if again != "y":
            break


def _manage_list():
    people = get_all_people()
    if not people:
        print("No one registered yet.")
        return
    for p in people:
        print(f'  [{p["id"]}] {p["name"]} - {p["relationship"]}')


def _manage_edit():
    _manage_list()
    people = get_all_people()
    if not people:
        return
    try:
        person_id = int(input("Enter the id to edit: ").strip())
    except ValueError:
        print("Invalid id.")
        return
    person = get_person(person_id)
    if person is None:
        print("No person with that id.")
        return

    print(f'Editing [{person["id"]}] {person["name"]} - {person["relationship"]}')
    new_name = input(f'New name (blank to keep "{person["name"]}"): ').strip()
    new_relationship = input(f'New relationship (blank to keep "{person["relationship"]}"): ').strip()
    rescan = input("Re-capture their face? (y/n): ").strip().lower() == "y"

    new_encoding = _prompt_new_encoding() if rescan else None
    update_person(
        person_id,
        name=new_name or None,
        relationship=new_relationship or None,
        encoding=new_encoding,
    )
    print("Updated Successfully")


def _manage_delete():
    _manage_list()
    people = get_all_people()
    if not people:
        return
    try:
        person_id = int(input("Enter the id to delete: ").strip())
    except ValueError:
        print("Invalid id.")
        return
    if get_person(person_id) is None:
        print("No person with that id.")
        return
    confirm = input("Are you sure? (y/n): ").strip().lower()
    if confirm == "y":
        delete_person(person_id)
        print("Deleted.")


def run_manage():
    menu = """
1. Register new person(s)
2. List registered people
3. Edit a person
4. Delete a person
5. Done (return)
"""
    while True:
        print(menu)
        choice = input("Choose an option: ").strip()
        if choice == "1":
            _manage_register_loop()
        elif choice == "2":
            _manage_list()
        elif choice == "3":
            _manage_edit()
        elif choice == "4":
            _manage_delete()
        elif choice == "5":
            break
        else:
            print("Invalid choice.")


# --------------------------------------------------------------------------
# CLI entrypoint
# --------------------------------------------------------------------------

def print_usage():
    print(__doc__)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        run_live_test()
        sys.exit(0)

    command = sys.argv[1]

    if command == "register":
        if len(sys.argv) < 4:
            print("Usage: python face_system.py register <name> <relationship> [image_path]")
            sys.exit(1)
        person_name = sys.argv[2]
        person_relationship = sys.argv[3]
        photo_path = sys.argv[4] if len(sys.argv) > 4 else None
        register_person(person_name, person_relationship, photo_path)

    elif command == "test":
        test_args = sys.argv[2:]
        voice_enabled = "--no-voice" not in test_args
        numeric_args = [a for a in test_args if a != "--no-voice"]
        test_tolerance = float(numeric_args[0]) if numeric_args else DEFAULT_TOLERANCE
        run_live_test(tolerance=test_tolerance, voice=voice_enabled)

    elif command == "manage":
        run_manage()

    elif command == "directions":
        if len(sys.argv) < 4:
            print("Usage: python face_system.py directions <current_lat> <current_lon>")
            sys.exit(1)
        announce_directions_home(float(sys.argv[2]), float(sys.argv[3]), voice_io=Voice(enabled=True))

    else:
        print_usage()
        sys.exit(1)
