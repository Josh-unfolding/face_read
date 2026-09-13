"""Voice I/O helpers: speak text out loud, and listen for spoken answers.

Kept deliberately simple and reliable for a live demo:
  - Text-to-speech: pyttsx3 (offline, no network needed).
  - Speech-to-text: SpeechRecognition using Google's free web API
    (accurate, but needs internet -- fine since network will be
    available at the demo).

Every external call (mic access, network request, TTS engine) is wrapped
so a single failure -- no mic, no wifi for a second, TTS glitch -- degrades
to a text prompt instead of crashing the whole program mid-demo.
"""

import speech_recognition as sr
import pyttsx3

_recognizer = sr.Recognizer()

try:
    _tts_engine = pyttsx3.init()
except Exception:
    _tts_engine = None  # TTS unavailable; speak() will just print instead


def speak(text):
    """Say `text` out loud. Always prints too, and never raises."""
    print(f"[voice] {text}")
    if _tts_engine is None:
        return
    try:
        _tts_engine.say(text)
        _tts_engine.runAndWait()
    except Exception:
        # Don't let a TTS hiccup take down the demo.
        pass


def listen(timeout=6, phrase_time_limit=6):
    """Listen on the default microphone for one phrase.

    Returns the recognized text, or None if nothing was heard, the mic
    isn't available, or the network/API call failed.
    """
    try:
        with sr.Microphone() as source:
            _recognizer.adjust_for_ambient_noise(source, duration=0.5)
            try:
                audio = _recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            except sr.WaitTimeoutError:
                return None
    except OSError:
        # No microphone available at all.
        print("[voice] No microphone detected.")
        return None

    try:
        return _recognizer.recognize_google(audio).strip()
    except sr.UnknownValueError:
        print("[voice] Speech recognition service unavailable (check internet connection).")
        return None



def ask(question, retries=2):
    """Speak `question`, listen for a spoken answer, retrying on silence/garble.


    If voice input keeps failing (bad mic, no network, unclear audio),

    falls back to typed input so the flow never hangs. Always returns a
    string (possibly empty if the person gives nothing at all).

    """

    speak(question)
    for _ in range(retries):
        answer = listen()

        if answer:

            return answer
        speak("Sorry, I didn't catch that.")

    print(f"(Voice input unavailable) {question}")

    return input("  Type your answer instead: ").strip()
