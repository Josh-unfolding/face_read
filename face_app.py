"""Live webcam face recognition with on-the-fly registration.

Runs the webcam feed and draws a box + "Name - Relationship" around every
detected face, exactly like face_test.py used to. The new part: whenever an
UNKNOWN face is seen, the video pauses and the terminal asks for that
person's name and relationship. Answer it and they're saved to the database
immediately and recognized correctly from then on -- no separate
registration step needed.

Controls:
    ESC   - quit
    (terminal) type a name when prompted, or leave it blank to skip
             registering that face for a little while

Usage:
    python face_app.py            # default tolerance
    python face_app.py 0.55       # custom tolerance
"""

import sys
import time

import cv2

from face_engine import FaceEngine, DEFAULT_TOLERANCE

BOX_COLOR_KNOWN = (0, 200, 0)
BOX_COLOR_UNKNOWN = (0, 0, 255)
WINDOW_TITLE = "Face Recognition - ESC to quit"

# How long to wait before asking about the *same* still-unknown face again
# after the user chooses to skip. Prevents being asked every single frame.
SKIP_COOLDOWN_SECONDS = 8


def label_for(result):
    if result["name"] == "UNKNOWN":
        return "UNKNOWN"
    return f'{result["name"]} - {result["relationship"]}'


def draw_results(frame, results):
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


def prompt_and_register(engine, encoding):
    """Ask the terminal for a name/relationship and save the person.

    Returns True if someone was registered, False if the user skipped.
    """
    print("\nNew face detected!")
    name = input("  Name (leave blank to skip): ").strip()
    if not name:
        print("  Skipped.\n")
        return False

    relationship = input("  Relationship: ").strip()
    engine.register_person(encoding, name, relationship)
    print(f"  Saved {name} successfully!\n")
    return True


def run(tolerance=DEFAULT_TOLERANCE):
    engine = FaceEngine(tolerance=tolerance)
    print(f"Loaded {len(engine.known_names)} known face(s) from database. Tolerance={tolerance}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    last_skip_time = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            results = engine.recognize_faces(frame)
            draw_results(frame, results)

            cv2.imshow(WINDOW_TITLE, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

            unknown = next((r for r in results if r["name"] == "UNKNOWN"), None)
            if unknown is not None and (time.time() - last_skip_time) > SKIP_COOLDOWN_SECONDS:
                # Make sure the box is actually on screen before we block on input().
                cv2.imshow(WINDOW_TITLE, frame)
                cv2.waitKey(1)

                registered = prompt_and_register(engine, unknown["encoding"])
                if not registered:
                    last_skip_time = time.time()
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    test_tolerance = float(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TOLERANCE
    run(tolerance=test_tolerance)
