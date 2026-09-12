"""Register a new person (name + relationship + face encoding) into the database.

Usage:
    python face_registration.py "Rahul" "Son"                 # capture from webcam
    python face_registration.py "Rahul" "Son" path/to/photo.jpg  # register from an image file
"""

import sys

import cv2
import face_recognition

from face_engine import FaceEngine

CAMERA_WINDOW_TITLE = "Register Face - press SPACE to capture, ESC to cancel"


def capture_frame_from_webcam():
    """Open the webcam and let the user pick a frame with SPACE. Returns a BGR frame or None."""
    cap = cv2.VideoCapture(0)
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
        engine = FaceEngine()
        encoding = engine.encode_single_face(frame)

    if encoding is None:
        print("No face detected. Registration failed.")
        return False

    engine = FaceEngine()
    engine.register_person(encoding, name, relationship)
    print("Saved Successfully")
    return True


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python face_registration.py <name> <relationship> [image_path]")
        sys.exit(1)

    person_name = sys.argv[1]
    person_relationship = sys.argv[2]
    photo_path = sys.argv[3] if len(sys.argv) > 3 else None

    register_person(person_name, person_relationship, photo_path)
