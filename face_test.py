"""Live webcam test for the recognition pipeline.

Draws a box + "Name - Relationship" (or "UNKNOWN") around every detected face.
Press ESC to quit.
"""

import sys

import cv2

from face_engine import FaceEngine, DEFAULT_TOLERANCE

BOX_COLOR_KNOWN = (0, 200, 0)
BOX_COLOR_UNKNOWN = (0, 0, 255)


def label_for(result):
    if result["name"] == "UNKNOWN":
        return "UNKNOWN"
    return f'{result["name"]} - {result["relationship"]}'


def run(tolerance=DEFAULT_TOLERANCE):
    engine = FaceEngine(tolerance=tolerance)
    print(f"Loaded {len(engine.known_names)} known face(s) from database. Tolerance={tolerance}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            results = engine.recognize_faces(frame)

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

            cv2.imshow("Face Recognition Test - ESC to quit", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    test_tolerance = float(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TOLERANCE
    run(tolerance=test_tolerance)
