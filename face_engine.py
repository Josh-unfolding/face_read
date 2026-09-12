"""Core recognition pipeline: detect -> encode -> compare -> return name/relationship.

Known encodings are loaded from the database once at startup and kept in RAM,
so recognition never touches SQLite on a per-frame basis.
"""

import cv2
import face_recognition
import numpy as np

import database

DEFAULT_TOLERANCE = 0.6


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
        people = database.get_all_people()
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

    def _best_match(self, encoding):
        if not self.known_encodings:
            return None
        distances = face_recognition.face_distance(self.known_encodings, encoding)
        best_index = int(np.argmin(distances))
        if distances[best_index] <= self.tolerance:
            return {
                "name": self.known_names[best_index],
                "relationship": self.known_relationships[best_index],
                "distance": float(distances[best_index]),
            }
        return None

    def recognize_faces(self, frame):
        """Detect every face in the frame and return a list of results, one per face:
        {"location": (top, right, bottom, left), "name": str, "relationship": str|None,
         "distance": float|None, "encoding": np.ndarray}
        Unmatched faces get name="UNKNOWN", relationship=None.

        The raw "encoding" is always included (even for known faces) so callers
        can register an unknown face on the spot without re-running detection.
        """
        locations, encodings = self.detect_and_encode(frame)
        results = []
        for location, encoding in zip(locations, encodings):
            match = self._best_match(encoding)
            if match:
                results.append(
                    {
                        "location": location,
                        "name": match["name"],
                        "relationship": match["relationship"],
                        "distance": match["distance"],
                        "encoding": encoding,
                    }
                )
            else:
                results.append(
                    {
                        "location": location,
                        "name": "UNKNOWN",
                        "relationship": None,
                        "distance": None,
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
        person_id = database.add_person(name, relationship, encoding)
        self.reload()
        return person_id