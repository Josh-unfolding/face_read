"""SQLite storage for known people and their face encodings."""

import pickle
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "face_database.db"


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
