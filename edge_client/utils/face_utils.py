"""
Helpers for loading enrolled face embeddings and matching detected faces
against them.
"""
import pickle
import os
import numpy as np
import face_recognition

EMBEDDINGS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "embeddings.pkl")
MATCH_TOLERANCE = 0.5  # lower = stricter match. 0.6 is face_recognition's default.


def load_known_faces():
    """Returns (list_of_names, list_of_embeddings) or ([], []) if none saved yet."""
    if not os.path.exists(EMBEDDINGS_PATH):
        return [], []
    with open(EMBEDDINGS_PATH, "rb") as f:
        data = pickle.load(f)
    return data["names"], data["embeddings"]


def save_known_faces(names, embeddings):
    os.makedirs(os.path.dirname(EMBEDDINGS_PATH), exist_ok=True)
    with open(EMBEDDINGS_PATH, "wb") as f:
        pickle.dump({"names": names, "embeddings": embeddings}, f)


def add_student_embedding(name, image_paths):
    """
    Computes an averaged face embedding for a student from one or more
    enrollment images and appends it to the saved embeddings file.
    """
    names, embeddings = load_known_faces()
    student_encodings = []

    for path in image_paths:
        image = face_recognition.load_image_file(path)
        encs = face_recognition.face_encodings(image)
        if encs:
            student_encodings.append(encs[0])
        else:
            print(f"  Warning: no face found in {path}, skipping.")

    if not student_encodings:
        raise ValueError(f"No usable face found in any provided image for {name}.")

    avg_encoding = np.mean(student_encodings, axis=0)
    names.append(name)
    embeddings.append(avg_encoding)
    save_known_faces(names, embeddings)
    print(f"Enrolled '{name}' with {len(student_encodings)} sample(s).")


def match_face(face_encoding, known_names, known_embeddings):
    """
    Returns (name, distance) for the closest match under MATCH_TOLERANCE,
    or (None, None) if no match.
    """
    if not known_embeddings:
        return None, None

    distances = face_recognition.face_distance(known_embeddings, face_encoding)
    best_idx = int(np.argmin(distances))
    best_distance = distances[best_idx]

    if best_distance <= MATCH_TOLERANCE:
        return known_names[best_idx], best_distance
    return None, None
