"""
One-time setup: creates login accounts for students & faculty, and maps
each class period to a subject + the faculty who teaches it.

Run this once, after the database exists (start app.py at least once, or
this script will create the tables itself):

    cd cloud_backend
    python seed_data.py

Edit the STUDENTS / FACULTY / SUBJECTS lists below to match your real
class roster before running. Re-running is safe — existing rows are
skipped instead of duplicated.
"""
import os
import sqlite3

from werkzeug.security import generate_password_hash

from app import init_db

DB_PATH = os.environ.get("DB_PATH", "attendance.db")

# IMPORTANT: `full_name` for each student must exactly match the name used
# in edge_client/enroll.py when that student was enrolled (the same name
# that ends up in attendance.student_name). That's how a login is matched
# back to that student's face-recognition attendance history.
STUDENTS = [
    {"username": "bryer", "password": "student123", "full_name": "Bryer", "roll_number": "CLG-101"},
    {"username": "liam", "password": "student123", "full_name": "Liam", "roll_number": "CLG-102"},
    {"username": "sofia", "password": "student123", "full_name": "Sofia", "roll_number": "CLG-103"},
    {"username": "noah", "password": "student123", "full_name": "Noah", "roll_number": "CLG-104"},
    {"username": "maya", "password": "student123", "full_name": "Maya", "roll_number": "CLG-105"},
    {"username": "ethan", "password": "student123", "full_name": "Ethan", "roll_number": "CLG-106"},
]

FACULTY = [
    {"username": "dharshini", "password": "faculty123", "full_name": "Dharshini S"},
    {"username": "narmadha", "password": "faculty123", "full_name": "Narmadha K"},
]

# period_name must match a "name" entry in edge_client/config.py PERIODS
SUBJECTS = [
    {"period_name": "Period 1", "subject_name": "Data Structures", "faculty_username": "dharshini"},
    {"period_name": "Period 2", "subject_name": "Computer Networks", "faculty_username": "narmadha"},
    {"period_name": "Period 3", "subject_name": "Operating Systems", "faculty_username": "dharshini"},
]


def run():
    init_db()
    conn = sqlite3.connect(DB_PATH)

    for s in STUDENTS:
        try:
            conn.execute(
                "INSERT INTO users (username, password_hash, role, full_name, roll_number) "
                "VALUES (?, ?, 'student', ?, ?)",
                (s["username"], generate_password_hash(s["password"]), s["full_name"], s["roll_number"]),
            )
            print(f"  created student login : {s['username']} / {s['password']}")
        except sqlite3.IntegrityError:
            print(f"  skipped (exists)       : {s['username']}")

    for f in FACULTY:
        try:
            conn.execute(
                "INSERT INTO users (username, password_hash, role, full_name, roll_number) "
                "VALUES (?, ?, 'faculty', ?, NULL)",
                (f["username"], generate_password_hash(f["password"]), f["full_name"]),
            )
            print(f"  created faculty login  : {f['username']} / {f['password']}")
        except sqlite3.IntegrityError:
            print(f"  skipped (exists)       : {f['username']}")

    for subj in SUBJECTS:
        try:
            conn.execute(
                "INSERT INTO subjects (subject_name, period_name, faculty_username) VALUES (?, ?, ?)",
                (subj["subject_name"], subj["period_name"], subj["faculty_username"]),
            )
            print(f"  mapped {subj['period_name']:10s} -> {subj['subject_name']} ({subj['faculty_username']})")
        except sqlite3.IntegrityError:
            print(f"  skipped (exists)       : {subj['period_name']}")

    conn.commit()
    conn.close()
    print("\nDone. Run 'python app.py' and log in at http://localhost:5000/login")


if __name__ == "__main__":
    run()
