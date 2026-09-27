"""
Real-time face recognition attendance system with DUAL attendance marking:

  Pathway 1 (AUTO):  attendance is marked the moment a recognized face has
                      appeared for MIN_CONSECUTIVE_FRAMES in a row, no blink
                      required. Fast, but weaker to a printed photo held up.

  Pathway 2 (BLINK):  attendance is marked (independently) the moment a
                      genuine blink (EAR dip + recovery) is detected on a
                      recognized face. Slower, but confirms liveness.

Both pathways run at the same time and do not block each other. A student
can be marked AUTO first and then upgraded to "AUTO+BLINK" a moment later
when they blink, or the reverse.

Loads existing enrolled faces from data/embeddings.pkl and roll numbers
from data/roll_numbers.json (the same files your enroll.py already
writes to) -- no re-enrollment needed.

Usage:
    python realtime_attendance_dual.py --source test_videos/classroom_small2.mp4 --period "Period 1"
    python realtime_attendance_dual.py --source 0 --period "Period 1"     (webcam)
"""

import argparse
import csv
import json
import os
import pickle
import sqlite3
import time
from datetime import datetime

import cv2
import dlib
import face_recognition
import numpy as np

# ---------------------------------------------------------------------------
# SECTION 1: Paths and pinned configuration
#
# NOTE: PROCESS_EVERY_N_FRAMES must stay an integer and FRAME_RESIZE_SCALE
# must stay 1.0 for this project's 480p test video -- smaller resize values
# make faces too small for the detector to find at all. These were tuned
# and confirmed working; don't change them casually.
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDINGS_PATH = os.path.join(BASE_DIR, "data", "embeddings.pkl")
ROLL_NUMBERS_PATH = os.path.join(BASE_DIR, "data", "roll_numbers.json")
SHAPE_PREDICTOR_PATH = os.path.join(BASE_DIR, "models", "shape_predictor_68_face_landmarks.dat")
ATTENDANCE_DB_PATH = os.path.join(BASE_DIR, "data", "attendance_dual.db")

PROCESS_EVERY_N_FRAMES = 3
FRAME_RESIZE_SCALE = 1.0
MATCH_TOLERANCE = 0.5

MIN_CONSECUTIVE_FRAMES = 5       # frames in a row needed for AUTO marking
GRACE_PERIOD_FRAMES = 15         # frames a face can vanish (occlusion) before the
                                  # consecutive-frame counter resets to 0
EAR_THRESHOLD = 0.22
BLINK_CONSEC_FRAMES = 2

LEFT_EYE_IDX = list(range(36, 42))
RIGHT_EYE_IDX = list(range(42, 48))

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)


# ---------------------------------------------------------------------------
# SECTION 2: Dataset loading (known faces + roll numbers)
# ---------------------------------------------------------------------------
def load_known_faces():
    """Loads enrolled names + face embeddings from data/embeddings.pkl."""
    if not os.path.exists(EMBEDDINGS_PATH):
        raise FileNotFoundError(
            f"No enrolled faces found at {EMBEDDINGS_PATH}. Run enroll.py first."
        )
    with open(EMBEDDINGS_PATH, "rb") as f:
        data = pickle.load(f)
    return data["names"], data["embeddings"]


def load_roll_numbers():
    """Loads name -> roll number mapping from data/roll_numbers.json."""
    if not os.path.exists(ROLL_NUMBERS_PATH):
        return {}
    with open(ROLL_NUMBERS_PATH, "r") as f:
        return json.load(f)


def match_face(face_encoding, known_names, known_embeddings):
    """Returns (name, distance) for the closest match under MATCH_TOLERANCE."""
    if not known_embeddings:
        return None, None
    distances = face_recognition.face_distance(known_embeddings, face_encoding)
    best_idx = int(np.argmin(distances))
    best_distance = distances[best_idx]
    if best_distance <= MATCH_TOLERANCE:
        return known_names[best_idx], best_distance
    return None, None


# ---------------------------------------------------------------------------
# SECTION 3: Eye Aspect Ratio (EAR) + blink detection
# ---------------------------------------------------------------------------
def shape_to_points(shape):
    return [(shape.part(i).x, shape.part(i).y) for i in range(68)]


def _euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def eye_aspect_ratio(eye_points):
    a = _euclidean(eye_points[1], eye_points[5])
    b = _euclidean(eye_points[2], eye_points[4])
    c = _euclidean(eye_points[0], eye_points[3])
    return (a + b) / (2.0 * c) if c != 0 else 0.0


def get_ear_from_landmarks(landmarks):
    left_eye = [landmarks[i] for i in LEFT_EYE_IDX]
    right_eye = [landmarks[i] for i in RIGHT_EYE_IDX]
    return (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0


class BlinkTracker:
    """Detects one full closed->open blink cycle from a stream of EAR values."""

    def __init__(self):
        self.closed_frame_count = 0
        self.blinked_ever = False  # sticky: True forever once a blink is seen

    def update(self, ear):
        if ear is None:
            return
        if ear < EAR_THRESHOLD:
            self.closed_frame_count += 1
        else:
            if self.closed_frame_count >= BLINK_CONSEC_FRAMES:
                self.blinked_ever = True
            self.closed_frame_count = 0


# ---------------------------------------------------------------------------
# SECTION 4: Lightweight anti-spoof heuristic (texture check)
#
# True depth-based liveness needs stereo/IR hardware we don't have here.
# As a low-cost heuristic, a printed photo or a screen held up to the camera
# tends to have a flatter, blurrier texture than a real face at the same
# distance. Laplacian variance is a common blur proxy: lower variance often
# means a flatter/blurrier surface. This is a supplementary signal only --
# it is logged alongside each mark, not used to block attendance, since it
# can misfire under poor lighting or camera focus issues.
# ---------------------------------------------------------------------------
def texture_liveness_score(gray_frame, box):
    top, right, bottom, left = box
    top, left = max(top, 0), max(left, 0)
    roi = gray_frame[top:bottom, left:right]
    if roi.size == 0:
        return 0.0
    return float(cv2.Laplacian(roi, cv2.CV_64F).var())


# ---------------------------------------------------------------------------
# SECTION 5: Unified attendance storage (SQLite + CSV)
# ---------------------------------------------------------------------------
def init_attendance_db():
    conn = sqlite3.connect(ATTENDANCE_DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS attendance_log (
            name TEXT PRIMARY KEY,
            roll_number TEXT,
            first_detection_time TEXT,
            auto_marked_time TEXT,
            blink_verified_time TEXT,
            method_used TEXT
        )
    """)
    conn.commit()
    return conn


def upsert_attendance_row(conn, name, roll, state):
    method_parts = []
    if state["auto_marked_time"]:
        method_parts.append("AUTO")
    if state["blink_verified_time"]:
        method_parts.append("BLINK")
    method_used = "+".join(method_parts) if method_parts else ""

    conn.execute("""
        INSERT INTO attendance_log (name, roll_number, first_detection_time, auto_marked_time, blink_verified_time, method_used)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(name) DO UPDATE SET
            roll_number=excluded.roll_number,
            auto_marked_time=excluded.auto_marked_time,
            blink_verified_time=excluded.blink_verified_time,
            method_used=excluded.method_used
    """, (name, roll, state["first_detection_time"], state["auto_marked_time"],
          state["blink_verified_time"], method_used))
    conn.commit()


def export_csv(student_states, roll_numbers):
    """Rewrites today's CSV with the current full attendance table."""
    filename = os.path.join(BASE_DIR, f"attendance_{datetime.now().strftime('%Y-%m-%d')}.csv")
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Roll Number", "First Detection Time",
                          "Auto-Marked Time", "Blink-Verified Time", "Method Used"])
        for name, state in student_states.items():
            method_parts = []
            if state["auto_marked_time"]:
                method_parts.append("AUTO")
            if state["blink_verified_time"]:
                method_parts.append("BLINK")
            writer.writerow([
                name,
                roll_numbers.get(name, ""),
                state["first_detection_time"] or "",
                state["auto_marked_time"] or "",
                state["blink_verified_time"] or "",
                "+".join(method_parts) if method_parts else "PENDING",
            ])
    return filename


# ---------------------------------------------------------------------------
# SECTION 6: Main video processing loop
# ---------------------------------------------------------------------------
def run(source, period_name, duration_seconds=None):
    known_names, known_embeddings = load_known_faces()
    roll_numbers = load_roll_numbers()
    total_enrolled = len(set(known_names))

    db_conn = init_attendance_db()

    # per-student tracking state, keyed by recognized name
    student_states = {}   # name -> dict(consecutive_frames, last_seen_frame, times..., blink_tracker)

    # video source can be an integer webcam index (passed as string "0") or a file path
    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
    print("Video opened successfully:", cap.isOpened())

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if not source_fps or source_fps <= 0:
        source_fps = 25
    frame_delay = 1.0 / source_fps

    frame_count = 0
    start_time = time.time()
    last_boxes = []  # persisted so drawing continues on skipped frames

    print(f"Starting session for {period_name}. Press 'q' to stop.")

    while True:
        loop_start = time.time()

        if duration_seconds and (time.time() - start_time) > duration_seconds:
            print("Period time window ended.")
            break

        ret, frame = cap.read()
        if not ret:
            print("Video ended.")
            break

        frame_count += 1
        display_frame = cv2.resize(frame, (960, 540))
        scale_x = 960 / frame.shape[1]
        scale_y = 540 / frame.shape[0]

        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            small_frame = cv2.resize(frame, (0, 0), fx=FRAME_RESIZE_SCALE, fy=FRAME_RESIZE_SCALE)
            rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            face_locations = face_recognition.face_locations(rgb_small)
            face_encodings = face_recognition.face_encodings(rgb_small, face_locations)
            dlib_faces = detector(gray)

            new_boxes = []
            seen_this_frame = set()

            for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
                scale = 1 / FRAME_RESIZE_SCALE
                top, right, bottom, left = (int(top * scale), int(right * scale),
                                             int(bottom * scale), int(left * scale))

                name, distance = match_face(encoding, known_names, known_embeddings)

                # --- find matching dlib landmarks for EAR + liveness heuristic ---
                ear = None
                for df in dlib_faces:
                    if abs(df.left() - left) < 40 and abs(df.top() - top) < 40:
                        shape = predictor(gray, df)
                        points = shape_to_points(shape)
                        ear = get_ear_from_landmarks(points)
                        break

                liveness_score = texture_liveness_score(gray, (top, right, bottom, left))

                if name:
                    seen_this_frame.add(name)
                    now_str = datetime.now().strftime("%H:%M:%S")

                    # --- initialize or update this student's tracking state ---
                    if name not in student_states:
                        student_states[name] = {
                            "consecutive_frames": 0,
                            "last_seen_frame": frame_count,
                            "first_detection_time": now_str,
                            "auto_marked_time": None,
                            "blink_verified_time": None,
                            "blink_tracker": BlinkTracker(),
                        }

                    state = student_states[name]

                    # Grace period: only reset the consecutive-frame streak if the
                    # gap since last seen is bigger than GRACE_PERIOD_FRAMES
                    # (handles brief occlusion/turning away without losing progress)
                    gap = frame_count - state["last_seen_frame"]
                    if gap > GRACE_PERIOD_FRAMES:
                        state["consecutive_frames"] = 1
                    else:
                        state["consecutive_frames"] += 1
                    state["last_seen_frame"] = frame_count

                    # --- Pathway 1: AUTO marking ---
                    if (state["consecutive_frames"] >= MIN_CONSECUTIVE_FRAMES
                            and state["auto_marked_time"] is None):
                        state["auto_marked_time"] = now_str
                        upsert_attendance_row(db_conn, name, roll_numbers.get(name, ""), state)
                        export_csv(student_states, roll_numbers)
                        print(f"[AUTO-MARKED] {name} at {now_str} (liveness score: {liveness_score:.1f})")

                    # --- Pathway 2: BLINK verification (independent of Pathway 1) ---
                    state["blink_tracker"].update(ear)
                    if state["blink_tracker"].blinked_ever and state["blink_verified_time"] is None:
                        state["blink_verified_time"] = now_str
                        upsert_attendance_row(db_conn, name, roll_numbers.get(name, ""), state)
                        export_csv(student_states, roll_numbers)
                        print(f"[BLINK-VERIFIED] {name} at {now_str}")

                    # --- build the on-screen label ---
                    roll = roll_numbers.get(name, "")
                    if state["auto_marked_time"] and state["blink_verified_time"]:
                        status = "MARKED (AUTO+BLINK)"
                    elif state["auto_marked_time"]:
                        status = "MARKED (AUTO)"
                    elif state["blink_verified_time"]:
                        status = "MARKED (BLINK)"
                    else:
                        status = f"PENDING ({state['consecutive_frames']}/{MIN_CONSECUTIVE_FRAMES} frames)"

                    blink_indicator = "Blink: YES" if state["blink_tracker"].blinked_ever else "Blink: no"
                    color = (255, 200, 0) if (state["auto_marked_time"] or state["blink_verified_time"]) else (0, 255, 0)

                    lines = [f"{name} ({roll})", status, blink_indicator]
                else:
                    color = (0, 0, 255)
                    lines = ["Unknown"]

                d_left, d_top = int(left * scale_x), int(top * scale_y)
                d_right, d_bottom = int(right * scale_x), int(bottom * scale_y)
                new_boxes.append((d_left, d_top, d_right, d_bottom, lines, color))

            last_boxes = new_boxes

        # --- draw boxes + stacked text labels on every displayed frame ---
        for (left, top, right, bottom, lines, color) in last_boxes:
            cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)
            # stack each line above the previous one, moving upward from the box top,
            # so multiple lines never overlap each other
            line_height = 18
            start_y = max(top - (len(lines) * line_height) - 4, line_height)
            for i, text in enumerate(lines):
                y = start_y + i * line_height
                cv2.putText(display_frame, text, (left, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        # --- live attendance counter overlay ---
        marked_count = sum(1 for s in student_states.values()
                            if s["auto_marked_time"] or s["blink_verified_time"])
        cv2.putText(display_frame, f"Attendance: {marked_count}/{total_enrolled} marked",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(display_frame, f"Period: {period_name}   Time: {datetime.now().strftime('%H:%M:%S')}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("Attendance - Dual Pathway (press q to close)", display_frame)

        elapsed = time.time() - loop_start
        wait_ms = max(int((frame_delay - elapsed) * 1000), 1)
        if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
            print("Closed manually.")
            break

    cap.release()
    cv2.destroyAllWindows()

    csv_path = export_csv(student_states, roll_numbers)
    db_conn.close()

    print("\n=== Session summary ===")
    for name, state in student_states.items():
        method = []
        if state["auto_marked_time"]:
            method.append("AUTO")
        if state["blink_verified_time"]:
            method.append("BLINK")
        print(f"  {name} ({roll_numbers.get(name, '')}): {'+'.join(method) if method else 'never marked'}")
    print(f"CSV exported to: {csv_path}")
    print(f"SQLite log at: {ATTENDANCE_DB_PATH}")


# ---------------------------------------------------------------------------
# SECTION 7: CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dual-pathway face recognition attendance system")
    parser.add_argument("--source", default="0",
                         help="Webcam index (e.g. 0) or path to a video file")
    parser.add_argument("--period", required=True, help="Period name, e.g. 'Period 1'")
    parser.add_argument("--duration", type=int, default=None,
                         help="Optional session duration in seconds (omit to run until 'q' or video end)")
    args = parser.parse_args()

    run(args.source, args.period, args.duration)
