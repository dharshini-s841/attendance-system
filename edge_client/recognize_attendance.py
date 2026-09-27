"""
Main capture loop: detects + recognizes faces, checks liveness via blink
detection, and marks attendance. Runs at the video's natural speed and
keeps the last-known face boxes visible even on frames that are skipped
for performance. Runs until you press 'q' to close it manually.
"""
import argparse
import time
from datetime import datetime

import cv2
import dlib
import face_recognition
import requests

import config
from utils.face_utils import load_known_faces, match_face
from utils.blink_detection import get_ear_from_landmarks, BlinkTracker
import local_db

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(config.SHAPE_PREDICTOR_PATH)


def shape_to_points(shape):
    return [(shape.part(i).x, shape.part(i).y) for i in range(68)]


def sync_to_cloud(student_name, period, classroom_id):
    try:
        resp = requests.post(
            f"{config.CLOUD_API_URL}/api/attendance",
            json={
                "student_name": student_name,
                "period": period,
                "classroom_id": classroom_id,
                "timestamp": datetime.now().isoformat(),
            },
            headers={"Authorization": f"Bearer {config.CLOUD_API_KEY}"},
            timeout=3,
        )
        if resp.status_code == 200:
            return True
        print(f"  Cloud rejected record ({resp.status_code}), queueing locally.")
    except requests.exceptions.RequestException:
        print("  Cloud unreachable, queueing locally.")
    local_db.queue_attendance(student_name, period, classroom_id)
    return False


def run_session(period_name, duration_seconds=None):
    known_names, known_embeddings = load_known_faces()
    if not known_embeddings:
        print("No enrolled students found. Run enroll.py first.")
        return

    local_db.init_db()
    marked_this_session = set()
    trackers = {}

    cap = cv2.VideoCapture("test_videos/classroom_small2.mp4")
    print("Video opened successfully:", cap.isOpened())

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if not source_fps or source_fps <= 0:
        source_fps = 25
    frame_delay = 1.0 / source_fps

    frame_count = 0
    start_time = time.time()
    last_boxes = []

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

        if frame_count % config.PROCESS_EVERY_N_FRAMES == 0:
            small_frame = cv2.resize(frame, (0, 0), fx=config.FRAME_RESIZE_SCALE, fy=config.FRAME_RESIZE_SCALE)
            rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(rgb_small)
            print(f"[frame {frame_count}] faces found: {len(face_locations)}")
            face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            dlib_faces = detector(gray)

            new_boxes = []
            for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
                scale = 1 / config.FRAME_RESIZE_SCALE
                top, right, bottom, left = int(top * scale), int(right * scale), int(bottom * scale), int(left * scale)

                name, distance = match_face(encoding, known_names, known_embeddings)
                label = name if name else "Unknown"
                color = (0, 255, 0) if name else (0, 0, 255)

                ear = None
                for df in dlib_faces:
                    if abs(df.left() - left) < 40 and abs(df.top() - top) < 40:
                        shape = predictor(gray, df)
                        points = shape_to_points(shape)
                        ear = get_ear_from_landmarks(points)
                        break

                status = ""
                if name:
                    if name not in trackers:
                        trackers[name] = BlinkTracker()
                    tracker = trackers[name]

                    if ear is not None:
                        tracker.update(ear)

                    if name in marked_this_session:
                        status = "Already marked"
                    elif tracker.has_blinked():
                        sync_to_cloud(name, period_name, config.CLASSROOM_ID)
                        marked_this_session.add(name)
                        status = "Attendance marked!"
                        color = (255, 200, 0)
                    else:
                        status = "Verifying liveness (blink)..."

                d_left, d_top = int(left * scale_x), int(top * scale_y)
                d_right, d_bottom = int(right * scale_x), int(bottom * scale_y)
                new_boxes.append((d_left, d_top, d_right, d_bottom, f"{label} {status}", color))

            last_boxes = new_boxes

        for (left, top, right, bottom, text, color) in last_boxes:
            cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)
            cv2.putText(display_frame, text, (left, max(top - 10, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        cv2.imshow("Attendance - Live (press q to close)", display_frame)

        elapsed = time.time() - loop_start
        wait_ms = max(int((frame_delay - elapsed) * 1000), 1)
        if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
            print("Closed manually.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Session ended. Marked: {sorted(marked_this_session)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--period", required=True, help="Period name, must match config.PERIODS")
    parser.add_argument("--duration", type=int, default=None, help="Optional session duration in seconds")
    args = parser.parse_args()
    run_session(args.period, args.duration)
