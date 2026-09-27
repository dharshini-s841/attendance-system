"""
Enroll a student's face into the system, with an optional roll number.

Usage:
    python enroll.py --name "John Doe" --roll "R101" --images path1.jpg
    python enroll.py --name "John Doe" --roll "R101" --webcam
"""
import argparse
import json
import os
import cv2
from utils.face_utils import add_student_embedding

KNOWN_FACES_DIR = os.path.join(os.path.dirname(__file__), "data", "known_faces")
ROLL_NUMBERS_PATH = os.path.join(os.path.dirname(__file__), "data", "roll_numbers.json")


def save_roll_number(name, roll):
    data = {}
    if os.path.exists(ROLL_NUMBERS_PATH):
        with open(ROLL_NUMBERS_PATH, "r") as f:
            data = json.load(f)
    data[name] = roll
    os.makedirs(os.path.dirname(ROLL_NUMBERS_PATH), exist_ok=True)
    with open(ROLL_NUMBERS_PATH, "w") as f:
        json.dump(data, f, indent=2)


def capture_from_webcam(name, num_samples=5):
    os.makedirs(os.path.join(KNOWN_FACES_DIR, name), exist_ok=True)
    cap = cv2.VideoCapture(0)
    saved_paths = []
    print("Press SPACE to capture a sample, ESC to stop early.")

    count = 0
    while count < num_samples:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Enrollment - press SPACE to capture", frame)
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break
        elif key == 32:  # SPACE
            path = os.path.join(KNOWN_FACES_DIR, name, f"{count}.jpg")
            cv2.imwrite(path, frame)
            saved_paths.append(path)
            count += 1
            print(f"Captured sample {count}/{num_samples}")

    cap.release()
    cv2.destroyAllWindows()
    return saved_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="Student name / unique ID")
    parser.add_argument("--roll", default="", help="Roll / register number (optional)")
    parser.add_argument("--images", nargs="*", help="Paths to enrollment images")
    parser.add_argument("--webcam", action="store_true", help="Capture samples from webcam instead")
    args = parser.parse_args()

    if args.webcam:
        image_paths = capture_from_webcam(args.name)
    elif args.images:
        image_paths = args.images
    else:
        raise SystemExit("Provide --images or --webcam")

    if not image_paths:
        raise SystemExit("No images captured/provided.")

    add_student_embedding(args.name, image_paths)

    if args.roll:
        save_roll_number(args.name, args.roll)
        print(f"Saved roll number '{args.roll}' for '{args.name}'.")


if __name__ == "__main__":
    main()
