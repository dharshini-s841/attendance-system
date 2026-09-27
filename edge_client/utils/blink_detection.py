"""
Blink detection via Eye Aspect Ratio (EAR).
Uses dlib's 68-point facial landmark indices for the eyes:
Left eye: 36-41   Right eye: 42-47
"""
import numpy as np

LEFT_EYE_IDX = list(range(36, 42))
RIGHT_EYE_IDX = list(range(42, 48))

EAR_THRESHOLD = 0.22       # below this = eye considered closed
CONSEC_FRAMES_CLOSED = 2   # min frames closed to count as a real blink (avoid noise)


def _euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def eye_aspect_ratio(eye_points):
    """eye_points: list of 6 (x, y) landmark tuples for one eye."""
    a = _euclidean(eye_points[1], eye_points[5])
    b = _euclidean(eye_points[2], eye_points[4])
    c = _euclidean(eye_points[0], eye_points[3])
    if c == 0:
        return 0.0
    return (a + b) / (2.0 * c)


def get_ear_from_landmarks(landmarks):
    """
    landmarks: list/array of 68 (x, y) points from dlib shape predictor.
    Returns average EAR across both eyes.
    """
    left_eye = [landmarks[i] for i in LEFT_EYE_IDX]
    right_eye = [landmarks[i] for i in RIGHT_EYE_IDX]
    left_ear = eye_aspect_ratio(left_eye)
    right_ear = eye_aspect_ratio(right_eye)
    return (left_ear + right_ear) / 2.0


class BlinkTracker:
    """
    Tracks EAR over consecutive frames for one tracked face to detect a blink.
    Call update(ear) every frame; check has_blinked() to see if a full blink
    (closed -> open transition) has occurred since last reset().
    """

    def __init__(self, ear_threshold=EAR_THRESHOLD, consec_frames=CONSEC_FRAMES_CLOSED):
        self.ear_threshold = ear_threshold
        self.consec_frames = consec_frames
        self.closed_frame_count = 0
        self.blinked = False

    def update(self, ear):
        if ear < self.ear_threshold:
            self.closed_frame_count += 1
        else:
            if self.closed_frame_count >= self.consec_frames:
                self.blinked = True
            self.closed_frame_count = 0

    def has_blinked(self):
        return self.blinked

    def reset(self):
        self.closed_frame_count = 0
        self.blinked = False
