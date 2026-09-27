"""
Minimal test: just tries to open and read a video file, nothing else.
Run this directly to check if OpenCV can actually read your video.
"""
import cv2

VIDEO_PATH = "test_videos/classroom_small2.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)
print("isOpened:", cap.isOpened())
print("Frame count reported:", cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("FPS reported:", cap.get(cv2.CAP_PROP_FPS))
print("Width x Height:", cap.get(cv2.CAP_PROP_FRAME_WIDTH), "x", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

frames_read = 0
for i in range(10):
    ret, frame = cap.read()
    print(f"Attempt {i}: ret={ret}", "shape=" + str(frame.shape) if ret else "")
    if ret:
        frames_read += 1

print("Total successful reads out of 10 attempts:", frames_read)
cap.release()
