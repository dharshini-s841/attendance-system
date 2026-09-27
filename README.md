# Facial Recognition Attendance System

Two parts, and this split is intentional:

- **`edge_client/`** — runs on the machine physically connected to the classroom camera. Does face detection, recognition, and blink-based liveness. This CANNOT run in the cloud, because it needs a real camera feed.
- **`cloud_backend/`** — a small REST API + dashboard that receives attendance records from the edge client and stores/displays them. THIS is what you deploy to the cloud.

The edge client posts each verified attendance mark to the cloud backend over HTTPS. If the cloud is unreachable, records queue locally and retry automatically.

---

## Part 1: Edge Client (runs locally, near the camera)

### Setup
```bash
cd edge_client
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Download the dlib landmark model and place it at `edge_client/models/shape_predictor_68_face_landmarks.dat`:
```bash
mkdir -p models
curl -L -o models/shape_predictor_68_face_landmarks.dat.bz2 \
  http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -d models/shape_predictor_68_face_landmarks.dat.bz2
```

### Configure
Edit `edge_client/config.py`:
- `PERIODS` — your fixed period start/end times
- `CLASSROOM_ID` — unique ID for this camera/room
- `CLOUD_API_URL` — set this once your backend is deployed (Part 2)
- `CLOUD_API_KEY` — must match the backend's key

### Enroll students
```bash
python enroll.py --name "Jane Doe" --webcam
# or with existing photos:
python enroll.py --name "Jane Doe" --images photo1.jpg photo2.jpg photo3.jpg
```

### Run
For manual testing of one session:
```bash
python recognize_attendance.py --period "Period 1" --duration 300
```

For the real fixed-schedule behavior (auto start/stop at configured times, continuous):
```bash
python scheduler.py
```

Test against a recorded/sample video instead of a webcam by changing
`cv2.VideoCapture(0)` to `cv2.VideoCapture("path/to/sample.mp4")` in
`recognize_attendance.py` — useful for your synthetic test videos.

---

## Part 2: Cloud Backend (deploy this)

### Run locally first to verify
```bash
cd cloud_backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export CLOUD_API_KEY="pick-a-real-secret"
python app.py
```
Visit `http://localhost:5000` for the dashboard. Test the API:
```bash
curl -X POST http://localhost:5000/api/attendance \
  -H "Authorization: Bearer pick-a-real-secret" \
  -H "Content-Type: application/json" \
  -d '{"student_name":"Jane Doe","period":"Period 1","classroom_id":"room-101"}'
```

### Deploying to the cloud

**Option A — Render / Railway (simplest, good for a student project)**
1. Push `cloud_backend/` to a GitHub repo
2. Create a new Web Service on Render or Railway, point it at the repo
3. Set environment variable `CLOUD_API_KEY` in the platform's dashboard
4. It will detect the `Dockerfile` automatically and deploy
5. You'll get a public URL like `https://your-app.onrender.com` — put this in `edge_client/config.py` as `CLOUD_API_URL`

**Option B — Docker + any VPS (DigitalOcean, AWS EC2, GCP Compute Engine)**
```bash
cd cloud_backend
docker build -t attendance-backend .
docker run -d -p 5000:5000 \
  -e CLOUD_API_KEY="pick-a-real-secret" \
  -v $(pwd)/data:/app/data \
  attendance-backend
```
Put a reverse proxy (Nginx/Caddy) in front for HTTPS, or use the platform's managed load balancer.

**Option C — AWS Elastic Beanstalk / Google Cloud Run**
Both accept the `Dockerfile` directly:
```bash
# Cloud Run example
gcloud run deploy attendance-backend \
  --source cloud_backend \
  --set-env-vars CLOUD_API_KEY=pick-a-real-secret \
  --allow-unauthenticated
```

### Note on the database
SQLite is fine for a prototype/demo but isn't ideal for concurrent cloud writes at scale.
For anything beyond a class demo, swap `sqlite3` calls in `cloud_backend/app.py` for
a managed Postgres instance (Render Postgres, AWS RDS, Supabase) — the query logic
stays nearly identical.

---

## Architecture Summary

```
[Classroom Camera] --> [edge_client: OpenCV + face_recognition + dlib]
                              |  detects face, matches embedding,
                              |  verifies blink (liveness)
                              v
                    POST /api/attendance (HTTPS)
                              v
                  [cloud_backend: Flask API + Dashboard]
                              v
                      [Database: SQLite/Postgres]
                              v
                   Viewable at your deployed dashboard URL
```

This is exactly the kind of split real deployed systems use: recognition happens
at the edge (low latency, no need to stream raw video to the cloud), and only
small JSON attendance records go over the network.
