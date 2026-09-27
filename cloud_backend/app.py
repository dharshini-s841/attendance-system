"""
Cloud backend: receives attendance records from one or more edge clients
(classrooms) over REST, stores them, and serves student/faculty dashboards.

Run locally:
    python app.py
Deploy: see README.md (Docker + Render/Railway/AWS instructions)

First-time setup (creates login accounts + subject/period mapping):
    python seed_data.py
"""
import os
import sqlite3
from datetime import datetime
from functools import wraps

from flask import Flask, request, jsonify, render_template, session, redirect, url_for, flash
from werkzeug.security import check_password_hash

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-change-me")

DB_PATH = os.environ.get("DB_PATH", "attendance.db")
API_KEY = os.environ.get("CLOUD_API_KEY", "change-me")  # must match edge_client/config.py

# A student is "eligible" (allowed to sit exams, per the pitch deck) at or
# above this overall attendance percentage.
ELIGIBILITY_THRESHOLD = 75.0


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_name TEXT NOT NULL,
            period TEXT NOT NULL,
            classroom_id TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
    """)
    # Login accounts for both students and faculty. `full_name` for a
    # student MUST match the name used in edge_client/enroll.py (and stored
    # in attendance.student_name), since that's how a login is matched back
    # to that student's face-recognition attendance records.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('student', 'faculty')),
            full_name TEXT NOT NULL,
            roll_number TEXT
        )
    """)
    # Maps each fixed class period (see edge_client/config.py PERIODS) to a
    # subject name and the faculty member who teaches it.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS subjects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject_name TEXT NOT NULL,
            period_name TEXT NOT NULL UNIQUE,
            faculty_username TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

def login_required(role=None):
    """Decorator: require a logged-in session, optionally of a specific role."""
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            if "user_id" not in session:
                flash("Please log in to continue.", "error")
                return redirect(url_for("login"))
            if role and session.get("role") != role:
                flash("You don't have access to that page.", "error")
                return redirect(url_for("login"))
            return f(*args, **kwargs)
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Auth routes
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    if "user_id" in session:
        if session.get("role") == "student":
            return redirect(url_for("student_dashboard"))
        return redirect(url_for("faculty_dashboard"))
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        role_selected = request.form.get("role", "student")

        conn = get_db()
        user = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        conn.close()

        if user and user["role"] == role_selected and check_password_hash(user["password_hash"], password):
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            session["role"] = user["role"]
            session["full_name"] = user["full_name"]
            session["roll_number"] = user["roll_number"]
            if user["role"] == "student":
                return redirect(url_for("student_dashboard"))
            return redirect(url_for("faculty_dashboard"))

        flash("Invalid username, password, or role.", "error")

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ---------------------------------------------------------------------------
# Student dashboard
# ---------------------------------------------------------------------------

@app.route("/student/dashboard")
@login_required(role="student")
def student_dashboard():
    conn = get_db()
    student_name = session["full_name"]
    today = datetime.now().strftime("%Y-%m-%d")

    daily = conn.execute(
        "SELECT * FROM attendance WHERE student_name = ? AND timestamp LIKE ? ORDER BY timestamp",
        (student_name, f"{today}%"),
    ).fetchall()

    subjects = conn.execute("SELECT * FROM subjects ORDER BY period_name").fetchall()
    subject_stats = []
    total_present, total_held = 0, 0

    for subj in subjects:
        period = subj["period_name"]
        sessions_held = conn.execute(
            "SELECT COUNT(DISTINCT substr(timestamp, 1, 10) || '|' || classroom_id) AS c "
            "FROM attendance WHERE period = ?",
            (period,),
        ).fetchone()["c"]
        present = conn.execute(
            "SELECT COUNT(*) AS c FROM attendance WHERE student_name = ? AND period = ?",
            (student_name, period),
        ).fetchone()["c"]
        pct = round((present / sessions_held) * 100, 1) if sessions_held else 0.0

        subject_stats.append({
            "subject": subj["subject_name"],
            "period": period,
            "present": present,
            "held": sessions_held,
            "percentage": pct,
        })
        total_present += present
        total_held += sessions_held

    overall_pct = round((total_present / total_held) * 100, 1) if total_held else 0.0
    eligible = overall_pct >= ELIGIBILITY_THRESHOLD
    conn.close()

    return render_template(
        "student_dashboard.html",
        student_name=student_name,
        roll_number=session.get("roll_number"),
        daily=daily,
        subject_stats=subject_stats,
        overall_pct=overall_pct,
        eligible=eligible,
        threshold=ELIGIBILITY_THRESHOLD,
    )


# ---------------------------------------------------------------------------
# Faculty dashboard
# ---------------------------------------------------------------------------

@app.route("/faculty/dashboard")
@login_required(role="faculty")
def faculty_dashboard():
    conn = get_db()
    username = session["username"]

    my_subjects = conn.execute(
        "SELECT * FROM subjects WHERE faculty_username = ? ORDER BY period_name",
        (username,),
    ).fetchall()

    selected_period = request.args.get("period") or (my_subjects[0]["period_name"] if my_subjects else None)

    class_records = []
    student_performance = []

    if selected_period:
        class_records = conn.execute(
            "SELECT * FROM attendance WHERE period = ? ORDER BY timestamp DESC LIMIT 200",
            (selected_period,),
        ).fetchall()

        sessions_held = conn.execute(
            "SELECT COUNT(DISTINCT substr(timestamp, 1, 10) || '|' || classroom_id) AS c "
            "FROM attendance WHERE period = ?",
            (selected_period,),
        ).fetchone()["c"]

        students = conn.execute(
            "SELECT * FROM users WHERE role = 'student' ORDER BY full_name"
        ).fetchall()

        for stu in students:
            present = conn.execute(
                "SELECT COUNT(*) AS c FROM attendance WHERE student_name = ? AND period = ?",
                (stu["full_name"], selected_period),
            ).fetchone()["c"]
            pct = round((present / sessions_held) * 100, 1) if sessions_held else 0.0
            student_performance.append({
                "name": stu["full_name"],
                "roll_number": stu["roll_number"],
                "present": present,
                "held": sessions_held,
                "percentage": pct,
                "status": "Good" if pct >= ELIGIBILITY_THRESHOLD else "At Risk",
            })
        student_performance.sort(key=lambda s: s["percentage"])

    class_summary = []
    for subj in my_subjects:
        sessions_held = conn.execute(
            "SELECT COUNT(DISTINCT substr(timestamp, 1, 10) || '|' || classroom_id) AS c "
            "FROM attendance WHERE period = ?",
            (subj["period_name"],),
        ).fetchone()["c"]
        total_marks = conn.execute(
            "SELECT COUNT(*) AS c FROM attendance WHERE period = ?",
            (subj["period_name"],),
        ).fetchone()["c"]
        class_summary.append({
            "subject": subj["subject_name"],
            "period": subj["period_name"],
            "sessions_held": sessions_held,
            "total_marks": total_marks,
        })

    conn.close()

    return render_template(
        "faculty_dashboard.html",
        faculty_name=session["full_name"],
        my_subjects=my_subjects,
        selected_period=selected_period,
        class_records=class_records,
        student_performance=student_performance,
        class_summary=class_summary,
        threshold=ELIGIBILITY_THRESHOLD,
    )


# ---------------------------------------------------------------------------
# Edge-client facing API (unchanged — do not require a browser login here)
# ---------------------------------------------------------------------------

def require_api_key(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        auth = request.headers.get("Authorization", "")
        token = auth.replace("Bearer ", "")
        if token != API_KEY:
            return jsonify({"error": "unauthorized"}), 401
        return f(*args, **kwargs)
    return wrapper


@app.route("/api/attendance", methods=["POST"])
@require_api_key
def post_attendance():
    data = request.get_json(force=True)
    required = ["student_name", "period", "classroom_id"]
    if not all(k in data for k in required):
        return jsonify({"error": "missing fields"}), 400

    timestamp = data.get("timestamp") or datetime.now().isoformat()

    conn = get_db()
    today = timestamp[:10]
    existing = conn.execute(
        "SELECT id FROM attendance WHERE student_name=? AND period=? AND classroom_id=? AND timestamp LIKE ?",
        (data["student_name"], data["period"], data["classroom_id"], f"{today}%"),
    ).fetchone()

    if existing:
        conn.close()
        return jsonify({"status": "already_marked"}), 200

    conn.execute(
        "INSERT INTO attendance (student_name, period, classroom_id, timestamp) VALUES (?, ?, ?, ?)",
        (data["student_name"], data["period"], data["classroom_id"], timestamp),
    )
    conn.commit()
    conn.close()
    return jsonify({"status": "marked"}), 200


@app.route("/api/attendance", methods=["GET"])
@require_api_key
def get_attendance():
    period = request.args.get("period")
    classroom_id = request.args.get("classroom_id")
    date = request.args.get("date")  # YYYY-MM-DD

    query = "SELECT * FROM attendance WHERE 1=1"
    params = []
    if period:
        query += " AND period=?"
        params.append(period)
    if classroom_id:
        query += " AND classroom_id=?"
        params.append(classroom_id)
    if date:
        query += " AND timestamp LIKE ?"
        params.append(f"{date}%")
    query += " ORDER BY timestamp DESC"

    conn = get_db()
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    init_db()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
