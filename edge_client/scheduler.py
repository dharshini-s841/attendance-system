"""
Runs continuously on the edge device. Triggers recognize_attendance.run_session()
at each configured period's start time, stopping it at end time. Also retries
syncing any locally-queued attendance records to the cloud every few minutes.

Usage:
    python scheduler.py
"""
import threading
from datetime import datetime, timedelta
import time

from apscheduler.schedulers.background import BackgroundScheduler

import config
import local_db
from recognize_attendance import run_session, sync_to_cloud

session_thread = None


def start_period_session(period):
    global session_thread
    start = datetime.strptime(period["start"], "%H:%M")
    end = datetime.strptime(period["end"], "%H:%M")
    duration_seconds = int((end - start).total_seconds())

    print(f"[{datetime.now()}] Starting {period['name']} for {duration_seconds}s")
    session_thread = threading.Thread(
        target=run_session, args=(period["name"], duration_seconds), daemon=True
    )
    session_thread.start()


def retry_pending_sync():
    rows = local_db.get_unsynced()
    if not rows:
        return
    print(f"[{datetime.now()}] Retrying sync for {len(rows)} queued record(s)...")
    for record_id, student_name, period, classroom_id, timestamp in rows:
        ok = sync_to_cloud(student_name, period, classroom_id)
        if ok:
            local_db.mark_synced(record_id)


def main():
    local_db.init_db()
    scheduler = BackgroundScheduler()

    for period in config.PERIODS:
        hour, minute = map(int, period["start"].split(":"))
        scheduler.add_job(
            start_period_session, "cron", hour=hour, minute=minute, args=[period],
            id=f"start-{period['name']}", replace_existing=True,
        )

    # Retry offline-queued attendance records every 5 minutes
    scheduler.add_job(retry_pending_sync, "interval", minutes=5, id="sync-retry")

    scheduler.start()
    print("Scheduler running. Configured periods:")
    for p in config.PERIODS:
        print(f"  {p['name']}: {p['start']} - {p['end']}")

    try:
        while True:
            time.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()


if __name__ == "__main__":
    main()
