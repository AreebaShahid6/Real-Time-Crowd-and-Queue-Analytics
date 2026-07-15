"""
queue_analytics.py — live queue length, service rate and wait-time estimation
from one camera, rendered as a full dashboard.

    pip install -U ultralytics opencv-python

    python zone_picker.py --source queue.mp4 --out zones.json
    python queue_analytics.py --source queue.mp4 --zones zones.json \
        --model yolo26s.pt --save demo.mp4 --show

Tracking logic is unchanged from the simple version; only the output layer is
richer. Each track ID runs a state machine:

    outside -> in_queue -> at_counter -> (leaves / vanishes) -> SERVED

A service event needs all three of: was in the queue, dwelled at the counter
>= MIN_SERVICE_SEC, then left without re-entering the queue. That triple
condition is what rejects staff, walk-pasts and phone-call step-outs.
"""

import argparse
import csv
import json
import math
import time
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO

import dashboard as dash

# ---------------------------------------------------------------- tunables ---
ENTER_FRAMES = 5        # consecutive frames inside a zone before the state flips
EXIT_FRAMES = 8         # consecutive frames outside before it flips back
GRACE_SEC = 1.0         # still counted in the queue this long after last seen
LOST_SEC = 2.0          # after this long unseen, finalise and drop the track
MIN_SERVICE_SEC = 3.0   # minimum counter dwell to count as a real service
WINDOW = 8              # rolling window of service events used for the rate
BLEND_N = 5             # events needed before we fully trust the measurement


class Track:
    """Per-ID state machine."""

    __slots__ = ("state", "q_hits", "s_hits", "o_hits", "was_in_queue",
                 "counter_enter_t", "last_counter_t", "served", "last_seen",
                 "box", "anchor", "joined_t")

    def __init__(self, t):
        self.state = "outside"
        self.q_hits = self.s_hits = self.o_hits = 0
        self.was_in_queue = False
        self.counter_enter_t = None
        self.last_counter_t = None
        self.served = False
        self.last_seen = t
        self.joined_t = None
        self.box = None
        self.anchor = None

    def counter_dwell(self):
        if self.counter_enter_t is None or self.last_counter_t is None:
            return 0.0
        return self.last_counter_t - self.counter_enter_t


def inside(poly, pt):
    return cv2.pointPolygonTest(poly, (float(pt[0]), float(pt[1])), False) >= 0


def fmt_wait(sec):
    if sec is None:
        return "calibrating"
    if sec < 90:
        return f"~{int(round(sec))} sec"
    return f"~{int(math.ceil(sec / 60))} min"


def try_serve(tr):
    if tr.served or not tr.was_in_queue:
        return False
    if tr.counter_dwell() < MIN_SERVICE_SEC:
        return False
    tr.served = True
    return True


def live_qlen(tracks, t):
    return sum(1 for tr in tracks.values()
               if tr.state == "in_queue" and (t - tr.last_seen) <= GRACE_SEC)


def estimate_interval(events, prior):
    """Seconds between departures; the prior fades out as evidence arrives."""
    n = len(events) - 1
    if n < 1:
        return prior
    measured = (events[-1] - events[0]) / n
    wgt = min(n, BLEND_N) / BLEND_N
    return wgt * measured + (1 - wgt) * prior


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--zones", default="zones.json")
    ap.add_argument("--model", default="yolo26s.pt")
    ap.add_argument("--tracker", default="botsort.yaml",
                    help="botsort.yaml | bytetrack.yaml | tracktrack.yaml, "
                         "or a path to your own edited copy")
    ap.add_argument("--conf", type=float, default=0.22,
                    help="detection confidence; 0.20-0.30 is a good accuracy range")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--prior-service", type=float, default=45.0,
                    help="assumed seconds per customer before anything is measured")
    ap.add_argument("--save", default=None, help="output video path")
    ap.add_argument("--log", default=None, help="CSV of service events")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    z = json.load(open(args.zones))
    queue_poly = np.array(z["queue"], np.int32)
    service_poly = np.array(z["service"], np.int32)

    src = int(args.source) if args.source.isdigit() else args.source
    live = isinstance(src, int) or str(src).startswith(("rtsp", "http"))

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"cannot open source: {args.source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    writer = None
    if args.save:
        writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (dash.W, dash.H))

    log_f = log_w = None
    if args.log:
        log_f = open(args.log, "w", newline="")
        log_w = csv.writer(log_f)
        log_w.writerow(["t_sec", "track_id", "counter_dwell_sec", "queue_len"])

    model = YOLO(args.model)

    heat = None
    bev = dash.BirdsEye(queue_poly)

    tracks = {}
    service_events = deque(maxlen=WINDOW)
    total_served = 0
    wait_ema = None
    qlen_ema = 0.0
    qlen_sum = 0.0
    qlen_max = 0
    frame_idx = 0
    t = 0.0
    t0 = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if heat is None:
            heat = dash.Heatmap(frame.shape)
        t = (time.time() - t0) if live else frame_idx / fps

        # Do not draw an old box when a detection is temporarily missed.
        # State/queue counting still uses last_seen + GRACE_SEC, but the visible
        # rectangle is shown only for detections produced on the current frame.
        for tr in tracks.values():
            tr.box = None

        res = model.track(
            frame,
            persist=True,       # carry tracker state between calls
            classes=[0],        # person only
            conf=args.conf,
            imgsz=args.imgsz,
            tracker=args.tracker,
            max_det=300,
            verbose=False,
        )[0]

        boxes = res.boxes
        if boxes is not None and boxes.id is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            ids = boxes.id.int().cpu().numpy()
        else:
            xyxy, ids = np.empty((0, 4)), np.empty((0,), int)

        for box, tid in zip(xyxy, ids):
            tid = int(tid)
            x1, y1, x2, y2 = box
            anchor = ((x1 + x2) / 2.0, y2)          # bottom-centre = feet

            tr = tracks.get(tid)
            if tr is None:
                tr = tracks[tid] = Track(t)
            tr.last_seen = t
            tr.box = (int(x1), int(y1), int(x2), int(y2))
            tr.anchor = anchor

            in_s = inside(service_poly, anchor)
            in_q = (not in_s) and inside(queue_poly, anchor)

            # hysteresis: boundary jitter must not flip the state machine
            if in_s:
                tr.s_hits = min(tr.s_hits + 1, ENTER_FRAMES)
                tr.q_hits = max(tr.q_hits - 1, 0)
                tr.o_hits = 0
            elif in_q:
                tr.q_hits = min(tr.q_hits + 1, ENTER_FRAMES)
                tr.s_hits = max(tr.s_hits - 1, 0)
                tr.o_hits = 0
            else:
                tr.o_hits += 1
                tr.q_hits = max(tr.q_hits - 1, 0)
                tr.s_hits = max(tr.s_hits - 1, 0)

            prev = tr.state

            if tr.s_hits >= ENTER_FRAMES:
                if prev != "at_counter":
                    tr.state = "at_counter"
                    tr.counter_enter_t = t
                tr.last_counter_t = t

            elif tr.q_hits >= ENTER_FRAMES:
                if prev == "at_counter":
                    tr.counter_enter_t = tr.last_counter_t = None   # went back
                tr.state = "in_queue"
                tr.was_in_queue = True
                if tr.joined_t is None:
                    tr.joined_t = t

            elif tr.o_hits >= EXIT_FRAMES:
                if prev == "at_counter" and try_serve(tr):
                    total_served += 1
                    service_events.append(t)
                    if log_w:
                        log_w.writerow([f"{t:.2f}", tid,
                                        f"{tr.counter_dwell():.2f}",
                                        live_qlen(tracks, t)])
                tr.state = "outside"

        # finalise tracks that vanished (served, then walked out of frame)
        for tid in [i for i, tr in tracks.items() if t - tr.last_seen > LOST_SEC]:
            tr = tracks.pop(tid)
            if tr.state == "at_counter" and try_serve(tr):
                total_served += 1
                service_events.append(tr.last_seen)
                if log_w:
                    log_w.writerow([f"{tr.last_seen:.2f}", tid,
                                    f"{tr.counter_dwell():.2f}",
                                    live_qlen(tracks, t)])

        # ---------------------------------------------------------- metrics
        qlen = live_qlen(tracks, t)
        qlen_ema = float(qlen) if frame_idx == 0 else 0.85 * qlen_ema + 0.15 * qlen
        qlen_sum += qlen
        qlen_max = max(qlen_max, qlen)

        avg_interval = estimate_interval(service_events, args.prior_service)
        rate = 60.0 / avg_interval if avg_interval else 0.0
        wait_sec = qlen_ema * avg_interval if avg_interval else None
        if wait_sec is not None:
            wait_ema = wait_sec if wait_ema is None else 0.93 * wait_ema + 0.07 * wait_sec

        shown_qlen = int(round(qlen_ema))
        wait_text = fmt_wait(wait_ema if service_events else None)

        # ---------------------------------------------------------- render
        active = [tr for tr in tracks.values()
                  if tr.anchor is not None and (t - tr.last_seen) <= GRACE_SEC]
        heat.update([tr.anchor for tr in active])

        bev_items = [(tr.anchor[0], tr.anchor[1],
                      dash.AMBER if tr.state == "at_counter" else dash.CYAN)
                     for tr in active if tr.state in ("in_queue", "at_counter")]

        feed = dash.annotate_feed(frame, queue_poly, service_poly, tracks, t,
                                  GRACE_SEC, shown_qlen, wait_text)

        canvas = dash.build(feed, heat, bev, {
            "current": shown_qlen,
            "average": qlen_sum / (frame_idx + 1),
            "maximum": qlen_max,
            "served": total_served,
            "rate": rate,
            "service_sec": avg_interval,
            "wait": wait_text,
            "bev_items": bev_items,
        })

        if writer:
            writer.write(canvas)
        if args.show:
            cv2.imshow("queue analytics", canvas)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1

    cap.release()
    if writer:
        writer.release()
    if log_f:
        log_f.close()
    cv2.destroyAllWindows()
    print(f"done — {total_served} service events over {t:.0f}s")


if __name__ == "__main__":
    main()