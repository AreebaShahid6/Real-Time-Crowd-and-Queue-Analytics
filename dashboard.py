"""
dashboard.py — the "finished product" renderer.

Builds a 1280x720 composite:

    +--------------------------------+--------------------+
    |                                | CROWD DENSITY      |
    |                                |   (heatmap)        |
    |        annotated feed          +--------------------+
    |   zone polygon + track boxes   | QUEUE STATISTICS   |
    |   [ QUEUE ZONE | COUNT: 7 ]    |   (numbers)        |
    |   [ EST. WAIT: ~8 MIN     ]    +--------------------+
    |                                | BIRD'S EYE VIEW    |
    |                                |   (top-down dots)  |
    +--------------------------------+--------------------+

Nothing here touches the tracking logic — it only consumes what
queue_analytics.py already computes.
"""

import cv2
import numpy as np

# ------------------------------------------------------------------ layout ---
W, H = 1280, 720
MAIN_W = 896
SIDE_X = MAIN_W
SIDE_W = W - MAIN_W
P1 = (0, 264)        # heatmap panel   y-range
P2 = (264, 512)      # stats panel
P3 = (512, 720)      # bird's eye

# ----------------------------------------------------------------- palette ---
BG = (16, 16, 18)
PANEL = (30, 30, 34)
EDGE = (60, 60, 66)
ACCENT = (90, 230, 110)     # green
CYAN = (255, 200, 0)
AMBER = (0, 190, 255)
WHITE = (240, 240, 240)
GREY = (150, 150, 155)
F = cv2.FONT_HERSHEY_SIMPLEX
FD = cv2.FONT_HERSHEY_DUPLEX


def put(img, text, org, scale, color, thick=1, font=F):
    cv2.putText(img, text, org, font, scale, color, thick, cv2.LINE_AA)


# ----------------------------------------------------------------- heatmap ---
class Heatmap:
    """Accumulates foot positions with exponential decay -> JET colormap."""

    def __init__(self, frame_shape, scale=8, decay=0.985, sigma=7):
        h, w = frame_shape[:2]
        self.scale = scale
        self.h, self.w = max(h // scale, 1), max(w // scale, 1)
        self.acc = np.zeros((self.h, self.w), np.float32)
        self.decay = decay
        self.sigma = sigma

    def update(self, points):
        self.acc *= self.decay
        for x, y in points:
            xi, yi = int(x / self.scale), int(y / self.scale)
            if 0 <= xi < self.w and 0 <= yi < self.h:
                self.acc[yi, xi] += 1.0

    def render(self, size):
        blur = cv2.GaussianBlur(self.acc, (0, 0), self.sigma)
        peak = float(blur.max())
        norm = blur / peak if peak > 1e-6 else blur
        u8 = np.clip(norm * 255.0, 0, 255).astype(np.uint8)
        col = cv2.applyColorMap(u8, cv2.COLORMAP_JET)
        return cv2.resize(col, size, interpolation=cv2.INTER_LINEAR)


# -------------------------------------------------------------- bird's eye ---
def _order_quad(pts):
    """tl, tr, br, bl."""
    pts = np.asarray(pts, np.float32)
    s = pts.sum(axis=1)
    d = (pts[:, 0] - pts[:, 1])
    return np.array([pts[np.argmin(s)], pts[np.argmax(d)],
                     pts[np.argmax(s)], pts[np.argmin(d)]], np.float32)


class BirdsEye:
    """Homography from the queue polygon to a flat top-down rectangle.

    If the polygon isn't a quad, we fall back to its minimum-area rectangle,
    so any zone shape still produces a usable warp.
    """

    CW, CH = 400, 300      # canonical canvas, resized to the panel afterwards

    def __init__(self, poly):
        poly = np.asarray(poly, np.float32)
        if len(poly) != 4:
            poly = cv2.boxPoints(cv2.minAreaRect(poly))
        src = _order_quad(poly)
        dst = np.array([[0, 0], [self.CW - 1, 0],
                        [self.CW - 1, self.CH - 1], [0, self.CH - 1]], np.float32)
        self.M = cv2.getPerspectiveTransform(src, dst)

    def render(self, items, size):
        """items: list of (x, y, bgr_color) in original frame coordinates."""
        img = np.full((self.CH, self.CW, 3), (14, 34, 16), np.uint8)
        for gx in range(0, self.CW, 40):
            cv2.line(img, (gx, 0), (gx, self.CH), (22, 52, 24), 1)
        for gy in range(0, self.CH, 40):
            cv2.line(img, (0, gy), (self.CW, gy), (22, 52, 24), 1)
        cv2.rectangle(img, (0, 0), (self.CW - 1, self.CH - 1), ACCENT, 2)

        if items:
            src = np.array([[x, y] for x, y, _ in items], np.float32).reshape(-1, 1, 2)
            proj = cv2.perspectiveTransform(src, self.M).reshape(-1, 2)
            for (px, py), (_, _, col) in zip(proj, items):
                if -40 < px < self.CW + 40 and -40 < py < self.CH + 40:
                    cv2.circle(img, (int(px), int(py)), 7, col, -1)
                    cv2.circle(img, (int(px), int(py)), 7, (255, 255, 255), 1)

        return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


# ------------------------------------------------------------------ panels ---
def _panel(canvas, y0, y1, title):
    x0, x1 = SIDE_X, W
    cv2.rectangle(canvas, (x0, y0), (x1, y1), PANEL, -1)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), EDGE, 1)
    put(canvas, title, (x0 + 12, y0 + 22), 0.46, ACCENT, 1, FD)
    return (x0 + 10, y0 + 34, x1 - 10, y1 - 10)   # content box


def _fit(img, w, h):
    ih, iw = img.shape[:2]
    s = min(w / iw, h / ih)
    nw, nh = max(int(iw * s), 1), max(int(ih * s), 1)
    out = np.full((h, w, 3), BG, np.uint8)
    y0, x0 = (h - nh) // 2, (w - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = cv2.resize(img, (nw, nh))
    return out


def annotate_feed(frame, qpoly, spoly, tracks, t, grace, qlen, wait_text):
    """Draw zones, vertices, boxes and the two banners onto the raw frame."""
    ov = frame.copy()
    cv2.fillPoly(ov, [qpoly], (40, 90, 40))
    cv2.fillPoly(ov, [spoly], (10, 70, 90))
    cv2.addWeighted(ov, 0.25, frame, 0.75, 0, frame)

    cv2.polylines(frame, [qpoly], True, ACCENT, 2, cv2.LINE_AA)
    cv2.polylines(frame, [spoly], True, AMBER, 2, cv2.LINE_AA)
    for p in qpoly:
        cv2.circle(frame, tuple(int(v) for v in p), 6, (255, 255, 255), -1)
        cv2.circle(frame, tuple(int(v) for v in p), 6, ACCENT, 2)

    for tid, tr in tracks.items():
        if tr.box is None or (t - tr.last_seen) > grace:
            continue
        col = {"in_queue": CYAN, "at_counter": AMBER}.get(tr.state, (120, 120, 120))
        x1, y1, x2, y2 = tr.box
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
        tag = f"{tid}"
        if tr.state == "at_counter":
            tag += f" {tr.counter_dwell():.0f}s"
        (tw, th), _ = cv2.getTextSize(tag, F, 0.42, 1)
        cv2.rectangle(frame, (x1, y1 - th - 7), (x1 + tw + 8, y1), col, -1)
        put(frame, tag, (x1 + 4, y1 - 5), 0.42, (20, 20, 20), 1)

    h, w = frame.shape[:2]
    _banner(frame, 20, h - 108, f"QUEUE ZONE  |  COUNT: {qlen}", ACCENT)
    _banner(frame, 20, h - 52, f"EST. WAIT: {wait_text.upper()}", AMBER)
    return frame


def _banner(img, x, y, text, color):
    (tw, th), _ = cv2.getTextSize(text, FD, 0.8, 2)
    cv2.rectangle(img, (x, y), (x + tw + 28, y + th + 22), color, -1)
    cv2.rectangle(img, (x, y), (x + tw + 28, y + th + 22), (255, 255, 255), 2)
    put(img, text, (x + 14, y + th + 12), 0.8, (20, 20, 20), 2, FD)


def build(feed, heat, bev, stats):
    """stats: dict with current/average/maximum/served/rate/wait/service_sec."""
    canvas = np.full((H, W, 3), BG, np.uint8)
    canvas[:, :MAIN_W] = _fit(feed, MAIN_W, H)

    # --- heatmap
    x0, y0, x1, y1 = _panel(canvas, *P1, "CROWD DENSITY HEATMAP")
    canvas[y0:y1, x0:x1] = heat.render((x1 - x0, y1 - y0))

    # --- statistics
    x0, y0, x1, y1 = _panel(canvas, *P2, "QUEUE STATISTICS")
    put(canvas, f"{stats['current']}", (x0 + 4, y0 + 52), 1.6, CYAN, 3, FD)
    put(canvas, "IN QUEUE NOW", (x0 + 4, y0 + 74), 0.42, GREY, 1)

    put(canvas, stats["wait"], (x0 + 132, y0 + 44), 0.85, AMBER, 2, FD)
    put(canvas, "ESTIMATED WAIT", (x0 + 132, y0 + 64), 0.4, GREY, 1)

    rows = [
        ("Average count", f"{stats['average']:.1f}"),
        ("Maximum count", f"{stats['maximum']}"),
        ("Total served", f"{stats['served']}"),
        ("Service rate", f"{stats['rate']:.1f}/min"),
        ("Avg service time", f"{stats['service_sec']:.0f}s"),
    ]
    yy = y0 + 104
    for k, v in rows:
        put(canvas, k, (x0 + 6, yy), 0.46, GREY, 1)
        put(canvas, v, (x1 - 6 - cv2.getTextSize(v, F, 0.46, 1)[0][0], yy),
            0.46, WHITE, 1)
        yy += 24

    # --- bird's eye
    x0, y0, x1, y1 = _panel(canvas, *P3, "BIRD'S EYE VIEW")
    canvas[y0:y1, x0:x1] = bev.render(stats["bev_items"], (x1 - x0, y1 - y0))
    put(canvas, f"Zone occupancy: {stats['current']}", (x0 + 6, y1 - 8),
        0.42, ACCENT, 1)

    return canvas
