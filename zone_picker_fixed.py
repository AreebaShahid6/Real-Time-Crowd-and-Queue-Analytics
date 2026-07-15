"""zone_picker_fixed.py

Robust click-to-define polygon zones for queue_analytics.py.

Controls
--------
Left click  : add a point to the current zone
Right click : undo the last point in the current zone
U           : undo the last point
N / Enter   : finish the current zone and move to the next zone
R           : reset the current zone
S           : save zones.json and quit
Q / Esc     : quit without saving

Example
-------
python zone_picker_fixed.py --source queue.mp4 --frame 30 --out zones.json
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Union

import cv2
import numpy as np

ZONES = ("queue", "service")
COLORS = {
    "queue": (255, 200, 0),     # cyan/yellow in BGR
    "service": (0, 220, 120),   # green in BGR
}
HEADER_H = 54


def get_screen_size() -> Tuple[int, int]:
    """Return usable screen dimensions, with a safe fallback."""
    if os.name == "nt":
        try:
            # Prevent Windows display scaling from confusing OpenCV coordinates.
            try:
                ctypes.windll.shcore.SetProcessDpiAwareness(1)
            except Exception:
                ctypes.windll.user32.SetProcessDPIAware()
            return (
                int(ctypes.windll.user32.GetSystemMetrics(0)),
                int(ctypes.windll.user32.GetSystemMetrics(1)),
            )
        except Exception:
            pass
    return 1280, 720


def open_source(source: str, frame_number: int) -> np.ndarray:
    """Open a video/webcam and return the requested frame."""
    src: Union[int, str] = int(source) if source.isdigit() else source

    if isinstance(src, str) and not Path(src).exists():
        raise SystemExit(f"Video file not found: {src}")

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {source}")

    frame = None
    ok = False

    # Seeking is quick for normal video files. Webcam sources simply read once.
    if isinstance(src, int):
        ok, frame = cap.read()
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(frame_number, 0))
        ok, frame = cap.read()

        # Some codecs cannot seek accurately. Fall back to sequential reading.
        if not ok or frame is None:
            cap.release()
            cap = cv2.VideoCapture(src)
            for _ in range(max(frame_number, 0) + 1):
                ok, frame = cap.read()
                if not ok:
                    break

    cap.release()

    if not ok or frame is None:
        raise SystemExit(
            f"Could not read frame {frame_number} from source: {source}"
        )
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="video path or webcam index")
    parser.add_argument("--frame", type=int, default=0, help="frame number to draw on")
    parser.add_argument("--out", default="zones.json", help="output JSON file")
    parser.add_argument(
        "--max-width",
        type=int,
        default=0,
        help="optional maximum display width; 0 means automatic",
    )
    parser.add_argument(
        "--max-height",
        type=int,
        default=0,
        help="optional maximum display height; 0 means automatic",
    )
    args = parser.parse_args()

    frame = open_source(args.source, args.frame)
    original_h, original_w = frame.shape[:2]

    screen_w, screen_h = get_screen_size()
    max_w = args.max_width or max(640, int(screen_w * 0.88))
    max_h = args.max_height or max(360, int(screen_h * 0.74))

    scale = min(max_w / original_w, max_h / original_h, 1.0)
    display_w = max(1, int(round(original_w * scale)))
    display_h = max(1, int(round(original_h * scale)))
    scale_x = display_w / original_w
    scale_y = display_h / original_h

    polygons: Dict[str, List[List[int]]] = {zone: [] for zone in ZONES}
    current_zone_index = [0]

    print(f"Source frame: {original_w}x{original_h}")
    print(f"Picker display: {display_w}x{display_h} (scale {scale:.3f})")
    print("Click inside the VIDEO area, below the instruction bar.")

    def display_to_original(x: int, y: int) -> Tuple[int, int] | None:
        image_y = y - HEADER_H
        if x < 0 or x >= display_w or image_y < 0 or image_y >= display_h:
            return None
        ox = int(round(x / scale_x))
        oy = int(round(image_y / scale_y))
        ox = max(0, min(original_w - 1, ox))
        oy = max(0, min(original_h - 1, oy))
        return ox, oy

    def original_to_display(point: List[int]) -> Tuple[int, int]:
        return (
            int(round(point[0] * scale_x)),
            HEADER_H + int(round(point[1] * scale_y)),
        )

    def on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        del flags, param
        if current_zone_index[0] >= len(ZONES):
            return

        zone = ZONES[current_zone_index[0]]

        if event == cv2.EVENT_LBUTTONDOWN:
            original_point = display_to_original(x, y)
            if original_point is None:
                print("Click ignored: click below the top instruction bar.")
                return

            polygons[zone].append([original_point[0], original_point[1]])
            print(
                f"{zone} point {len(polygons[zone])}: "
                f"display=({x}, {y - HEADER_H}), "
                f"original={original_point}"
            )

        elif event == cv2.EVENT_RBUTTONDOWN and polygons[zone]:
            removed = polygons[zone].pop()
            print(f"Removed {zone} point: {tuple(removed)}")

    window_name = "zone picker - click below the instruction bar"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, on_mouse)

    saved = False

    while True:
        resized = cv2.resize(
            frame, (display_w, display_h), interpolation=cv2.INTER_AREA
        )
        canvas = np.zeros((display_h + HEADER_H, display_w, 3), dtype=np.uint8)
        canvas[HEADER_H:, :] = resized

        for zone in ZONES:
            points = polygons[zone]
            color = COLORS[zone]
            display_points = [original_to_display(point) for point in points]

            for point in display_points:
                cv2.circle(canvas, point, 5, color, -1, cv2.LINE_AA)
                cv2.circle(canvas, point, 8, (255, 255, 255), 1, cv2.LINE_AA)

            if len(display_points) >= 2:
                cv2.polylines(
                    canvas,
                    [np.asarray(display_points, dtype=np.int32)],
                    False,
                    color,
                    2,
                    cv2.LINE_AA,
                )
            if len(display_points) >= 3:
                cv2.line(
                    canvas,
                    display_points[-1],
                    display_points[0],
                    color,
                    1,
                    cv2.LINE_AA,
                )

            if display_points:
                label_x = min(display_points[0][0] + 8, display_w - 110)
                label_y = max(display_points[0][1] - 8, HEADER_H + 20)
                cv2.putText(
                    canvas,
                    zone.upper(),
                    (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                    cv2.LINE_AA,
                )

        if current_zone_index[0] < len(ZONES):
            current_zone = ZONES[current_zone_index[0]]
            current_count = len(polygons[current_zone])
            title = f"DRAWING: {current_zone.upper()} | points: {current_count}"
        else:
            title = "BOTH ZONES COMPLETE - PRESS S TO SAVE"

        cv2.putText(
            canvas,
            title,
            (12, 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "Left click:add  Right click/U:undo  N/Enter:next  R:reset  S:save  Q/Esc:quit",
            (12, 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(20) & 0xFF

        # Handle the user closing the window with the X button.
        try:
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break

        if key in (ord("q"), 27):
            break

        if current_zone_index[0] < len(ZONES):
            zone = ZONES[current_zone_index[0]]

            if key == ord("u") and polygons[zone]:
                removed = polygons[zone].pop()
                print(f"Removed {zone} point: {tuple(removed)}")

            elif key == ord("r"):
                polygons[zone] = []
                print(f"Reset {zone} zone")

            elif key in (ord("n"), 13, 10):
                if len(polygons[zone]) < 3:
                    print(
                        f"{zone} needs at least 3 points; "
                        f"currently has {len(polygons[zone])}."
                    )
                else:
                    print(f"Finished {zone} with {len(polygons[zone])} points.")
                    current_zone_index[0] += 1
                    if current_zone_index[0] < len(ZONES):
                        print(f"Now draw the {ZONES[current_zone_index[0]]} zone.")

        if key == ord("s"):
            missing = [zone for zone in ZONES if len(polygons[zone]) < 3]
            if missing:
                details = ", ".join(
                    f"{zone}={len(polygons[zone])} point(s)" for zone in missing
                )
                print(f"Cannot save yet: {details}. Each zone needs at least 3 points.")
            else:
                output_path = Path(args.out)
                with output_path.open("w", encoding="utf-8") as output_file:
                    json.dump(polygons, output_file, indent=2)
                print(f"Saved zones -> {output_path.resolve()}")
                print(json.dumps(polygons, indent=2))
                saved = True
                break

    cv2.destroyAllWindows()

    if not saved:
        print("Zone picker closed without saving.")


if __name__ == "__main__":
    main()