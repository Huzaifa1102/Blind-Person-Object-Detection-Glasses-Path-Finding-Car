"""
PyQt5 Simulation - Intelligent Follower Robot
This module creates a full 2D simulated environment that uses the REAL
YOLOv8 + DeepSORT person tracking pipeline from the laptop webcam, and
visualizes a virtual differential-drive robot following the detected
person while simulating ultrasonic sensors and the production state
machine (IDLE, FOLLOWING, OBSTACLE_DETECTED, SCANNING, PATH_FOUND).
NOTE:
- This is a desktop-only simulation. It does not talk to real hardware.
- It reuses the same logical states and audio phrases as the laptop
  follower test in ProjectSoftware.py but adds a full GUI.
"""
import sys
import math
import time
import threading
import csv
from enum import Enum
from collections import deque
import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import pyttsx3
from PyQt5.QtCore import (
    Qt,
    QThread,
    pyqtSignal,
    QTimer,
    QRectF,
    QPointF,
)
from PyQt5.QtGui import (
    QPainter,
    QColor,
    QPen,
    QBrush,
    QFont,
    QPixmap,
    QImage,
)
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QPushButton,
    QTextEdit,
    QGroupBox,
    QFormLayout,
    QProgressBar,
    QSizePolicy,
)
# ---------------------------------------------------------------------------
# Shared enums and simple data types
# ---------------------------------------------------------------------------
class SystemState(Enum):
    FOLLOWING = "following"
    OBSTACLE_DETECTED = "obstacle_detected"
    SCANNING = "scanning"
    PATH_FOUND = "path_found"
    IDLE = "idle"
class RobotCommand(Enum):
    STOP = "stop"
    FOLLOW = "follow"
    RETURN_TO_FOLLOW = "return_to_follow"
# ---------------------------------------------------------------------------
# Audio manager (thread-safe pyttsx3)
# ---------------------------------------------------------------------------
class AudioManager:
    def _init_(self, enable_audio: bool = True):
        self.enable_audio = enable_audio
        if enable_audio:
            try:
                self.engine = pyttsx3.init()
                self.engine.setProperty("rate", 150)
            except Exception as e:
                print(f"[AUDIO] Initialization failed: {e}")
                self.enable_audio = False
    def speak(self, text: str):
        if not self.enable_audio:
            print(f"[AUDIO DISABLED] {text}")
            return
        print(f"[AUDIO] {text}")
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            print(f"[AUDIO ERROR] {e}")
    def shutdown(self):
        if self.enable_audio:
            try:
                self.engine.stop()
            except Exception:
                pass
# ---------------------------------------------------------------------------
# Vision processing thread (YOLOv8 + DeepSORT)
# ---------------------------------------------------------------------------
class VisionThread(QThread):
    """
    Runs YOLOv8 + DeepSORT on webcam frames in the background.
    Signals
    -------
    frame_ready(frame_bgr, tracks, vision_stats)
        frame_bgr : np.ndarray (BGR image)
        tracks : list of DeepSort track objects
        vision_stats: dict with YOLO / tracker FPS and detection info
    """
    frame_ready = pyqtSignal(object, object, object)
    def _init_(self, camera_id: int = 0, parent=None):
        super()._init_(parent)
        self.camera_id = camera_id
        self._running = False
    def run(self):
        self._running = True
        cap = cv2.VideoCapture(self.camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not cap.isOpened():
            print("[VISION] Could not open webcam")
            return
        print("[VISION] Loading YOLOv8n model...")
        model = YOLO("yolov8n.pt")
        print("[VISION] Initializing DeepSORT...")
        tracker = DeepSort(
            max_age=30,
            n_init=3,
            nms_max_overlap=1.0,
            max_cosine_distance=0.3,
            nn_budget=None,
        )
        frame_count = 0
        last_fps_time = time.time()
        yolo_fps = 0.0
        tracker_fps = 0.0
        while self._running:
            ret, frame = cap.read()
            if not ret:
                print("[VISION] Failed to read frame")
                break
            t0 = time.time()
            results = model(frame, classes=[0], conf=0.5, verbose=False)
            t1 = time.time()
            detections = []
            person_count = 0
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    detections.append(([x1, y1, x2 - x1, y2 - y1], conf, "person"))
                    person_count += 1
            t2 = time.time()
            tracks = tracker.update_tracks(detections, frame=frame)
            t3 = time.time()
            frame_count += 1
            if frame_count % 10 == 0:
                total_dt = time.time() - last_fps_time
                if total_dt > 0:
                    yolo_fps = 10.0 / (t1 - t0 + 1e-6)
                    tracker_fps = 10.0 / (t3 - t2 + 1e-6)
                last_fps_time = time.time()
            vision_stats = {
                "persons": person_count,
                "yolo_fps": float(yolo_fps),
                "tracker_fps": float(tracker_fps),
            }
            self.frame_ready.emit(frame, tracks, vision_stats)
        cap.release()
        print("[VISION] Stopped")
    def stop(self):
        self._running = False
# ---------------------------------------------------------------------------
# Simulation engine (physics, sensors, state machine)
# ---------------------------------------------------------------------------
class SimulationEngine:
    """
    Core 2D simulation engine. Units are centimeters.
    """
    ROBOT_WIDTH = 30.0 # distance between wheels (cm)
    MAX_SPEED = 50.0 # cm/s
    MAX_TURN_RATE = 90.0 # deg/s
    def _init_(self):
        # Robot pose
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_angle = 0.0 # radians
        # Robot motion state
        self.current_speed = 0.0
        self.target_speed = 0.0
        self.current_turn_rate = 0.0 # deg/s
        self.target_turn_rate = 0.0
        # Accel / decel
        self.accel = self.MAX_SPEED / 0.5 # reach max in 0.5 s
        self.decel = self.MAX_SPEED / 0.3
        # Person
        self.person_visible = False
        self.person_x = 100.0
        self.person_y = 0.0
        self.person_heading = 0.0 # radians, direction the person is "facing"
        self.person_id = None
        self.person_conf = 0.0
        self.last_person_time = 0.0
        self.last_person_move_time = time.time()
        # Sensors
        self.sensor_distances = {"left": 200.0, "center": 200.0, "right": 200.0}
        self.sensor_max_range = 200.0
        # Obstacles: list of dicts {x, y, w, h}
        self.obstacles = []
        # State machine
        self.state = SystemState.IDLE
        self.state_changed_time = time.time()
        self.last_follow_command_time = 0.0
        self.last_state_change_reason = ""
        # Statistics
        self.total_distance = 0.0
        self.obstacles_avoided = 0
        self.following_time = 0.0
        self.target_switches = 0
        self.commands_sent = 0
        self.state_changes = 0
        # Scan / path planning
        self.scan_result = None # {"direction": "left/right/center", "clearance": value}
        self.scan_started_time = None # when actual LiDAR-like scan begins (after moving in front)
        self.scan_start_angle = None
        self.max_clearance = 0.0
        self.best_angle = 0.0
        # Path instruction / post-scan behaviour
        self.path_instruction_time = None
        self.path_person_ref = None # (x, y) of person when instruction given
        # Person-obstacle interaction
        self.person_obstacle_threshold = 50.0 # cm (can be tuned)
        self.last_blocking_obstacle = None
        # Misc
        self._last_update_time = time.time()
        # Person-centric obstacle info
        self.last_person_obstacle_distance = None
        # Create some initial random obstacles
        self._init_random_obstacles()
    # ---------------------- obstacle management ----------------------
    def _init_random_obstacles(self, count: int = 5):
        rng = np.random.default_rng()
        for _ in range(count):
            x = float(rng.integers(-300, 300))
            y = float(rng.integers(-300, 300))
            w = float(rng.integers(40, 120))
            h = float(rng.integers(40, 120))
            self.obstacles.append({"x": x, "y": y, "w": w, "h": h})
    def add_random_obstacle(self):
        rng = np.random.default_rng()
        x = float(rng.integers(-300, 300))
        y = float(rng.integers(-300, 300))
        w = float(rng.integers(40, 120))
        h = float(rng.integers(40, 120))
        self.obstacles.append({"x": x, "y": y, "w": w, "h": h})
    def clear_obstacles(self):
        self.obstacles.clear()
    # ---------------------- person mapping ---------------------------
    def update_person_from_bbox(
        self,
        bbox_ltrb,
        frame_width: int,
        person_id: int,
        confidence: float,
    ):
        """
        Map YOLO/DeepSORT detection to 2D position relative to robot.
        bbox_ltrb: (x1, y1, x2, y2)
        """
        if self.person_id is not None and self.person_id != person_id:
            self.target_switches += 1
        self.person_id = person_id
        self.person_conf = confidence
        self.person_visible = True
        self.last_person_time = time.time()
        x1, y1, x2, y2 = bbox_ltrb
        center_x = (x1 + x2) / 2.0
        width = (x2 - x1)
        # Normalized center and width
        center_x_norm = (center_x - frame_width / 2.0) / (frame_width / 2.0)
        width_norm = max(width / frame_width, 1e-3)
        # Distance / angle mapping as per specification
        distance_to_person = 300.0 / width_norm # cm
        angle_to_person_deg = center_x_norm * 45.0
        angle_to_person_rad = math.radians(angle_to_person_deg)
        # Place in world coordinates
        angle_global = self.robot_angle + angle_to_person_rad
        self.person_x = self.robot_x + distance_to_person * math.cos(angle_global)
        self.person_y = self.robot_y + distance_to_person * math.sin(angle_global)
        self.person_heading = angle_global
    def move_person(self, dx_cm: float, dy_cm: float):
        """
        Manually move the virtual person in world coordinates (cm).
        Used when controlling the person with keyboard arrows.
        """
        self.person_x += dx_cm
        self.person_y += dy_cm
        # Update facing direction based on movement vector
        if abs(dx_cm) > 1e-3 or abs(dy_cm) > 1e-3:
            self.person_heading = math.atan2(dy_cm, dx_cm)
        if self.person_id is None:
            self.person_id = 1
        self.person_conf = 1.0
        self.person_visible = True
        now = time.time()
        self.last_person_time = now
        self.last_person_move_time = now
    def place_robot_behind_person(self, distance_cm: float = 100.0):
        """
        Place robot behind the user (conceptually) for FOLLOWING mode.
        We interpret "behind" as extending the current line between
        person and robot so that the robot remains on the same side
        but at a controlled distance.
        """
        if not self.person_visible:
            return
        dx = self.robot_x - self.person_x
        dy = self.robot_y - self.person_y
        dist = math.hypot(dx, dy)
        if dist < 1e-3:
            # Fallback: use opposite of robot heading
            ux = -math.cos(self.robot_angle)
            uy = -math.sin(self.robot_angle)
        else:
            ux = dx / dist
            uy = dy / dist
        self.robot_x = self.person_x + ux * distance_cm
        self.robot_y = self.person_y + uy * distance_cm
        self.current_speed = 0.0
        self.target_speed = 0.0
    def behind_target_position(self, distance_cm: float = 100.0):
        """
        Compute where the robot should be when sitting behind the person
        at the given distance, without actually moving it yet.
        """
        if not self.person_visible:
            return self.robot_x, self.robot_y
        dx = self.robot_x - self.person_x
        dy = self.robot_y - self.person_y
        dist = math.hypot(dx, dy)
        if dist < 1e-3:
            ux = -math.cos(self.robot_angle)
            uy = -math.sin(self.robot_angle)
        else:
            ux = dx / dist
            uy = dy / dist
        bx = self.person_x + ux * distance_cm
        by = self.person_y + uy * distance_cm
        return bx, by
    def place_robot_in_front_of_person(self, distance_cm: float = 50.0):
        """
        Place robot in front of the user for SCANNING / PATH planning.
        We interpret "in front" as the opposite side of the current
        person-robot line, at a small offset ahead of the user.
        """
        if not self.person_visible:
            return
        dx = self.robot_x - self.person_x
        dy = self.robot_y - self.person_y
        dist = math.hypot(dx, dy)
        if dist < 1e-3:
            # Fallback: use robot heading
            ux = math.cos(self.robot_angle)
            uy = math.sin(self.robot_angle)
        else:
            ux = dx / dist
            uy = dy / dist
        # Front is opposite of current vector from person to robot
        self.robot_x = self.person_x - ux * distance_cm
        self.robot_y = self.person_y - uy * distance_cm
        self.current_speed = 0.0
        self.target_speed = 0.0
    # ---------------------- sensor simulation ------------------------
    def _point_in_obstacle(self, x: float, y: float) -> bool:
        for ob in self.obstacles:
            if ob["x"] <= x <= ob["x"] + ob["w"] and ob["y"] <= y <= ob["y"] + ob["h"]:
                return True
        return False
    def _cast_single_ray(self, angle_rad: float) -> float:
        """
        Cast one ray from robot and return first hit distance (cm).
        """
        step = 1.0
        for dist in np.arange(0.0, self.sensor_max_range + step, step):
            tx = self.robot_x + dist * math.cos(angle_rad)
            ty = self.robot_y + dist * math.sin(angle_rad)
            if self._point_in_obstacle(tx, ty):
                return float(dist)
        return float(self.sensor_max_range)
    def _cast_ray_from(self, ox: float, oy: float, angle_rad: float, max_range: float):
        """
        Generic ray-cast from arbitrary origin (ox, oy) with given angle.
        """
        step = 1.0
        for dist in np.arange(0.0, max_range + step, step):
            tx = ox + dist * math.cos(angle_rad)
            ty = oy + dist * math.sin(angle_rad)
            if self._point_in_obstacle(tx, ty):
                return float(dist)
        return float(max_range)
    def update_sensors(self):
        # Sensor rays: -30, 0, +30 degrees from robot heading
        angles_deg = {"left": -30.0, "center": 0.0, "right": 30.0}
        for key, ang_deg in angles_deg.items():
            ang_rad = self.robot_angle + math.radians(ang_deg)
            self.sensor_distances[key] = self._cast_single_ray(ang_rad)
    def person_sensor_distances(self, max_range: float = 200.0) -> dict:
        """
        Compute sensor-like distances from the person's position, to
        simulate the microwave sensors on the smart glasses.
        """
        if not self.person_visible:
            return {"left": max_range, "center": max_range, "right": max_range}
        angles_deg = {"left": -30.0, "center": 0.0, "right": 30.0}
        result = {}
        base = self.person_heading
        for key, ang_deg in angles_deg.items():
            ang_rad = base + math.radians(ang_deg)
            result[key] = self._cast_ray_from(
                self.person_x, self.person_y, ang_rad, max_range
            )
        return result
    def _distance_person_to_obstacle(self, ob: dict) -> float:
        """
        Compute shortest Euclidean distance from person to an axis-aligned
        rectangular obstacle. If inside, distance is 0.
        """
        if not self.person_visible:
            return float("inf")
        px, py = self.person_x, self.person_y
        x1, y1 = ob["x"], ob["y"]
        x2, y2 = x1 + ob["w"], y1 + ob["h"]
        dx = 0.0
        if px < x1:
            dx = x1 - px
        elif px > x2:
            dx = px - x2
        dy = 0.0
        if py < y1:
            dy = y1 - py
        elif py > y2:
            dy = py - y2
        if dx == 0.0 and dy == 0.0:
            return 0.0
        return math.hypot(dx, dy)
    def nearest_person_obstacle(self):
        """
        Returns (min_distance, obstacle_dict) for the closest obstacle
        to the person. If none, returns (inf, None).
        """
        if not self.person_visible or not self.obstacles:
            return float("inf"), None
        best_dist = float("inf")
        best_ob = None
        for ob in self.obstacles:
            d = self._distance_person_to_obstacle(ob)
            if d < best_dist:
                best_dist = d
                best_ob = ob
        return best_dist, best_ob
    # ---------------------- person repositioning ----------------------
    def reposition_person_outside_threshold(self, margin_cm: float = 10.0):
        """
        Move the person just outside the safety threshold from the last
        blocking obstacle, along the direction away from that obstacle.
        """
        if not self.person_visible or self.last_blocking_obstacle is None:
            return
        ob = self.last_blocking_obstacle
        px, py = self.person_x, self.person_y
        x1, y1 = ob["x"], ob["y"]
        x2, y2 = x1 + ob["w"], y1 + ob["h"]
        # Closest point on obstacle rectangle to person
        cx = min(max(px, x1), x2)
        cy = min(max(py, y1), y2)
        vx = px - cx
        vy = py - cy
        dist = math.hypot(vx, vy)
        if dist < 1e-3:
            # If person is essentially on top of obstacle, choose arbitrary direction
            vx, vy = 1.0, 0.0
            dist = 1.0
        desired_dist = self.person_obstacle_threshold + margin_cm
        scale = desired_dist / dist
        new_px = cx + vx * scale
        new_py = cy + vy * scale
        self.person_x = new_px
        self.person_y = new_py
        # Face the person away from the obstacle (same direction as movement)
        self.person_heading = math.atan2(self.person_y - cy, self.person_x - cx)
        now = time.time()
        self.last_person_time = now
        self.last_person_move_time = now
    # ---------------------- state machine ----------------------------
    def _set_state(self, new_state: SystemState, reason: str = ""):
        if new_state is self.state:
            return
        self.state = new_state
        self.state_changed_time = time.time()
        if new_state != SystemState.SCANNING:
            # Reset scan timing when leaving SCANNING
            self.scan_started_time = None
            self.scan_start_angle = None
            self.max_clearance = 0.0
            self.best_angle = 0.0
        if new_state != SystemState.PATH_FOUND:
            # Reset path-instruction timing when leaving PATH_FOUND
            self.path_instruction_time = None
            self.path_person_ref = None
        self.state_changes += 1
        self.last_state_change_reason = reason
        print(f"[STATE] -> {self.state.value.upper()} ({reason})")
        # High-level positional behavior to match system blueprint:
        # - FOLLOWING: robot stays behind the user
        # - SCANNING / PATH_FOUND: robot will move to front using animation
        if new_state == SystemState.FOLLOWING:
            self.place_robot_behind_person()
    def _desired_follow_behavior(self):
        """
        Compute speed and turn rate to follow person.
        """
        if not self.person_visible:
            self.target_speed = 0.0
            self.target_turn_rate = 0.0
            return
        # Relative position in robot frame
        dx = self.person_x - self.robot_x
        dy = self.person_y - self.robot_y
        distance = math.hypot(dx, dy)
        angle_to_person = math.atan2(dy, dx) - self.robot_angle
        angle_to_person = math.atan2(
            math.sin(angle_to_person), math.cos(angle_to_person)
        ) # normalize
        # Maintain 100–200 cm distance
        desired_speed = 0.0
        if distance > 200.0:
            desired_speed = self.MAX_SPEED * 0.8
        elif distance < 100.0:
            desired_speed = -self.MAX_SPEED * 0.5 # small backwards
        else:
            desired_speed = 0.0
        # Turn towards person
        desired_turn = math.degrees(angle_to_person) * 1.5 # proportional
        desired_turn = max(-self.MAX_TURN_RATE, min(self.MAX_TURN_RATE, desired_turn))
        self.target_speed = desired_speed
        self.target_turn_rate = desired_turn
    # ---------------------- physics update ---------------------------
    def update(self, dt: float):
        """
        dt: seconds since last call
        """
        now = time.time()
        if self.state == SystemState.FOLLOWING:
            self.following_time += dt
        # Update robot-mounted sensors (used for LiDAR-like scanning)
        self.update_sensors()
        if self.state == SystemState.IDLE:
            if self.person_visible:
                self._set_state(SystemState.FOLLOWING, "person_detected")
        elif self.state == SystemState.FOLLOWING:
            # Obstacle detection relative to the PERSON, not the robot.
            # When an obstacle is within threshold distance of the person,
            # we trigger OBSTACLE_DETECTED so the robot can move to the
            # front and scan.
            person_obs_dist, ob = self.nearest_person_obstacle()
            if person_obs_dist <= self.person_obstacle_threshold:
                self.last_person_obstacle_distance = person_obs_dist
                self.last_blocking_obstacle = ob
                self._set_state(
                    SystemState.OBSTACLE_DETECTED,
                    f"person_obstacle_{person_obs_dist:.1f}cm",
                )
            else:
                self._desired_follow_behavior()
        elif self.state == SystemState.OBSTACLE_DETECTED:
            self.target_speed = 0.0
            self.target_turn_rate = 0.0
            if (now - self.state_changed_time) > 2.0:
                self._set_state(SystemState.SCANNING, "waited_2s_after_obstacle")
        elif self.state == SystemState.SCANNING:
            # Phase 1: move from behind the person to a point in front of them.
            desired_offset = 30.0 # cm in front of person (a few cm ahead)
            if self.person_visible:
                dx = self.robot_x - self.person_x
                dy = self.robot_y - self.person_y
                dist_pr = math.hypot(dx, dy)
                if dist_pr < 1e-3:
                    ux = math.cos(self.robot_angle)
                    uy = math.sin(self.robot_angle)
                else:
                    ux = dx / dist_pr
                    uy = dy / dist_pr
                front_x = self.person_x - ux * desired_offset
                front_y = self.person_y - uy * desired_offset
                to_front_dx = front_x - self.robot_x
                to_front_dy = front_y - self.robot_y
                dist_to_front = math.hypot(to_front_dx, to_front_dy)
                if dist_to_front > 5.0:
                    # Still moving towards front target – drive the robot there.
                    angle_target = math.atan2(to_front_dy, to_front_dx)
                    angle_err = math.atan2(
                        math.sin(angle_target - self.robot_angle),
                        math.cos(angle_target - self.robot_angle),
                    )
                    desired_turn = math.degrees(angle_err) * 2.0
                    desired_turn = max(
                        -self.MAX_TURN_RATE, min(self.MAX_TURN_RATE, desired_turn)
                    )
                    self.target_turn_rate = desired_turn
                    self.target_speed = self.MAX_SPEED * 0.6
                else:
                    # Arrived in front of person – start LiDAR-like scan.
                    self.target_speed = 0.0
                    if self.scan_started_time is None:
                        self.scan_started_time = now
                        self.scan_start_angle = self.robot_angle
                        self.max_clearance = self.sensor_distances["center"]
                        self.best_angle = self.robot_angle
                    # Perform 360 degree scan by rotating at constant rate
                    self.target_turn_rate = self.MAX_TURN_RATE
                    # Update max clearance
                    if self.sensor_distances["center"] > self.max_clearance:
                        self.max_clearance = self.sensor_distances["center"]
                        self.best_angle = self.robot_angle
                    # Check if completed 360 degrees
                    delta_angle = math.fmod(self.robot_angle - self.scan_start_angle + 4 * math.pi, 2 * math.pi)
                    if delta_angle >= 2 * math.pi or (now - self.scan_started_time) > 8.0:
                        rel_angle = self.best_angle - self.person_heading
                        rel_angle = math.atan2(math.sin(rel_angle), math.cos(rel_angle))
                        if abs(rel_angle) < math.radians(30):
                            dir_str = "center"
                        elif rel_angle < 0:
                            dir_str = "left"
                        else:
                            dir_str = "right"
                        self.scan_result = {"direction": dir_str, "clearance": self.max_clearance}
                        self._set_state(SystemState.PATH_FOUND, "scan_complete")
                    # Safety: if we've spent too long in SCANNING overall,
                    # force completion even if scan_started_time was never set.
                    elif (now - self.state_changed_time) > 8.0:
                        rel_angle = self.robot_angle - self.person_heading
                        rel_angle = math.atan2(math.sin(rel_angle), math.cos(rel_angle))
                        if abs(rel_angle) < math.radians(30):
                            dir_str = "center"
                        elif rel_angle < 0:
                            dir_str = "left"
                        else:
                            dir_str = "right"
                        self.scan_result = {
                            "direction": dir_str,
                            "clearance": self.sensor_distances["center"],
                        }
                        self._set_state(SystemState.PATH_FOUND, "scan_timeout_fallback")
            else:
                # If we somehow lost the person, stop scanning and go idle.
                self.target_speed = 0.0
                self.target_turn_rate = 0.0
                self._set_state(SystemState.IDLE, "scan_cancel_no_person")
        elif self.state == SystemState.PATH_FOUND:
            # After scanning, the robot should drive back behind the person,
            # then wait for the user to react to the instruction.
            self.target_turn_rate = 0.0
            if self.person_visible:
                # First, ensure the person is moved just outside the
                # obstacle safety threshold, away from the blocking obstacle.
                if self.path_instruction_time is None:
                    self.reposition_person_outside_threshold(margin_cm=10.0)
                # Drive towards the desired "behind" position
                behind_x, behind_y = self.behind_target_position(distance_cm=100.0)
                dx = behind_x - self.robot_x
                dy = behind_y - self.robot_y
                dist_to_behind = math.hypot(dx, dy)
                if dist_to_behind > 5.0:
                    # Still returning to place behind the person
                    angle_target = math.atan2(dy, dx)
                    angle_err = math.atan2(
                        math.sin(angle_target - self.robot_angle),
                        math.cos(angle_target - self.robot_angle),
                    )
                    desired_turn = math.degrees(angle_err) * 2.0
                    desired_turn = max(
                        -self.MAX_TURN_RATE, min(self.MAX_TURN_RATE, desired_turn)
                    )
                    self.target_turn_rate = desired_turn
                    self.target_speed = self.MAX_SPEED * 0.6
                else:
                    # Arrived behind the person – hold position and start
                    # the "instruction" phase if not already started.
                    self.target_speed = 0.0
                    if self.path_instruction_time is None:
                        self.path_instruction_time = now
                        self.path_person_ref = (self.person_x, self.person_y)
                    # Check for person movement since instruction
                    if self.path_person_ref is not None:
                        ref_x, ref_y = self.path_person_ref
                        moved_dist = math.hypot(
                            self.person_x - ref_x, self.person_y - ref_y
                        )
                    else:
                        moved_dist = 0.0
                    # If user has started moving, resume following
                    if moved_dist > 10.0:
                        self._set_state(SystemState.FOLLOWING, "person_started_moving")
                    # If 5 seconds pass with no movement, go to IDLE
                    elif self.path_instruction_time is not None and (
                        now - self.path_instruction_time
                    ) > 50.0:
                        self._set_state(
                            SystemState.IDLE, "no_movement_after_instruction"
                        )
            else:
                # Lost the person – stop and go idle
                self.target_speed = 0.0
                self._set_state(SystemState.IDLE, "path_phase_no_person")
        # Smooth speed
        if self.current_speed < self.target_speed:
            self.current_speed = min(
                self.current_speed + self.accel * dt, self.target_speed
            )
        else:
            self.current_speed = max(
                self.current_speed - self.decel * dt, self.target_speed
            )
        # Smooth turn
        turn_step = self.MAX_TURN_RATE * dt * 2.0
        if self.current_turn_rate < self.target_turn_rate:
            self.current_turn_rate = min(
                self.current_turn_rate + turn_step, self.target_turn_rate
            )
        else:
            self.current_turn_rate = max(
                self.current_turn_rate - turn_step, self.target_turn_rate
            )
        # Integrate motion
        v = self.current_speed
        self.robot_x += v * math.cos(self.robot_angle) * dt
        self.robot_y += v * math.sin(self.robot_angle) * dt
        self.robot_angle += math.radians(self.current_turn_rate) * dt
        self.total_distance += abs(v) * dt
        # Global IDLE condition: if the person has not moved for 5 seconds
        # while we are in FOLLOWING, the system should become idle.
        # We do NOT apply this during obstacle-handling (OBSTACLE_DETECTED,
        # SCANNING, PATH_FOUND) so that the full process can complete.
        if (
            self.state == SystemState.FOLLOWING
            and self.person_visible
            and self.last_person_move_time is not None
            and (now - self.last_person_move_time) > 5.0
        ):
            self._set_state(SystemState.IDLE, "person_not_moving_5s")
    # ---------------------- statistics snapshot ---------------------
    def snapshot_stats(self) -> dict:
        return {
            "state": self.state.value,
            "person_id": self.person_id,
            "person_conf": self.person_conf,
            "person_visible": self.person_visible,
            "robot_x": self.robot_x,
            "robot_y": self.robot_y,
            "robot_angle_deg": math.degrees(self.robot_angle),
            "speed": self.current_speed,
            "turn_rate": self.current_turn_rate,
            "sensor_left": self.sensor_distances["left"],
            "sensor_center": self.sensor_distances["center"],
            "sensor_right": self.sensor_distances["right"],
            "total_distance": self.total_distance / 100.0, # meters
            "obstacles_avoided": self.obstacles_avoided,
            "following_time": self.following_time,
            "target_switches": self.target_switches,
            "commands_sent": self.commands_sent,
            "state_changes": self.state_changes,
        }
# ---------------------------------------------------------------------------
# Simulation canvas widget (2D environment)
# ---------------------------------------------------------------------------
class SimulationWidget(QWidget):
    def _init_(self, engine: SimulationEngine, parent=None):
        super()._init_(parent)
        self.engine = engine
        self.setMinimumSize(1000, 700)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setAutoFillBackground(True)
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w = self.width()
        h = self.height()
        cx = w / 2.0
        cy = h / 2.0
        # Background
        painter.fillRect(self.rect(), QColor("#1e1e1e"))
        # Coordinate transform: world cm -> pixels (1 px = 1 cm)
        scale = 1.0
        def world_to_screen(x_cm, y_cm):
            return QPointF(cx + x_cm * scale, cy - y_cm * scale)
        # Grid
        painter.setPen(QPen(QColor(80, 80, 80), 1))
        grid_step = 50.0 # cm
        for gx in range(-800, 801, int(grid_step)):
            p1 = world_to_screen(gx, -400)
            p2 = world_to_screen(gx, 400)
            painter.drawLine(p1, p2)
        for gy in range(-400, 401, int(grid_step)):
            p1 = world_to_screen(-800, gy)
            p2 = world_to_screen(800, gy)
            painter.drawLine(p1, p2)
        # Safe zone (transparent green around robot)
        painter.setBrush(QBrush(QColor(0, 255, 0, 30)))
        painter.setPen(Qt.NoPen)
        safe_radius = 200.0 # cm
        center_pt = world_to_screen(self.engine.robot_x, self.engine.robot_y)
        painter.drawEllipse(
            center_pt, safe_radius * scale, safe_radius * scale
        )
        # Obstacles
        for ob in self.engine.obstacles:
            rect = QRectF(
                world_to_screen(ob["x"], ob["y"]).x(),
                world_to_screen(ob["x"], ob["y"]).y() - ob["h"] * scale,
                ob["w"] * scale,
                ob["h"] * scale,
            )
            painter.setBrush(QBrush(QColor("#757575")))
            painter.setPen(QPen(QColor("#F44336"), 2))
            painter.drawRect(rect)
        # Person
        if self.engine.person_visible:
            px = self.engine.person_x
            py = self.engine.person_y
            heading = self.engine.person_heading
            # Body as ellipse elongated along heading
            body_len = 40.0
            body_w = 20.0
            body_center_pt = world_to_screen(px, py)
            painter.setBrush(QBrush(QColor("#2196F3")))
            painter.setPen(QPen(QColor("#FFFFFF"), 2))
            painter.save()
            painter.translate(body_center_pt)
            painter.rotate(math.degrees(heading))
            painter.drawEllipse(QRectF(-body_len / 2, -body_w / 2, body_len, body_w))
            painter.restore()
            # Head as a small circle in front of the body
            head_r = 10.0
            head_offset = body_len / 2
            head_cx = px + head_offset * math.cos(heading)
            head_cy = py + head_offset * math.sin(heading)
            head_center = world_to_screen(head_cx, head_cy)
            painter.setBrush(QBrush(QColor("#BBDEFB")))
            painter.drawEllipse(head_center, head_r, head_r)
            # Heading line from head center forward
            face_tip_x = head_cx + head_r * 1.5 * math.cos(heading)
            face_tip_y = head_cy + head_r * 1.5 * math.sin(heading)
            face_tip = world_to_screen(face_tip_x, face_tip_y)
            painter.setPen(QPen(QColor("#FFEB3B"), 2))
            painter.drawLine(head_center, face_tip)
            # Label
            painter.setFont(QFont("Arial", 9))
            painter.setPen(QColor("#FFFFFF"))
            painter.drawText(
                body_center_pt + QPointF(-25, -25),
                f"ID {self.engine.person_id} "
                f"({self.engine.person_conf:.2f})",
            )
            # Person-mounted sensors (microwave-like) - visualize 3 short rays
            person_sensors = self.engine.person_sensor_distances(max_range=200.0)
            angles_deg_p = {"left": -30.0, "center": 0.0, "right": 30.0}
            origin_p = world_to_screen(self.engine.person_x, self.engine.person_y)
            for key, ang_deg in angles_deg_p.items():
                ang_rad = math.radians(ang_deg)
                dist = person_sensors[key]
                end_x = self.engine.person_x + dist * math.cos(ang_rad)
                end_y = self.engine.person_y + dist * math.sin(ang_rad)
                end_pt = world_to_screen(end_x, end_y)
                # Color by distance
                if dist > 100.0:
                    col = QColor("#00E5FF") # safe cyan
                elif dist > 50.0:
                    col = QColor("#FFEA00") # warning yellow
                else:
                    col = QColor("#FF1744") # danger red
                painter.setPen(QPen(col, 2, Qt.DashLine))
                painter.drawLine(origin_p, end_pt)
        # Ultrasonic sensor rays
        ray_colors = {}
        for key, dist in self.engine.sensor_distances.items():
            if dist > 40.0:
                ray_colors[key] = QColor("#00C853") # green
            elif dist > 20.0:
                ray_colors[key] = QColor("#FFD600") # yellow
            else:
                ray_colors[key] = QColor("#D50000") # red
        angles_deg = {"left": -30.0, "center": 0.0, "right": 30.0}
        origin = world_to_screen(self.engine.robot_x, self.engine.robot_y)
        painter.setFont(QFont("Consolas", 9))
        for key, ang_deg in angles_deg.items():
            ang_rad = self.engine.robot_angle + math.radians(ang_deg)
            dist = self.engine.sensor_distances[key]
            end_x = self.engine.robot_x + dist * math.cos(ang_rad)
            end_y = self.engine.robot_y + dist * math.sin(ang_rad)
            end_pt = world_to_screen(end_x, end_y)
            painter.setPen(QPen(ray_colors[key], 2))
            painter.drawLine(origin, end_pt)
            painter.drawText(
                end_pt + QPointF(5, -5),
                f"{int(dist)}cm",
            )
        # Robot
        # Triangle 40x30 cm (40 forward length, 30 width)
        body_len = 40.0
        body_w = 30.0
        # Robot color depends on state
        state = self.engine.state
        if state == SystemState.FOLLOWING:
            robot_color = QColor("#4CAF50") # green
        elif state == SystemState.OBSTACLE_DETECTED:
            robot_color = QColor("#F44336") # red
        elif state == SystemState.SCANNING:
            robot_color = QColor("#FFEB3B") # yellow
        elif state == SystemState.PATH_FOUND:
            robot_color = QColor("#00BCD4") # cyan
        else:
            robot_color = QColor("#9E9E9E") # gray
        angle = self.engine.robot_angle
        # Triangle vertices in robot frame
        p_front = QPointF(
            self.engine.robot_x + body_len * math.cos(angle),
            self.engine.robot_y + body_len * math.sin(angle),
        )
        p_back_left = QPointF(
            self.engine.robot_x
            - body_len * 0.4 * math.cos(angle)
            - body_w / 2.0 * math.sin(angle),
            self.engine.robot_y
            - body_len * 0.4 * math.sin(angle)
            + body_w / 2.0 * math.cos(angle),
        )
        p_back_right = QPointF(
            self.engine.robot_x
            - body_len * 0.4 * math.cos(angle)
            + body_w / 2.0 * math.sin(angle),
            self.engine.robot_y
            - body_len * 0.4 * math.sin(angle)
            - body_w / 2.0 * math.cos(angle),
        )
        poly = [
            world_to_screen(p_front.x(), p_front.y()),
            world_to_screen(p_back_left.x(), p_back_left.y()),
            world_to_screen(p_back_right.x(), p_back_right.y()),
        ]
        painter.setBrush(QBrush(robot_color))
        painter.setPen(QPen(QColor("#000000"), 2))
        painter.drawPolygon(*poly)
        # Wheels (small circles near back corners)
        wheel_r = 5.0
        for p_back in (p_back_left, p_back_right):
            wp = world_to_screen(p_back.x(), p_back.y())
            painter.setBrush(QBrush(QColor("#212121")))
            painter.setPen(QPen(QColor("#000000"), 1))
            painter.drawEllipse(wp, wheel_r, wheel_r)
        # Compass
        painter.setPen(QPen(QColor("#00ACC1"), 2))
        painter.setFont(QFont("Arial", 10, QFont.Bold))
        cx_compass = w - 80
        cy_compass = 80
        painter.drawEllipse(QPointF(cx_compass, cy_compass), 40, 40)
        painter.drawText(cx_compass - 6, cy_compass - 30, "N")
        painter.drawText(cx_compass - 6, cy_compass + 40, "S")
        painter.drawText(cx_compass - 40, cy_compass + 5, "W")
        painter.drawText(cx_compass + 30, cy_compass + 5, "E")
# ---------------------------------------------------------------------------
# System state panel
# ---------------------------------------------------------------------------
class SystemStatePanel(QGroupBox):
    def _init_(self, parent=None):
        super()._init_("SYSTEM STATE", parent)
        self.setStyleSheet("QGroupBox { color: white; }")
        layout = QFormLayout()
        layout.setLabelAlignment(Qt.AlignLeft)
        self.lbl_state = QLabel("[IDLE]")
        self.lbl_target_id = QLabel("None")
        self.lbl_target_distance = QLabel("N/A")
        self.lbl_target_angle = QLabel("N/A")
        self.lbl_speed = QLabel("0.0 m/s")
        self.lbl_direction = QLabel("Stopped")
        self.lbl_battery = QLabel("100%")
        self.pb_left = QProgressBar()
        self.pb_center = QProgressBar()
        self.pb_right = QProgressBar()
        for pb in (self.pb_left, self.pb_center, self.pb_right):
            pb.setRange(0, 200)
            pb.setTextVisible(True)
        self.lbl_audio = QLabel("Enabled")
        self.lbl_last_command = QLabel("None")
        layout.addRow("Current State:", self.lbl_state)
        layout.addRow("Target ID:", self.lbl_target_id)
        layout.addRow("Distance:", self.lbl_target_distance)
        layout.addRow("Angle:", self.lbl_target_angle)
        layout.addRow("Speed:", self.lbl_speed)
        layout.addRow("Direction:", self.lbl_direction)
        layout.addRow("Battery:", self.lbl_battery)
        layout.addRow("Sensor Left:", self.pb_left)
        layout.addRow("Sensor Center:", self.pb_center)
        layout.addRow("Sensor Right:", self.pb_right)
        layout.addRow("Audio:", self.lbl_audio)
        layout.addRow("Last Command:", self.lbl_last_command)
        self.setLayout(layout)
    def update_from_engine(self, engine: SimulationEngine):
        # State label + background color
        state = engine.state
        state_text = state.name
        bg_color = "#757575"
        if state == SystemState.FOLLOWING:
            bg_color = "#2E7D32"
        elif state == SystemState.OBSTACLE_DETECTED:
            bg_color = "#B71C1C"
        elif state == SystemState.SCANNING:
            bg_color = "#F9A825"
        elif state == SystemState.PATH_FOUND:
            bg_color = "#00838F"
        self.lbl_state.setText(f"[{state_text}]")
        self.lbl_state.setStyleSheet(
            f"QLabel {{ background-color: {bg_color}; color: white; padding: 4px; }}"
        )
        # Target
        if engine.person_visible:
            dx = engine.person_x - engine.robot_x
            dy = engine.person_y - engine.robot_y
            dist = math.hypot(dx, dy) / 100.0 # meters
            ang = math.degrees(
                math.atan2(dy, dx) - engine.robot_angle
            )
            self.lbl_target_id.setText(str(engine.person_id))
            self.lbl_target_distance.setText(f"{dist:.2f} m")
            self.lbl_target_angle.setText(f"{ang:+.1f}°")
        else:
            self.lbl_target_id.setText("None")
            self.lbl_target_distance.setText("N/A")
            self.lbl_target_angle.setText("N/A")
        # Robot speed/direction
        speed_m_s = engine.current_speed / 100.0
        self.lbl_speed.setText(f"{speed_m_s:.2f} m/s")
        if abs(speed_m_s) < 0.05:
            direction = "Stopped"
        elif speed_m_s > 0:
            direction = "Forward"
        else:
            direction = "Backward"
        self.lbl_direction.setText(direction)
        # Fake battery (static for now)
        self.lbl_battery.setText("87%")
        # Sensors
        self.pb_left.setValue(int(engine.sensor_distances["left"]))
        self.pb_center.setValue(int(engine.sensor_distances["center"]))
        self.pb_right.setValue(int(engine.sensor_distances["right"]))
# ---------------------------------------------------------------------------
# Controls & statistics panel
# ---------------------------------------------------------------------------
class ControlsStatsPanel(QGroupBox):
    def _init_(self, parent=None):
        super()._init_("CONTROLS & STATISTICS", parent)
        self.setStyleSheet("QGroupBox { color: white; }")
        main_layout = QVBoxLayout()
        # Top row - buttons
        btn_row = QHBoxLayout()
        self.btn_reset = QPushButton("Reset Tracking")
        self.btn_add_obstacle = QPushButton("Add Random Obstacle")
        self.btn_clear_obstacles = QPushButton("Clear Obstacles")
        self.btn_pause = QPushButton("Pause/Resume")
        self.btn_screenshot = QPushButton("Screenshot")
        self.btn_export_log = QPushButton("Export Log")
        for btn in (
            self.btn_reset,
            self.btn_add_obstacle,
            self.btn_clear_obstacles,
            self.btn_pause,
            self.btn_screenshot,
            self.btn_export_log,
        ):
            btn_row.addWidget(btn)
        main_layout.addLayout(btn_row)
        # Statistics labels
        self.txt_stats = QTextEdit()
        self.txt_stats.setReadOnly(True)
        self.txt_stats.setStyleSheet(
            "QTextEdit { background-color: #121212; color: #BDBDBD; font-family: Consolas; font-size: 10pt; }"
        )
        main_layout.addWidget(self.txt_stats, stretch=1)
        # Event log
        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setStyleSheet(
            "QTextEdit { background-color: #101010; color: #9E9E9E; font-family: Consolas; font-size: 9pt; }"
        )
        main_layout.addWidget(self.txt_log, stretch=1)
        self.setLayout(main_layout)
        self._log_entries = []
    def append_log(self, text: str):
        timestamp = time.strftime("%H:%M:%S")
        line = f"[{timestamp}] {text}"
        self._log_entries.append(line)
        self.txt_log.append(line)
        self.txt_log.verticalScrollBar().setValue(
            self.txt_log.verticalScrollBar().maximum()
        )
    def update_stats(self, sim_stats: dict, vision_stats: dict, runtime_s: float):
        def fmt_time(sec):
            m, s = divmod(int(sec), 60)
            h, m = divmod(m, 60)
            return f"{h:02d}:{m:02d}:{s:02d}"
        txt = []
        txt.append("PERFORMANCE METRICS")
        txt.append("=" * 55)
        txt.append(
            f"Vision Processing: YOLO FPS: {vision_stats.get('yolo_fps', 0.0):.1f} "
            f"DeepSORT FPS: {vision_stats.get('tracker_fps', 0.0):.1f}"
        )
        txt.append("")
        txt.append(
            f"Simulation: Total Distance: {sim_stats['total_distance']:.2f} m "
            f"State Changes: {sim_stats['state_changes']}"
        )
        txt.append(
            f"Robot: Speed: {sim_stats['speed']/100.0:.2f} m/s "
            f"Turn Rate: {sim_stats['turn_rate']:.1f} deg/s"
        )
        txt.append(
            f"Sensors: L={sim_stats['sensor_left']:.1f} cm, "
            f"C={sim_stats['sensor_center']:.1f} cm, "
            f"R={sim_stats['sensor_right']:.1f} cm"
        )
        txt.append("")
        txt.append(
            f"Following Time: {fmt_time(sim_stats['following_time'])} "
            f"Target Switches: {sim_stats['target_switches']}"
        )
        txt.append(
            f"Runtime: {fmt_time(runtime_s)}"
        )
        self.txt_stats.setPlainText("\n".join(txt))
# ---------------------------------------------------------------------------
# Main application window
# ---------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def _init_(self):
        super()._init_()
        self.setWindowTitle("Intelligent Follower Robot - Complete Simulation")
        self.setGeometry(50, 50, 1600, 900)
        self.engine = SimulationEngine()
        self.audio = AudioManager(enable_audio=True)
        self.vision_thread = VisionThread(camera_id=0)
        self.last_vision_stats = {"persons": 0, "yolo_fps": 0.0, "tracker_fps": 0.0}
        self.paused = False
        self.start_time = time.time()
        self._prev_state = self.engine.state
        self._build_ui()
        # Simple helper so that important log entries are also voiced.
        # Use this for system alerts and state logs that the user must hear.
        self._speak_logs = True
        self._path_spoken = False # has current PATH_FOUND instruction been spoken?
        # Connect signals
        self.vision_thread.frame_ready.connect(self.on_frame_ready)
        self.controls.btn_reset.clicked.connect(self.on_reset_tracking)
        self.controls.btn_add_obstacle.clicked.connect(self.on_add_obstacle)
        self.controls.btn_clear_obstacles.clicked.connect(self.on_clear_obstacles)
        self.controls.btn_pause.clicked.connect(self.on_pause_resume)
        self.controls.btn_screenshot.clicked.connect(self.on_screenshot)
        self.controls.btn_export_log.clicked.connect(self.on_export_log)
        # Simulation timer (60 FPS)
        self.sim_timer = QTimer(self)
        self.sim_timer.timeout.connect(self.update_simulation)
        self.sim_timer.start(int(1000 / 60))
        # FPS timer for webcam panel overlay
        self.webcam_fps = 0.0
        self.webcam_frame_count = 0
        self.webcam_last_time = time.time()
        # YOLO target selection
        self.target_person_id = None
        # Ensure window receives key events
        self.setFocusPolicy(Qt.StrongFocus)
        self.audio.speak("System ready. Following mode activated.")
        self.engine._set_state(SystemState.FOLLOWING, "startup")
        # Start vision thread
        self.vision_thread.start()
    # ---------------------- UI construction --------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QGridLayout()
        central.setLayout(root_layout)
        # Left column: webcam + system state
        left_layout = QVBoxLayout()
        # Webcam group
        gb_webcam = QGroupBox("WEBCAM FEED")
        gb_webcam.setStyleSheet("QGroupBox { color: white; }")
        vb_cam = QVBoxLayout()
        self.lbl_camera = QLabel()
        self.lbl_camera.setFixedSize(320, 240)
        self.lbl_camera.setStyleSheet(
            "QLabel { background-color: #000000; border: 1px solid #333333; }"
        )
        self.lbl_detection_info = QLabel("Persons: 0 | Target: None | Conf: 0.00")
        self.lbl_detection_info.setStyleSheet("QLabel { color: #BDBDBD; }")
        vb_cam.addWidget(self.lbl_camera, alignment=Qt.AlignCenter)
        vb_cam.addWidget(self.lbl_detection_info)
        gb_webcam.setLayout(vb_cam)
        left_layout.addWidget(gb_webcam)
        # System state panel
        self.system_state_panel = SystemStatePanel()
        left_layout.addWidget(self.system_state_panel)
        left_container = QWidget()
        left_container.setLayout(left_layout)
        # Right column: simulation + controls/stats
        right_layout = QVBoxLayout()
        self.sim_widget = SimulationWidget(self.engine)
        right_layout.addWidget(self.sim_widget, stretch=3)
        self.controls = ControlsStatsPanel()
        right_layout.addWidget(self.controls, stretch=2)
        right_container = QWidget()
        right_container.setLayout(right_layout)
        # Root layout: 2 columns
        root_layout.addWidget(left_container, 0, 0)
        root_layout.addWidget(right_container, 0, 1)
        root_layout.setColumnStretch(0, 1)
        root_layout.setColumnStretch(1, 3)
        # Dark theme base
        self.setStyleSheet(
            """
            QMainWindow { background-color: #202020; }
            QLabel { color: #FFFFFF; }
            QPushButton {
                background-color: #424242;
                color: white;
                padding: 6px 12px;
            }
            QPushButton:hover {
                background-color: #616161;
            }
            """
        )
    def log_and_speak(self, text: str):
        """
        Convenience: write to event log and speak the same message so that
        all key alerts are available in audio form.
        """
        self.controls.append_log(text)
        if self._speak_logs:
            self.audio.speak(text)
    # ---------------------- vision callback --------------------------
    def on_frame_ready(self, frame_bgr, tracks, vision_stats):
        self.last_vision_stats = vision_stats
        h, w, _ = frame_bgr.shape
        # Select / keep target track
        target_track = None
        largest_area = 0.0
        num_confirmed = 0
        for track in tracks:
            if not track.is_confirmed():
                continue
            num_confirmed += 1
            track_id = track.track_id
            bbox = track.to_ltrb()
            x1, y1, x2, y2 = bbox
            area = max((x2 - x1) * (y2 - y1), 1.0)
            # Auto-select largest if none
            if self.target_person_id is None and area > largest_area:
                self.target_person_id = track_id
                largest_area = area
            # Draw bounding boxes
            is_target = (track_id == self.target_person_id)
            color = (0, 255, 0) if is_target else (0, 0, 255)
            cv2.rectangle(
                frame_bgr,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                color,
                2,
            )
            label = f"ID {track_id}"
            if is_target:
                label += " [TARGET]"
                target_track = track
            cv2.putText(
                frame_bgr,
                label,
                (int(x1), int(y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )
        # NOTE: Person position in the simulation is now controlled
        # via keyboard arrows, so we no longer map YOLO detections
        # into the virtual environment. The webcam panel still shows
        # all detections and track IDs for reference.
        # Draw center line, detection info
        frame_center_x = w // 2
        cv2.line(
            frame_bgr,
            (frame_center_x, 0),
            (frame_center_x, h),
            (255, 255, 0),
            1,
        )
        # Convert to QPixmap
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        qimg = QImage(
            rgb.data,
            rgb.shape[1],
            rgb.shape[0],
            rgb.strides[0],
            QImage.Format_RGB888,
        )
        pix = QPixmap.fromImage(qimg).scaled(
            self.lbl_camera.width(),
            self.lbl_camera.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.lbl_camera.setPixmap(pix)
        # Detection info label
        persons = vision_stats.get("persons", 0)
        target = self.engine.person_id if self.engine.person_visible else "None"
        conf = self.engine.person_conf if self.engine.person_visible else 0.0
        self.lbl_detection_info.setText(
            f"Persons: {persons} | Target ID: {target} | Confidence: {conf:.2f}"
        )
    # ---------------------- simulation update ------------------------
    def update_simulation(self):
        if self.paused:
            return
        now = time.time()
        dt = 1.0 / 60.0
        prev_state = self.engine.state
        self.engine.update(dt)
        # Handle high-level state transition side-effects / audio
        new_state = self.engine.state
        if new_state != prev_state:
            if new_state == SystemState.OBSTACLE_DETECTED:
                # Glasses / microwave sensor would have triggered this in hardware
                self.log_and_speak("Obstacle detected ahead. Please stop.")
            elif new_state == SystemState.SCANNING:
                # Robot has moved to the front and is scanning with LiDAR
                self.log_and_speak("Robot moved in front of user and started scanning.")
                self.log_and_speak("Scanning environment.")
            elif new_state == SystemState.PATH_FOUND:
                # We'll announce the safe path only after the robot has
                # returned behind the person; see logic further below.
                self._path_spoken = False
            elif new_state == SystemState.FOLLOWING and prev_state != SystemState.IDLE:
                # Robot returns to follower position behind user
                self.log_and_speak("Robot returned behind user and resumed following.")
                self.log_and_speak("Resuming following mode.")
            elif new_state == SystemState.IDLE and prev_state == SystemState.PATH_FOUND:
                self.log_and_speak("No movement detected. System entering idle state.")
        self._prev_state = new_state
        # Update panels
        self.system_state_panel.update_from_engine(self.engine)
        sim_stats = self.engine.snapshot_stats()
        runtime_s = now - self.start_time
        self.controls.update_stats(sim_stats, self.last_vision_stats, runtime_s)
        # If we're in PATH_FOUND and the robot has reached its place behind
        # the person, speak the instruction (relative to the person) once.
        if (
            self.engine.state == SystemState.PATH_FOUND
            and not self._path_spoken
            and self.engine.path_instruction_time is not None
        ):
            sr = self.engine.scan_result or {}
            raw_dir = sr.get("direction", "center")
            clearance = int(sr.get("clearance", 0))
            if clearance <= 0:
                self.log_and_speak("No clear path detected. Please turn around.")
            else:
                # Phrase direction relative to the person
                if raw_dir == "left":
                    phrase_dir = "to your left"
                elif raw_dir == "right":
                    phrase_dir = "to your right"
                else:
                    phrase_dir = "straight ahead of you"
                self.log_and_speak(
                    f"Safe path {phrase_dir}. Clearance {clearance} centimeters."
                )
            self._path_spoken = True
        # Event log examples
        if self.engine.state == SystemState.FOLLOWING and self.engine.person_visible:
            dx = self.engine.person_x - self.engine.robot_x
            dy = self.engine.person_y - self.engine.robot_y
            dist_m = math.hypot(dx, dy) / 100.0
            ang = math.degrees(
                math.atan2(dy, dx) - self.engine.robot_angle
            )
            self.controls.append_log(
                f"Following mode: Person detected at ({dist_m:.2f}m, {ang:+.1f}°)"
            )
        # Repaint simulation
        self.sim_widget.update()
    # ---------------------- button handlers --------------------------
    def on_reset_tracking(self):
        self.target_person_id = None
        self.engine.person_visible = False
        self.log_and_speak("Target tracking reset.")
    def on_add_obstacle(self):
        self.engine.add_random_obstacle()
        self.log_and_speak("Random obstacle added.")
    def on_clear_obstacles(self):
        self.engine.clear_obstacles()
        self.log_and_speak("All obstacles cleared.")
    def on_pause_resume(self):
        self.paused = not self.paused
        state = "paused" if self.paused else "resumed"
        self.log_and_speak(f"Simulation {state}.")
    def on_screenshot(self):
        # Simple screenshot of main window
        pixmap = self.grab()
        filename = time.strftime("simulation_%Y%m%d_%H%M%S.png")
        pixmap.save(filename, "PNG")
        self.log_and_speak(f"Screenshot saved as {filename}.")
    def on_export_log(self):
        filename = time.strftime("simulation_log_%Y%m%d_%H%M%S.csv")
        sim_stats = self.engine.snapshot_stats()
        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "timestamp",
                    "robot_x",
                    "robot_y",
                    "robot_angle_deg",
                    "person_x",
                    "person_y",
                    "person_visible",
                    "sensor_left",
                    "sensor_center",
                    "sensor_right",
                    "state",
                ]
            )
            writer.writerow(
                [
                    time.time(),
                    sim_stats["robot_x"],
                    sim_stats["robot_y"],
                    sim_stats["robot_angle_deg"],
                    self.engine.person_x,
                    self.engine.person_y,
                    int(self.engine.person_visible),
                    sim_stats["sensor_left"],
                    sim_stats["sensor_center"],
                    sim_stats["sensor_right"],
                    sim_stats["state"],
                ]
            )
        self.log_and_speak(f"Simulation log exported as {filename}.")
    # ---------------------- keyboard control -------------------------
    def keyPressEvent(self, event):
        step = 20.0 # cm per key press
        handled = False
        # Only allow manual person movement in IDLE or FOLLOWING.
        if self.engine.state in (SystemState.IDLE, SystemState.FOLLOWING):
            if event.key() == Qt.Key_Up:
                self.engine.move_person(0.0, step)
                handled = True
            elif event.key() == Qt.Key_Down:
                self.engine.move_person(0.0, -step)
                handled = True
            elif event.key() == Qt.Key_Left:
                self.engine.move_person(-step, 0.0)
                handled = True
            elif event.key() == Qt.Key_Right:
                self.engine.move_person(step, 0.0)
                handled = True
        if handled:
            # Keep focus on the main window so arrows continue to work
            self.setFocus()
            return
        super().keyPressEvent(event)
    # ---------------------- shutdown -------------------------------
    de