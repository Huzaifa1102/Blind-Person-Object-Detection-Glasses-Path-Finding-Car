"""
Main Control System for Intelligent Follower Robot
Handles video processing, person tracking, and robot control
"""

import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import requests
import pyttsx3
import threading
import time
import serial
import json
from enum import Enum

class SystemState(Enum):
    FOLLOWING = "following"
    OBSTACLE_DETECTED = "obstacle_detected"
    SCANNING = "scanning"
    PATH_FOUND = "path_found"
    IDLE = "idle"

class IntelligentFollowerSystem:
    def init(self, esp32_ip, robot_port='COM3'):
        # ESP32-CAM configuration
        self.esp32_ip = esp32_ip
        self.stream_url = f"http://{esp32_ip}/stream"
        self.sensor_url = f"http://{esp32_ip}/sensor"
        
        # Robot serial communication
        self.robot_serial = serial.Serial(robot_port, 115200, timeout=1)
        
        # YOLOv8 for person detection
        self.model = YOLO('yolov8n.pt')
        
        # DeepSORT for tracking
        self.tracker = DeepSort(
            max_age=30,
            n_init=3,
            nms_max_overlap=1.0,
            max_cosine_distance=0.3,
            nn_budget=None
        )
        
        # Text-to-speech engine
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)
        
        # System state
        self.current_state = SystemState.IDLE
        self.target_person_id = None
        self.obstacle_detected = False
        self.last_obstacle_check = time.time()
        
        # Threading locks
        self.state_lock = threading.Lock()
        self.speaking_lock = threading.Lock()
        
        # Frame storage
        self.current_frame = None
        self.frame_lock = threading.Lock()
        
        print("System initialized successfully")
    
    def speak(self, text):
        """Thread-safe text-to-speech"""
        def _speak():
            with self.speaking_lock:
                print(f"[AUDIO] {text}")
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
        
        threading.Thread(target=_speak, daemon=True).start()
    
    def check_ultrasonic_sensor(self):
        """Check ESP32 ultrasonic sensor for obstacles"""
        try:
            response = requests.get(self.sensor_url, timeout=2)
            data = response.json()
            return data['obstacle'], data['distance']
        except Exception as e:
            print(f"Error reading sensor: {e}")
            return False, -1
    
    def send_robot_command(self, command, data=None):
        """Send commands to the robot via serial"""
        try:
            message = {
                'cmd': command,
                'data': data or {}
            }
            self.robot_serial.write((json.dumps(message) + '\n').encode())
            print(f"[ROBOT] Sent: {command}")
        except Exception as e:
            print(f"Error sending command: {e}")
    
    def get_person_center(self, bbox):
        """Calculate center point of bounding box"""
        x1, y1, x2, y2 = bbox
        return int((x1 + x2) / 2), int((y1 + y2) / 2)
    
    def process_frame(self):
        """Process video frame for person detection and tracking"""
        if self.current_frame is None:
            return
        
        with self.frame_lock:
            frame = self.current_frame.copy()
        
        # Run YOLO detection
        results = self.model(frame, classes=[0], conf=0.5, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                detections.append(([x1, y1, x2-x1, y2-y1], conf, 'person'))
        
        # Update tracker
        tracks = self.tracker.update_tracks(detections, frame=frame)
        
        # Find target person or closest person
        target_track = None
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            bbox = track.to_ltrb()
            
            # Draw bounding box
            color = (0, 255, 0) if track_id == self.target_person_id else (255, 0, 0)
            cv2.rectangle(frame, 
                         (int(bbox[0]), int(bbox[1])), 
                         (int(bbox[2]), int(bbox[3])), 
                         color, 2)
            cv2.putText(frame, f"ID: {track_id}", 
                       (int(bbox[0]), int(bbox[1]-10)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # If no target, select the closest/largest person
            if self.target_person_id is None:
                self.target_person_id = track_id
                self.speak("Target person locked")
            
            if track_id == self.target_person_id:
                target_track = track
        
        # Control robot based on target person
        if target_track is not None and self.current_state == SystemState.FOLLOWING:
            bbox = target_track.to_ltrb()
            center_x, center_y = self.get_person_center(bbox)
            
            # Calculate relative position for robot
            frame_center_x = frame.shape[1] // 2
            offset = center_x - frame_center_x
            
            # Send follow command
            self.send_robot_command('follow', {
                'offset': offset,
                'bbox_width': bbox[2] - bbox[0]
            })
        
        return frame
    
    def video_stream_thread(self):
        """Continuously capture video from ESP32-CAM"""
        while True:
            try:
                stream = requests.get(self.stream_url, stream=True, timeout=10)
                bytes_data = bytes()
                
                for chunk in stream.iter_content(chunk_size=1024):
                    bytes_data += chunk
                    a = bytes_data.find(b'\xff\xd8')  # JPEG start
                    b = bytes_data.find(b'\xff\xd9')  # JPEG end
                    
                    if a != -1 and b != -1:
                        jpg = bytes_data[a:b+2]
                        bytes_data = bytes_data[b+2:]
                        
                        frame = cv2.imdecode(
                            np.frombuffer(jpg, dtype=np.uint8), 
                            cv2.IMREAD_COLOR
                        )
                        
                        if frame is not None:
                            with self.frame_lock:
                                self.current_frame = frame
                                
            except Exception as e:
                print(f"Stream error: {e}")
                time.sleep(2)
    
    def obstacle_monitoring_thread(self):
        """Monitor ultrasonic sensor for obstacles"""
        while True:
            if self.current_state == SystemState.FOLLOWING:
                obstacle, distance = self.check_ultrasonic_sensor()
                
                if obstacle and not self.obstacle_detected:
                    self.obstacle_detected = True
                    self.handle_obstacle_detected()
                elif not obstacle and self.obstacle_detected:
                    self.obstacle_detected = False
            
            time.sleep(0.5)
    
    def handle_obstacle_detected(self):
        """Handle obstacle detection event"""
        with self.state_lock:
            if self.current_state != SystemState.FOLLOWING:
                return
            
            self.current_state = SystemState.OBSTACLE_DETECTED
            
        print("[STATE] Obstacle detected!")
        self.speak("Obstacle detected ahead. Please stop.")
        
        # Stop the robot
        self.send_robot_command('stop')
        
        # Wait a moment for user to stop
        time.sleep(2)
        
        # Start scanning
        self.start_scanning()
    
    def start_scanning(self):
        """Initiate environmental scanning"""
        with self.state_lock:
            self.current_state = SystemState.SCANNING
        
        print("[STATE] Starting environmental scan")
        self.speak("Scanning environment")
        
        # Command robot to move forward and scan
        self.send_robot_command('scan_environment')
        
        # Wait for scan results (robot will send back via serial)
        self.wait_for_scan_results()
    
    def wait_for_scan_results(self):
        """Wait for robot to complete scanning and send path data"""
        timeout = 30  # 30 second timeout
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.robot_serial.in_waiting:
                try:
                    line = self.robot_serial.readline().decode('utf-8').strip()
                    data = json.loads(line)
                    
                    if data.get('type') == 'scan_complete':
                        self.handle_scan_complete(data)
                        return
                except Exception as e:
                    print(f"Error reading scan results: {e}")
            
            time.sleep(0.1)
        
        # Timeout - return to following
        print("[ERROR] Scan timeout")
        self.speak("Scan failed. Please be cautious.")
        self.resume_following()
    
    def handle_scan_complete(self, scan_data):
        """Process scan results and provide guidance"""
        with self.state_lock:
            self.current_state = SystemState.PATH_FOUND
        
        safe_direction = scan_data.get('safe_direction', 'center')
        clearance = scan_data.get('clearance', 0)
        
        print(f"[STATE] Path found: {safe_direction}, clearance: {clearance}cm")
        
        # Provide audio guidance
        if safe_direction == 'left':
            self.speak(f"Safe path on the left. Clearance {clearance} centimeters.")
        elif safe_direction == 'right':
            self.speak(f"Safe path on the right. Clearance {clearance} centimeters.")
        elif safe_direction == 'center':
            self.speak(f"Path ahead is clear. You may proceed.")
        else:
            self.speak("No clear path detected. Please turn around.")
        
        # Command robot to return to following position
        time.sleep(2)
        self.send_robot_command('return_to_follow')
        
        # Resume following after a delay
        time.sleep(3)
        self.resume_following()
    
    def resume_following(self):
        """Resume normal following state"""
        with self.state_lock:
            self.current_state = SystemState.FOLLOWING
        
        print("[STATE] Resuming following mode")
        self.obstacle_detected = False
    
    def run(self):
        """Main system loop"""
        print("Starting Intelligent Follower System...")
        
        # Start video stream thread
        video_thread = threading.Thread(target=self.video_stream_thread, daemon=True)
        video_thread.start()
        
        # Start obstacle monitoring thread
        obstacle_thread = threading.Thread(target=self.obstacle_monitoring_thread, daemon=True)
        obstacle_thread.start()
        
        # Wait for first frame
        while self.current_frame is None:
            time.sleep(0.1)
        
        # Initial state
        with self.state_lock:
            self.current_state = SystemState.FOLLOWING
        
        self.speak("System ready. Following mode activated.")
        
        # Main processing loop
        try:
            while True:
                frame = self.process_frame()
                
                if frame is not None:
                    # Display state on frame
                    cv2.putText(frame, f"State: {self.current_state.value}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                               1, (0, 255, 255), 2)
                    
                    cv2.imshow('Intelligent Follower System', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                time.sleep(0.03)  # ~30 FPS
                
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            self.send_robot_command('stop')
            cv2.destroyAllWindows()
            self.robot_serial.close()

if _name_ == "_main_":
    # Configuration
    ESP32_IP = "10.135.22.93"  # Replace with your ESP32-CAM IP
    ROBOT_PORT = "COM3"  # Replace with your robot's serial port
    
    # Initialize and run system
    system = IntelligentFollowerSystem(ESP32_IP, ROBOT_PORT)
    system.run()