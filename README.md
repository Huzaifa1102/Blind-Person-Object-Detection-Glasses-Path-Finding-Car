# Intelligent Follower Robot System

This repository contains the full software stack for an **Intelligent Follower Robot**, a sophisticated IoT and AI project designed for autonomous person tracking and environmental guidance. The system integrates real-time computer vision (YOLOv8 + DeepSORT), embedded sensor arrays, and a multi-state autonomous controller.

---

## 🚀 System Architecture

The project is divided into three primary layers:

### 1. Vision & Control Layer (Python)

* **Object Detection:** Uses **YOLOv8** to identify people with high confidence.
* **Target Tracking:** Implements **DeepSORT** to maintain a persistent lock on a specific individual, even in crowded environments.
* **Guidance System:** Features a text-to-speech (TTS) engine that provides real-time audio navigation instructions such as "Safe path on the right".
* **State Machine:** Manages transitions between `FOLLOWING`, `OBSTACLE_DETECTED`, `SCANNING`, `PATH_FOUND`, and `IDLE`.

### 2. Simulation Environment (PyQt5)

A comprehensive 2D simulation environment allows for testing the logic without physical hardware.

* **Virtual Physics:** Simulates differential drive kinematics and LiDAR/Ultrasonic ray-casting.
* **Sensor Mapping:** Virtual microwave sensors on "smart glasses" and LiDAR on the robot are simulated to detect environmental boundaries.
* **Live Dashboard:** Displays real-time telemetry including speed, turn rate, and sensor distances.

### 3. Hardware Layer (ESP32 / Arduino)

* 
**`Camera.ino`**: Configures an **ESP32-CAM** to stream low-latency JPEG frames via an MJPEG server.


* 
**`RCLidar.ino`**: Manages the robot chassis, motor control via L298N, and parses data from a serial **LiDAR sensor**.


* 
**`Glasses.ino`**: Implements a wearable obstacle detection unit using an **HC-SR04 ultrasonic sensor** and a web-based data server.



---

## 🛠 Tech Stack

* 
**Languages:** Python 3.x, C++ (Arduino/ESP32).


* **AI Models:** YOLOv8 (Ultralytics), DeepSORT.
* **GUI & Sim:** PyQt5, OpenCV, NumPy.
* 
**Communication:** WebSockets, HTTP REST (JSON), Serial.


* 
**Hardware:** ESP32, ESP32-CAM, L298N Motor Driver, HC-SR04, TF-Luna LiDAR.



---

## 📂 File Structure

| File | Description |
| --- | --- |
| `Model.py` | The core AI controller. Handles YOLO/DeepSORT and serial robot commands. |
| `Simulation.py` | PyQt5-based 2D simulator for testing the autonomous logic. |
| `Camera.ino` | Firmware for the ESP32-CAM video streaming node.

 |
| `RCLidar.ino` | Main robot firmware including motor logic and LiDAR parsing.

 |
| `Glasses.ino` | Wearable sensor firmware for auxiliary obstacle detection.

 |

---

## ⚙️ Installation & Usage

### Prerequisites

```bash
pip install ultralytics deep-sort-realtime opencv-python numpy requests pyttsx3 pyserial PyQt5

```

### Running the Simulation

To test the logic and UI without a robot:

```bash
python Simulation.py

```

### Running the Real Robot

1. 
**Flash Firmware**: Upload the `.ino` files to your respective ESP32 boards using the Arduino IDE.


2. 
**Configure Network**: Ensure all ESP32s and your laptop are on the same WiFi network.


3. **Update IP**: Edit the `ESP32_IP` in `Model.py` with the IP address printed in the Serial Monitor.
4. **Launch**:
```bash
python Model.py

```



---

## 🤖 Logic Flow

1. **FOLLOWING**: The robot tracks and follows the target person.
2. **OBSTACLE_DETECTED**: If sensors detect a blockage, the robot stops and alerts the user via audio.
3. **SCANNING**: The robot moves in front of the user and performs a 360° environment sweep.
4. **PATH_FOUND**: The system identifies the best clearance and providing audio instructions to the user.
5. **RESUME**: Once a path is clear or the user moves, the robot returns to the "follow" position.

---

## 👨‍💻 Author

**Huzaifa Mudassar** *BS Artificial Intelligence Student*
