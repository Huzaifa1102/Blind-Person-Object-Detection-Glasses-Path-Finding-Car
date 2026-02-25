#include <WiFi.h>
#include <WebServer.h>

// --- WiFi Credentials ---
const char* ssid = "Faizan";
const char* password = "123456789";

// --- Pin Definitions (UPDATED) ---
// Speed Control Pins (Enable Pins)
const int ENA = 26; 
const int ENB = 27;

// Motor A (Right Side)
const int IN1 = 5;
const int IN2 = 18;
// Motor B (Left Side)
const int IN3 = 19;
const int IN4 = 21;

// LiDAR Definitions (Serial2)
// User specified: TX->32, RX->33. 
// Note: LiDAR TX connects to ESP RX (32), LiDAR RX connects to ESP TX (33)
#define RXD2 32 
#define TXD2 33
int distance = 0; 

// --- Server & Logic ---
WebServer server(80);
bool autopilot = false;

// --- HTML Page ---
const char index_html[] PROGMEM = R"rawliteral(
<!DOCTYPE HTML><html>
<head>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body { font-family: Arial; text-align: center; margin:0px auto; padding-top: 30px; background-color: #f4f4f4;}
    .button { padding: 10px 20px; font-size: 24px; text-align: center; cursor: pointer; outline: none; color: #fff; background-color: #4CAF50; border: none; border-radius: 15px; margin: 5px; box-shadow: 0 5px #999; }
    .button:active { background-color: #3e8e41; box-shadow: 0 2px #666; transform: translateY(4px); }
    .stop { background-color: #d9534f; }
    .auto { background-color: #008CBA; }
    .data { font-size: 30px; color: #333; font-weight: bold; padding: 20px; }
  </style>
</head>
<body>
  <h1>ESP32 LiDAR Car</h1>
  <div class="data">Distance: <span id="dist">0</span> cm</div>
  
  <button class="button" onmousedown="move('forward')" onmouseup="move('stop')" ontouchstart="move('forward')" ontouchend="move('stop')">Forward</button><br>
  
  <button class="button" onmousedown="move('s_left')" onmouseup="move('stop')" ontouchstart="move('s_left')" ontouchend="move('stop')">Slight L</button>
  <button class="button" onmousedown="move('left')" onmouseup="move('stop')" ontouchstart="move('left')" ontouchend="move('stop')">Left</button>
  <button class="button" onmousedown="move('right')" onmouseup="move('stop')" ontouchstart="move('right')" ontouchend="move('stop')">Right</button>
  <button class="button" onmousedown="move('s_right')" onmouseup="move('stop')" ontouchstart="move('s_right')" ontouchend="move('stop')">Slight R</button><br>
  
  <button class="button" onmousedown="move('backward')" onmouseup="move('stop')" ontouchstart="move('backward')" ontouchend="move('stop')">Backward</button><br><br>
  
  <button class="button stop" onclick="move('stop')">STOP ALL</button>
  <button class="button auto" onclick="toggleAuto()">Toggle Autopilot</button>
  <p>Autopilot Status: <span id="autoStatus">OFF</span></p>

<script>
  setInterval(function() {
    fetch('/read_dist').then(response => response.text()).then(data => {
      document.getElementById("dist").innerHTML = data;
    });
  }, 100);

  function move(action) {
    fetch('/action?go=' + action);
  }

  function toggleAuto() {
    fetch('/toggle_auto').then(response => response.text()).then(data => {
      document.getElementById("autoStatus").innerHTML = data;
    });
  }
</script>
</body>
</html>
)rawliteral";

// --- Motor Control Functions ---
void stopMotors() {
  digitalWrite(IN1, LOW); digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW); digitalWrite(IN4, LOW);
}

void moveForward() {
  digitalWrite(IN1, HIGH); digitalWrite(IN2, LOW);
  digitalWrite(IN3, HIGH); digitalWrite(IN4, LOW);
}

void moveBackward() {
  digitalWrite(IN1, LOW); digitalWrite(IN2, HIGH);
  digitalWrite(IN3, LOW); digitalWrite(IN4, HIGH);
}

void turnLeft() {
  digitalWrite(IN1, HIGH); digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW); digitalWrite(IN4, HIGH);
}

void turnRight() {
  digitalWrite(IN1, LOW); digitalWrite(IN2, HIGH);
  digitalWrite(IN3, HIGH); digitalWrite(IN4, LOW);
}

void slightLeft() {
  digitalWrite(IN1, HIGH); digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW); digitalWrite(IN4, LOW);
}

void slightRight() {
  digitalWrite(IN1, LOW); digitalWrite(IN2, LOW);
  digitalWrite(IN3, HIGH); digitalWrite(IN4, LOW);
}

// --- LiDAR Function ---
void readLidar() {
  if (Serial2.available() >= 9) {
    if (Serial2.read() == 0x59) {
      if (Serial2.read() == 0x59) {
        int distL = Serial2.read();
        int distH = Serial2.read();
        // Skip remaining bytes (strength, mode, etc.)
        for(int i=0; i<5; i++) Serial2.read(); 
        distance = distL + (distH * 256);
      }
    }
  }
}

// --- Web Handlers ---
void handleRoot() { server.send(200, "text/html", index_html); }
void handleDistance() { server.send(200, "text/plain", String(distance)); }
void handleToggleAuto() {
  autopilot = !autopilot;
  stopMotors();
  server.send(200, "text/plain", autopilot ? "ON" : "OFF");
}

void handleAction() {
  if (autopilot) return;
  String go = server.arg("go");
  if (go == "forward") moveForward();
  else if (go == "backward") moveBackward();
  else if (go == "left") turnLeft();
  else if (go == "right") turnRight();
  else if (go == "s_left") slightLeft();
  else if (go == "s_right") slightRight();
  else stopMotors();
  server.send(200, "text/plain", "OK");
}

// --- Setup & Loop ---
void setup() {
  Serial.begin(115200);
  
  // LiDAR Serial Init: RX=32, TX=33
  Serial2.begin(115200, SERIAL_8N1, RXD2, TXD2);

  pinMode(ENA, OUTPUT); pinMode(ENB, OUTPUT);
  pinMode(IN1, OUTPUT); pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT); pinMode(IN4, OUTPUT);

  // Set Motors to Max Speed
  digitalWrite(ENA, HIGH); 
  digitalWrite(ENB, HIGH);
  
  stopMotors();

  Serial.println("Connecting to WiFi...");
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  
  Serial.println("\nWiFi Connected.");
  Serial.print("IP Address: ");
  Serial.println(WiFi.localIP());

  server.on("/", handleRoot);
  server.on("/read_dist", handleDistance);
  server.on("/toggle_auto", handleToggleAuto);
  server.on("/action", handleAction);
  
  server.begin();
}

void loop() {
  server.handleClient();
  readLidar();

  if (autopilot) {
    if (distance > 0 && distance < 20) {
      turnRight(); // Obstacle Avoidance
    } else {
      moveForward();
    }
    delay(50);
  }
}