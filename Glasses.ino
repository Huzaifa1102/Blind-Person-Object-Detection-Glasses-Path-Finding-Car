#include <WiFi.h>
#include <WebServer.h>

// ===========================
// WIFI CONFIGURATION
// ===========================
const char* ssid = "Faizan";
const char* password = "123456789";

// ===========================
// PIN DEFINITIONS
// ===========================
#define TRIG_PIN  5
#define ECHO_PIN  18

WebServer server(80);
volatile long currentDistance = 0;

// Measure Distance using HC-SR04
long readSonar() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);

  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  long duration = pulseIn(ECHO_PIN, HIGH, 30000); // 30ms timeout

  if (duration == 0) return 999;

  long distance = duration * 0.034 / 2;

  if (distance > 400 || distance <= 0) return 999;
  return distance;
}

// ===========================
// JSON ENDPOINT
// ===========================
void handleData() {
  String json = "{\"distance\": " + String(currentDistance) + "}";
  server.send(200, "application/json", json);
}

// ===========================
// HTML PAGE
// ===========================
void handleRoot() {
  String page = R"====(
    <html>
    <head>
      <title>Distance Monitor</title>
      <style>
        body {
          font-family: Arial;
          text-align: center;
          margin-top: 50px;
        }
        .box {
          font-size: 40px;
          padding: 20px;
          display: inline-block;
          border: 2px solid #333;
          border-radius: 10px;
        }
      </style>
      <script>
        function updateData() {
          fetch('/data')
            .then(response => response.json())
            .then(data => {
              document.getElementById('dist').innerHTML = data.distance + " cm";
            });
        }

        setInterval(updateData, 200); // update every 200ms
      </script>
    </head>
    <body>
      <h2>HC-SR04 Distance Reading</h2>
      <div class="box" id="dist">Loading...</div>
    </body>
    </html>
  )====";

  server.send(200, "text/html", page);
}

void setup() {
  Serial.begin(115200);

  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);

  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("\nConnected!");
  Serial.print("IP Address: ");
  Serial.println(WiFi.localIP());

  server.on("/", handleRoot);
  server.on("/data", handleData);
  server.begin();
}

void loop() {
  server.handleClient();
  currentDistance = readSonar();
  delay(100);
}
