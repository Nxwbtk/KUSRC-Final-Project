#include <WiFi.h>
#include <WiFiUdp.h>

// servo motor pins
# define servo_motor_left D0
# define servo_motor_right D1
// servo motor pwm
# define freq 50
# define resolution 10
# define standby_duty 76
# define movement_duty_tolerance  4
# define turning_duty_tolerance 4
# define delay_ms 300

# define rgb_pin D2   //optional

const char* ssid = "ESP32C3-FB-GCCUIP";
const char* password = "password";
const int udpPort = 4242;
char last_movement = 's';

WiFiUDP udp;
char incomingPacket[10];

void setup() {
    delay(500);
    Serial.begin(115200);
    WiFi.softAP(ssid, password);
    Serial.println("WiFi AP started");
    Serial.print("IP Address: ");
    Serial.println(WiFi.softAPIP());

    udp.begin(udpPort);
    Serial.printf("UDP server started on port %d\n", udpPort);
    delay(500);

    ledcAttach(servo_motor_left, freq, resolution);
    ledcAttach(servo_motor_right, freq, resolution);
    ledcWrite(servo_motor_left, standby_duty);
    ledcWrite(servo_motor_right, standby_duty);
}

void loop() {
    int packetSize = udp.parsePacket();
    if (packetSize && packetSize < 9) {
        int len = udp.read(incomingPacket, 10);
        if (len > 0) {
            incomingPacket[len] = '\0';
        }
        Serial.printf("Message: %s\n", incomingPacket);
        
        // return respons
        udp.beginPacket(udp.remoteIP(), udp.remotePort());
        udp.print("Yes Sir! I will ");
        udp.print(incomingPacket);
        udp.endPacket();

        // movement controller
        move(incomingPacket);
    }
}

void move(char *str){
  if (last_movement != str[0]) {
      ledcWrite(servo_motor_left, standby_duty);
      ledcWrite(servo_motor_right, standby_duty);
      delay(delay_ms);
  } 

  if (str[0] == 's') {
    ledcWrite(servo_motor_left, standby_duty);
    ledcWrite(servo_motor_right, standby_duty);
  } else if (str[0] == 'f') {
    ledcWrite(servo_motor_left, standby_duty - movement_duty_tolerance);
    ledcWrite(servo_motor_right, standby_duty + movement_duty_tolerance);
  } else if (str[0] == 'b') {
    ledcWrite(servo_motor_left, standby_duty + movement_duty_tolerance);
    ledcWrite(servo_motor_right, standby_duty - movement_duty_tolerance);
  } else if (str[0] == 'l') {
    ledcWrite(servo_motor_left, standby_duty - turning_duty_tolerance);
    ledcWrite(servo_motor_right, standby_duty - turning_duty_tolerance);
  } else if (str[0] == 'r') {
    ledcWrite(servo_motor_left, standby_duty + turning_duty_tolerance);
    ledcWrite(servo_motor_right, standby_duty + turning_duty_tolerance);
  }
  last_movement = str[0];
}
