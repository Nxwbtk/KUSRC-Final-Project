#include <WiFi.h>
#include <WiFiUdp.h>
#include <Adafruit_NeoPixel.h>  // optional library, you can delete this line
#include "neopixel_map.h"

// servo motor pins
#define servo_motor_left D0
#define servo_motor_right D1

// Servo Motor PWM Configuration
// - The servo operates using Pulse Width Modulation (PWM)
// - Set duty cycle values as follows:
//   - 77: Neutral position (stop)
//   - 20: Maximum clockwise rotation
//   - 132: Maximum counterclockwise rotation
// - PWM Settings:
//   - Frequency: 50 Hz
//   - Resolution: 10-bit
// - Tolerance Values:
//   - `movement_duty_tolerance`: 12 (for movement adjustments)
//   - `turning_duty_tolerance`: 8 (for turning adjustments)
// - Delay: 100 ms between moving change
#define freq 50
#define resolution 10
#define standby_duty 77
#define movement_duty_tolerance 12
#define turning_duty_tolerance 8
#define delay_ms 100

// NeoPixel Setup
// - This section configures an Adafruit NeoPixel LED strip.
// - If you do not have a NeoPixel 1x8 strip or the Adafruit_NeoPixel library,
//   you can comment out this section.
// - Configuration:
//   - `rgb_pin`: Defines the pin connected to the NeoPixel strip (D2).
//   - `NUMPIXELS`: Number of LEDs in the strip (16).
//   - NeoPixel object is initialized only if the Adafruit_NeoPixel library is included.
#define rgb_pin D2
#ifdef ADAFRUIT_NEOPIXEL_H
#define NUMPIXELS 16
Adafruit_NeoPixel pixels(NUMPIXELS, rgb_pin, NEO_GRB + NEO_KHZ800);
#endif

const char* ssid = "ESP32C3-FB-GCCUIP";
const char* password = "password";
const int udpPort = 4242;

char curr_movement = 's';
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

#ifdef ADAFRUIT_NEOPIXEL_H
  pixels.clear();
  for (int i = 0; i < NUMPIXELS; i++) {
    pixels.setPixelColor(i, pixels.Color(140, 60, 0));
    pixels.show();
    delay(100);
  }

  pixels.clear();
  pixels.show();
  delay(500);

  for (int j = 0; j < 3; j++) {
    for (int i = 0; i < NUMPIXELS; i++) {
      pixels.setPixelColor(i, pixels.Color(0,200, 0));
    }
    pixels.show();
    delay(100);

    pixels.clear();
    pixels.show();
    delay(200);
  }
  delay(500);
#endif

  xTaskCreatePinnedToCore(movement_controller, "movement_controller", 4096, NULL, 1, NULL, 0);
  xTaskCreatePinnedToCore(rgb_controller, "rgb_controller", 4096, NULL, 1, NULL, 0);
}

void loop() {
  // do nothing
}

void rgb_controller(void* pvParameters) {
  static uint8_t frame_count = 0;

  while (1) {
    if (curr_movement != last_movement) {
      frame_count = 0;
    }

#ifdef ADAFRUIT_NEOPIXEL_H
    rgb_palette_t cur_color_palette;
    rgb_color_t* frame_color_palette;

    if (curr_movement == 's')
      cur_color_palette = stop_palette;
    else if (curr_movement == 'f')
      cur_color_palette = forward_palette;
    else if (curr_movement == 'b')
      cur_color_palette = backward_palette;
    else if (curr_movement == 'l')
      cur_color_palette = left_palette;
    else if (curr_movement == 'r')
      cur_color_palette = right_palette;

    frame_color_palette = cur_color_palette.render_color_palette[frame_count];

    pixels.clear();
    for (int i = 0; i < 16; i++) {
      pixels.setPixelColor(i, pixels.Color(frame_color_palette[i].r, frame_color_palette[i].g, frame_color_palette[i].b));
    }
    pixels.show();
    if (frame_count < 4) {
      frame_count += 1;
    } else {
      frame_count = 0;
    }
#endif
    vTaskDelay(100 / portTICK_PERIOD_MS);
  }
}

void movement_controller(void* pvParameters) {
  while (1) {
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
      curr_movement = incomingPacket[0];

      move(incomingPacket);
    }
    vTaskDelay(50 / portTICK_PERIOD_MS);
  }
}

void move(char* str) {
  if (last_movement != str[0]) {
    last_movement = str[0];
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
}
