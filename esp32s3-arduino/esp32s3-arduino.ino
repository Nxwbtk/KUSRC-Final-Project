#include <WiFi.h>
#include <PubSubClient.h>
#define left1 11   //in1
#define left2 12   //in2
#define right1 13  //in3
#define right2 14  //in4
#define ON_LEFT 80
#define ON_RIGHT 80

const char* ssid = "snakeCase_2G";
const char* password = "passwordFor0x2D6Room";
const char* mqttServer = "192.168.1.146";

const int mqttPort = 1883;
const char* mqttUser = "rabbitmq";
const char* mqttPassword = "password";
const char* mqttTopic = "direction";

WiFiClient espClient;
PubSubClient client(espClient);

void setup() {
  rgbLedWrite(RGB_BUILTIN, 0, 0, 0);
  Serial.begin(115200);
  WiFi.begin(ssid, password);

  ledcAttach(left1, 5000, 8);
  ledcAttach(left2, 5000, 8);
  ledcAttach(right1, 5000, 8);
  ledcAttach(right2, 5000, 8);

  stop();

  client.setServer(mqttServer, mqttPort);
  client.setCallback(callback);
}

void loop() {
  while (WiFi.status() != WL_CONNECTED)
    reconnect_wifi();
  while (!client.connected())
    reconnect_mqtt();
  client.loop();
}

void callback(char* topic, byte* payload, unsigned int length) {
  Serial.print("Direction: ");
  char a;
  char msg[10];
  bzero(msg, 10);
  if (length > 0)
    memcpy(msg, payload, length);
  else
    memcpy(msg, "stop", length);
  Serial.println(msg);

  a = msg[0];
  if (a == 's') stop();
  if (a == 'f') forward();
  if (a == 'b') backward();
  if (a == 'l') turn_left();
  if (a == 'r') turn_right();
}

void stop() {
  ledcWrite(left1, 0);
  ledcWrite(left2, 0);
  ledcWrite(right1, 0);
  ledcWrite(right2, 0);
}

void forward() {
  ledcWrite(left1, ON_LEFT + 5);
  ledcWrite(left2, 0);
  ledcWrite(right1, ON_RIGHT);
  ledcWrite(right2, 0);
}

void backward() {
  ledcWrite(left1, 0);
  ledcWrite(left2, ON_LEFT - 5);
  ledcWrite(right1, 0);
  ledcWrite(right2, ON_RIGHT);
}

void turn_left() {
  ledcWrite(left1, ON_LEFT);
  ledcWrite(left2, 0);
  ledcWrite(right1, 0);
  ledcWrite(right2, ON_RIGHT);
}

void turn_right() {
  ledcWrite(left1, 0);
  ledcWrite(left2, ON_LEFT);
  ledcWrite(right1, ON_RIGHT);
  ledcWrite(right2, 0);
}

void reconnect_wifi() {
  rgbLedWrite(RGB_BUILTIN, 64, 0, 0);
  delay(1000);
  Serial.print(".");
  rgbLedWrite(RGB_BUILTIN, 0, 0, 0);
}

void reconnect_mqtt() {
  Serial.print("Connecting to RabbitMQ MQTT...");
  if (client.connect("ESP32_Client", mqttUser, mqttPassword)) {
    rgbLedWrite(RGB_BUILTIN, 0, 64, 0);
    Serial.println("Connected!");
    client.subscribe(mqttTopic);
    delay(500);
    rgbLedWrite(RGB_BUILTIN, 0, 0, 0);
  } else {
    rgbLedWrite(RGB_BUILTIN, 52, 12, 0);
    Serial.print("Failed, rc=");
    Serial.print(client.state());
    Serial.println(" Retrying in 5 seconds...");
    delay(2000);
    rgbLedWrite(RGB_BUILTIN, 0, 0, 0);
    delay(2000);
  }
}