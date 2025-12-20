# KUSRC-Final-Project

## Setup Instructions

### ESP32C3 Car Board Schematic
<img height="500" alt="main_schem" src="https://github.com/user-attachments/assets/d03f7eb8-6bdc-4f7d-b377-ae1955c94d68" />
<img height="500" alt="2025-12-20 15_51_44-Presentation1 - PowerPoint" src="https://github.com/user-attachments/assets/40c07beb-6079-4e31-b7b2-d7012e74bd46" />


### ESP32C3 (Arduino IDE)
1. Install Adafruit NeoPixel by Adafruit Library Manager on Arduino IDE
2. Install ESP32 by Espressif System on Boards Manager
3. Upload code into XIAO ESP32C3
4. Check Serial Monitor output (115200 baud)
<img width="731" height="473" alt="2025-12-20 15_05_38-esp32c3_socket_server _ Arduino IDE 2 3 6" src="https://github.com/user-attachments/assets/d33e9d69-83bd-4b6c-8d04-ccb9a542ec9b" />

5. Connect to WiFi `ESP32C3-FB-GCCUIP` and password `password` and note the IP Address `192.168.4.1:4242`

### Python Camera Software Setup

1. to to path python-desktop

   ```sh
   cd .\python-desktop\
   ```

3. Initialize the virtual environment:

   ```sh
   python -m venv venv
   ```

4. Activate the virtual environment:

   ```sh
   .\venv\Scripts\activate
   ```

5. Install requirement package:

   ```sh
   pip install -r requirements.txt
   ```

6. Run the Camera python software:

   ```sh
   python .\main.py
   ```

7. Select camera from your laptop or pc, enter Server IP, enter Server port
<img height="500" alt="2025-12-20 16_07_53-Full-Body Gesture Controlled Car using Image Processing" src="https://github.com/user-attachments/assets/088428de-f464-4e5a-a293-be49bff001aa" />

8. Click `Test Connection` button (if it can connect, the system will show Connected)
<img width="240" height="316" alt="2025-12-20 16_11_03-Clipboard" src="https://github.com/user-attachments/assets/3ce506cf-2a57-4663-b6b0-9704b642697f" />

9. Clict `Start Detection` button to start Body Gesture detecting
10. Following the Control Methods
<img height="500" alt="2025-12-20 16_15_40-Word _ Word _ Microsoft Teams" src="https://github.com/user-attachments/assets/957ded74-f94d-414c-8aa4-61e9f651d87b" />

11. Click `Stop Detection` to stop the Body Gesture detecting
<img width="240" height="335" alt="image" src="https://github.com/user-attachments/assets/42a356e4-e86b-4e39-9b65-c760e3ff7a5f" />
