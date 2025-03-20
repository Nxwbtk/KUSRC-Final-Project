import cv2, numpy as np, tensorflow as tf, customtkinter as ctk, socket
from mediapipe.python.solutions import drawing_utils as mp_drawing, pose as mp_pose
from PIL import Image

pose_list = ['stop', 'forward', 'backward', 'left', 'right']
default_pose = pose_list[0]

_model = tf.keras.models.load_model('pose.keras')
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

MY_POSE_CONNECTIONS = frozenset([(16, 14), (14, 12), (12, 11), (11,13), (13,15)])

def getValueFromIMP(landmarks):
		R_SHOULDER = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
		R_ELBOW = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
		R_WRIST = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
		L_SHOULDER = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
		L_ELBOW = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
		L_WRIST = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]

		return np.array([[
			R_SHOULDER.x, R_SHOULDER.y, R_SHOULDER.z, R_SHOULDER.visibility,
			R_ELBOW.x, R_ELBOW.y, R_ELBOW.z, R_ELBOW.visibility,
			R_WRIST.x, R_WRIST.y, R_WRIST.z, R_WRIST.visibility,
			L_SHOULDER.x, L_SHOULDER.y, L_SHOULDER.z, L_SHOULDER.visibility,
			L_ELBOW.x, L_ELBOW.y, L_ELBOW.z, L_ELBOW.visibility,
			L_WRIST.x, L_WRIST.y, L_WRIST.z, L_WRIST.visibility
		]])

class PoseDetectionApp(ctk.CTk):
    is_start_dectection = False

    def __init__(self):
        super().__init__()

        self.title('Full-Body Gesture-Controlled Car using Image Processing')
        self.geometry('680x660')
        self.grid_columnconfigure((0), weight=1)

        self.available_cameras = self.get_available_cameras()
        self.camera_var = ctk.StringVar(value=str(self.available_cameras[0] if self.available_cameras else '0'))

        self.camera_label = ctk.CTkLabel(self, text='Select Camera')
        self.camera_label.grid(row=0, column=0, padx=5, pady=5, sticky='w')

        self.camera_dropdown = ctk.CTkComboBox(self, values=[str(cam) for cam in self.available_cameras], command=self.change_camera, variable=self.camera_var, state='readonly')
        self.camera_dropdown.grid(row=0, column=1, padx=5, pady=5, sticky='ew')

        self.label_ip = ctk.CTkLabel(self, text='Server IP:', fg_color='transparent')
        self.label_ip.grid(row=1, column=0, padx=5, pady=5, sticky='w')

        self.textbox_ip = ctk.CTkTextbox(master=self, height=24, corner_radius=0)
        self.textbox_ip.grid(row=1, column=1, pady=5, sticky='nsew')
        self.textbox_ip.insert('0.0', '0.0.0.0')
        self.textbox_ip.edit_modified(False)
        self.textbox_ip.bind("<KeyRelease>", self.on_textbox_edit)

        self.label_port = ctk.CTkLabel(self, text='Server Port:', fg_color='transparent')
        self.label_port.grid(row=2, column=0, padx=5, pady=5, sticky='w')

        self.textbox_port = ctk.CTkTextbox(master=self, height=24, corner_radius=0)
        self.textbox_port.grid(row=2, column=1, pady=5, sticky='nsew')
        self.textbox_port.insert('0.0', '4242')
        self.textbox_port.edit_modified(False)
        self.textbox_port.bind("<KeyRelease>", self.on_textbox_edit)

        self.label_status = ctk.CTkLabel(self, text='Connection Status: Not tested yet.', fg_color='#FFFFFF', text_color='#000000')
        self.label_status.grid(row=3, column=0, padx=5, pady=5, sticky='ew')

        self.test_button = ctk.CTkButton(self, text='Test Connection', command=self.on_click_test)
        self.test_button.grid(row=3, column=1, padx=5, pady=5, sticky='ew')

        self.start_button = ctk.CTkButton(self, text='Start Detection', command=self.on_click_start, fg_color='#038C25', hover_color='#02731E', state='disabled')
        self.start_button.grid(row=4, column=0, padx=5, pady=(5, 0), sticky='ew', columnspan=2)

        self.capture = cv2.VideoCapture(int(self.camera_var.get()))

        self.video_label = ctk.CTkLabel(self, text='')
        self.video_label.grid(row=5, column=0, padx=20, pady=20, sticky='ew', columnspan=2)

        self.update_frame()

        self.protocol('WM_DELETE_WINDOW', self.on_closing)

    def on_textbox_edit(self, event=None):
        if (self.is_start_dectection == True):
            self.sent_udp(self.textbox_ip.get('0.0', 'end'), self.textbox_port.get('0.0', 'end'), default_pose)
        self.label_status.configure(text='Connection Status: Not tested yet.', fg_color='#FFFFFF', text_color='#000000')
        self.start_button.configure(text='Start Detection', fg_color='#038C25', hover_color='#02731E', state='disabled')
        self.test_button.configure(state='normal')
        self.is_start_dectection = False

    def get_available_cameras(self):
        available_cameras = []
        for i in range(3):
            cap = cv2.VideoCapture(i)
            if cap.read()[0]:
                available_cameras.append(i)
            cap.release()
        return available_cameras

    def update_frame(self):
        ret, frame = self.capture.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Running Tensorflow
            if self.is_start_dectection:
                results = pose.process(frame)

                if (results.pose_landmarks):
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, MY_POSE_CONNECTIONS)
                    np_output = _model.predict(getValueFromIMP(results.pose_landmarks.landmark))
                    current_pose = pose_list[np.argmax(np_output)] if np.max(np_output) > 0.6 else default_pose
                else:
                    current_pose = default_pose
                cv2.putText(frame, str(f'{current_pose} {np.max(np_output):.2f}'), (50, 50), 1, 2, (255, 0, 0), 3, cv2.FONT_HERSHEY_SIMPLEX)
                self.sent_udp(self.textbox_ip.get('0.0', 'end'), self.textbox_port.get('0.0', 'end'), current_pose)

            # Get current window size
            window_width = self.winfo_width()
            window_height = self.winfo_height() - 226 if self.winfo_height() > 226 else self.winfo_height()

            # Resize the frame to fit the window size
            resized_frame = cv2.resize(frame, (window_width, window_height))

            img = Image.fromarray(resized_frame)
            ctk_img = ctk.CTkImage(light_image=img, size=(window_width, window_height))

            self.video_label.configure(image=ctk_img)
            self.video_label.image = ctk_img

            self.after(10, self.update_frame)

    def change_camera(self, choice):
        try:
            self.capture.release()
            self.capture = cv2.VideoCapture(int(choice))
        except ValueError:
            print('Invalid camera index. Please enter a number.')

    def on_click_start(self):
        self.is_start_dectection = not self.is_start_dectection
        if self.is_start_dectection:
            self.start_button.configure(self, text='Stop Detection', fg_color='#fc0339', hover_color='#a30023', state='normal')
            self.test_button.configure(state='disabled')
            self.textbox_ip.configure(state='disabled')
            self.textbox_port.configure(state='disabled')
        else:
            self.start_button.configure(self, text='Start Detection', fg_color='#038C25', hover_color='#02731E')
            self.test_button.configure(state='normal')
            self.textbox_ip.configure(state='normal')
            self.textbox_port.configure(state='normal')
            self.sent_udp(self.textbox_ip.get('0.0', 'end'), self.textbox_port.get('0.0', 'end'), default_pose)


    def on_click_test(self):
        reachable = self.check_ip_reachable(self.textbox_ip.get('0.0', 'end'), self.textbox_port.get('0.0', 'end'))
        self.label_status.configure(text=f'Connection Status: {'Server Found!' if reachable else 'Server not Found'}', text_color='#00AA00' if reachable else '#AA0000')
        self.start_button.configure(state='normal' if reachable else 'disabled')

    def on_closing(self):
        self.capture.release()
        self.destroy()

    def check_ip_reachable(self, ip, port):
        host = ip.strip()
        host_port = int(port.strip())

        try:
            socket.inet_pton(socket.AF_INET, host)
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            message = default_pose.encode()
            sock.sendto(message, (host, host_port))
            sock.settimeout(2)
            data, server = sock.recvfrom(1024)
            sock.close()
            return True
        except :
            sock.close()
            return False

    def sent_udp(self, ip, port, pos):
        host = ip.strip()
        host_port = int(port.strip())

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.sendto(pos.encode(), (host, host_port))
        except:
            print('error')
        finally:
            sock.close()

if __name__ == '__main__':
    app = PoseDetectionApp()
    app.mainloop()
