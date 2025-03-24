import cv2
import numpy as np
import tensorflow as tf
import customtkinter as ctk
import socket
from mediapipe.python.solutions import drawing_utils as mp_drawing, pose as mp_pose
from PIL import Image

# Set customtkinter appearance to light with soft colors
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

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
    is_start_detection = False
    current_pose = default_pose
    confidence = 0.0

    def __init__(self):
        super().__init__()

        # Configure window
        self.title('Full-Body Gesture Controlled Car using Image Processing')
        self.geometry('880x580')
        self.minsize(800, 500)
        
        # Configure grid layout (2x2)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0)
        self.grid_rowconfigure(0, weight=1)
        
        # Create main frame
        self.main_frame = ctk.CTkFrame(self, fg_color="#F5F5F7", corner_radius=0)
        self.main_frame.grid(row=0, column=0, columnspan=2, sticky="nsew")
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)

        self.video_frame = ctk.CTkFrame(self.main_frame, corner_radius=6)
        self.video_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        self.video_frame.grid_columnconfigure(0, weight=1)
        self.video_frame.grid_rowconfigure(0, weight=1)
        
        # Video display
        self.video_display = ctk.CTkLabel(self.video_frame, text='')
        self.video_display.grid(row=0, column=0, sticky="nsew")


        # Create right control panel
        self.control_panel = ctk.CTkFrame(self.main_frame, width=200, corner_radius=6)
        self.control_panel.grid(row=0, column=1, padx=(0, 20), pady=20, sticky="nsew")
        
        # Control Panel Content
        self.control_label = ctk.CTkLabel(
            self.control_panel,
            text="Control",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.control_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        # Icons frame
        self.icons_frame = ctk.CTkFrame(self.control_panel, fg_color="transparent")
        self.icons_frame.grid(row=1, column=0, padx=20, pady=10)
        
        # About section
        self.about_label = ctk.CTkLabel(
            self.control_panel,
            text="About?",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.about_label.grid(row=2, column=0, padx=20, pady=(20, 5), sticky="w")

        self.about_text = ctk.CTkLabel(
            self.control_panel,
            text="This application allows users to control an RC car freely and enjoyably by using body gestures as the primary input",
            wraplength=160
        )
        self.about_text.grid(row=3, column=0, padx=20, pady=(0, 20))

        # Camera selection
        self.camera_label = ctk.CTkLabel(
            self.control_panel,
            text="Select Camera",
            font=ctk.CTkFont(size=14)
        )
        self.camera_label.grid(row=4, column=0, padx=20, pady=(20, 5), sticky="w")

        # Get available cameras
        self.available_cameras = self.get_available_cameras()
        self.camera_var = ctk.StringVar(value=str(self.available_cameras[0] if self.available_cameras else '0'))
        
        self.camera_dropdown = ctk.CTkComboBox(
            self.control_panel,
            values=[str(cam) for cam in self.available_cameras],
            command=self.change_camera,
            variable=self.camera_var,
            width=160
        )
        self.camera_dropdown.grid(row=5, column=0, padx=20, pady=(0, 20))

        # Server settings
        self.server_label = ctk.CTkLabel(
            self.control_panel,
            text="Server IP:",
            font=ctk.CTkFont(size=14)
        )
        self.server_label.grid(row=6, column=0, padx=20, pady=(20, 5), sticky="w")

        self.entry_ip = ctk.CTkEntry(self.control_panel, width=160)
        self.entry_ip.grid(row=7, column=0, padx=20, pady=(0, 20))
        self.entry_ip.insert(0, '0.0.0.0')
        self.entry_ip.bind("<KeyRelease>", self.on_entry_edit)

        self.port_label = ctk.CTkLabel(
            self.control_panel,
            text="Server Port:",
            font=ctk.CTkFont(size=14)
        )
        self.port_label.grid(row=8, column=0, padx=20, pady=(0, 5), sticky="w")

        self.entry_port = ctk.CTkEntry(self.control_panel, width=160)
        self.entry_port.grid(row=9, column=0, padx=20, pady=(0, 20))
        self.entry_port.insert(0, '4242')
        self.entry_port.bind("<KeyRelease>", self.on_entry_edit)

        # Buttons
        self.test_button = ctk.CTkButton(
            self.control_panel,
            text="Test Connection",
            command=self.on_click_test,
            width=160
        )
        self.test_button.grid(row=10, column=0, padx=20, pady=(0, 10))

        self.connection_status = ctk.CTkLabel(
            self.control_panel,
            text="Not Test Yet.",
            font=ctk.CTkFont(size=12)
        )
        self.connection_status.grid(row=11, column=0, padx=20, pady=(0, 10))

        self.detection_button = ctk.CTkButton(
            self.control_panel,
            text="Start Detection",
            command=self.on_click_start,
            width=160,
            fg_color="#4CAF50",
            hover_color="#388E3C",
            state='disabled'
        )
        self.detection_button.grid(row=12, column=0, padx=20, pady=(0, 20))

        self.capture = cv2.VideoCapture(0)  # Start with camera 0
        if not self.capture.isOpened():
            print("Error: Could not open camera")
            # Try fallback to camera 1 if camera 0 fails
            self.capture = cv2.VideoCapture(1)
        
        # Start video update
        self.update_frame()
        
        # Set close handler
        self.protocol('WM_DELETE_WINDOW', self.on_closing)
        # Create right control panel
        self.control_panel = ctk.CTkFrame(self.main_frame, width=200, corner_radius=6, fg_color="#EBEBEB")
        self.control_panel.grid(row=0, column=1, padx=(0, 20), pady=20, sticky="nsew")
        
        # Control Panel Content
        self.control_label = ctk.CTkLabel(
            self.control_panel,
            text="Control",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.control_label.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="w")

        # Icons placeholder (if needed)
        # Add your car and person icons here
        
        # About section
        self.about_label = ctk.CTkLabel(
            self.control_panel,
            text="About?",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.about_label.grid(row=2, column=0, padx=20, pady=(20, 5), sticky="w")

        self.about_text = ctk.CTkLabel(
            self.control_panel,
            text="This application allows users to control an RC car freely and enjoyably by using body gestures as the primary input",
            wraplength=160,
            justify="left"
        )
        self.about_text.grid(row=3, column=0, padx=20, pady=(0, 20), sticky="w")

        # Camera selection
        self.camera_label = ctk.CTkLabel(
            self.control_panel,
            text="Select Camera",
            font=ctk.CTkFont(size=12)
        )
        self.camera_label.grid(row=4, column=0, padx=20, pady=(0, 5), sticky="w")

        # Get available cameras
        self.available_cameras = self.get_available_cameras()
        self.camera_var = ctk.StringVar(value=str(self.available_cameras[0] if self.available_cameras else '0'))
        
        self.camera_dropdown = ctk.CTkComboBox(
            self.control_panel,
            values=[str(cam) for cam in self.available_cameras],
            command=self.change_camera,
            variable=self.camera_var,
            width=160,
            height=28
        )
        self.camera_dropdown.grid(row=5, column=0, padx=20, pady=(0, 10), sticky="w")

        # Server IP and Port entries
        self.server_label = ctk.CTkLabel(
            self.control_panel,
            text="Server IP:",
            font=ctk.CTkFont(size=12)
        )
        self.server_label.grid(row=6, column=0, padx=20, pady=(0, 5), sticky="w")

        self.entry_ip = ctk.CTkEntry(self.control_panel, width=160, height=28)
        self.entry_ip.grid(row=7, column=0, padx=20, pady=(0, 10), sticky="w")
        self.entry_ip.insert(0, '0.0.0.0')
        self.entry_ip.bind("<KeyRelease>", self.on_entry_edit)

        self.port_label = ctk.CTkLabel(
            self.control_panel,
            text="Server Port:",
            font=ctk.CTkFont(size=12)
        )
        self.port_label.grid(row=8, column=0, padx=20, pady=(0, 5), sticky="w")

        self.entry_port = ctk.CTkEntry(self.control_panel, width=160, height=28)
        self.entry_port.grid(row=9, column=0, padx=20, pady=(0, 10), sticky="w")
        self.entry_port.insert(0, '4242')
        self.entry_port.bind("<KeyRelease>", self.on_entry_edit)

        # Test Connection button
        self.test_button = ctk.CTkButton(
            self.control_panel,
            text="Test Connection",
            command=self.on_click_test,
            width=160,
            height=28
        )
        self.test_button.grid(row=10, column=0, padx=20, pady=(0, 5))

        # Connection status
        self.connection_status = ctk.CTkLabel(
            self.control_panel,
            text="Not Test Yet.",
            font=ctk.CTkFont(size=12)
        )
        self.connection_status.grid(row=11, column=0, padx=20, pady=(0, 5))

        # Start/Stop Detection button
        self.detection_button = ctk.CTkButton(
            self.control_panel,
            text="Start Detection",
            command=self.on_click_start,
            width=160,
            height=28,
            fg_color="#4CAF50",
            hover_color="#388E3C",
            state='disabled'
        )
        self.detection_button.grid(row=12, column=0, padx=20, pady=(0, 20))

    def on_entry_edit(self, event=None):
        if self.is_start_detection:
            self.sent_udp(self.entry_ip.get(), self.entry_port.get(), default_pose)
        self.connection_status.configure(text="Not Test Yet.", text_color="#555555")
        self.detection_button.configure(state='disabled')
        self.test_button.configure(state='normal')
        self.is_start_detection = False

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

            # Running pose detection when active
            if self.is_start_detection:
                results = pose.process(frame)

                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, 
                        results.pose_landmarks, 
                        MY_POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 128, 255), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(255, 128, 0), thickness=2)
                    )
                    np_output = _model.predict(getValueFromIMP(results.pose_landmarks.landmark))
                    self.confidence = np.max(np_output)
                    self.current_pose = pose_list[np.argmax(np_output)] if self.confidence > 0.6 else default_pose
                else:
                    self.current_pose = default_pose
                    self.confidence = 0.0
                
                self.sent_udp(self.entry_ip.get(), self.entry_port.get(), self.current_pose)

            # Get current window size for video display
            frame_width = self.video_frame.winfo_width()
            frame_height = self.video_frame.winfo_height()
            
            if frame_width > 10 and frame_height > 10:  # Avoid resize when window is initializing
                # Calculate aspect ratio preserving dimensions
                img_h, img_w = frame.shape[:2]
                aspect = img_w / img_h
                
                if frame_width / frame_height > aspect:
                    # Window is wider than image
                    new_height = frame_height
                    new_width = int(new_height * aspect)
                else:
                    # Window is taller than image
                    new_width = frame_width
                    new_height = int(new_width / aspect)
                
                # Resize the frame to fit while preserving aspect ratio
                frame = cv2.resize(frame, (new_width, new_height))
                
                # Convert to PhotoImage
                image = Image.fromarray(frame)
                photo = ctk.CTkImage(light_image=image, size=(new_width, new_height))
                self.video_display.configure(image=photo)
                self.video_display.image = photo  # Keep a reference!

        self.after(10, self.update_frame)  # Schedule the next update

    def change_camera(self, choice):
        try:
            self.capture.release()
            self.capture = cv2.VideoCapture(int(choice))
        except ValueError:
            print('Invalid camera index. Please enter a number.')

    def on_click_start(self):
        self.is_start_detection = not self.is_start_detection
        if self.is_start_detection:
            self.detection_button.configure(
                text='Stop Detection', 
                fg_color="#F44336", 
                hover_color="#D32F2F"
            )
            self.test_button.configure(state='disabled')
            self.entry_ip.configure(state='disabled')
            self.entry_port.configure(state='disabled')
        else:
            self.detection_button.configure(
                text='Start Detection', 
                fg_color="#4CAF50", 
                hover_color="#388E3C"
            )
            self.test_button.configure(state='normal')
            self.entry_ip.configure(state='normal')
            self.entry_port.configure(state='normal')
            self.sent_udp(self.entry_ip.get(), self.entry_port.get(), default_pose)

    def on_click_test(self):
        reachable = self.check_ip_reachable(self.entry_ip.get(), self.entry_port.get())
        if reachable:
            self.connection_status.configure(text="Connected", text_color="#4CAF50")
            self.detection_button.configure(state='normal')
        else:
            self.connection_status.configure(text="Connection Failed", text_color="#E53935")
            self.detection_button.configure(state='disabled')

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
        except:
            try:
                sock.close()
            except:
                pass
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