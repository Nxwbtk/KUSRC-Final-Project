import cv2
from mediapipe.python.solutions import drawing_utils as mp_drawing, pose as mp_pose

class   Camera:
    def __init__(self):
        self.cam = cv2.VideoCapture(0)
        self.pose = mp_pose.Pose(min_detection_confidence = 0.5, min_tracking_confidence = 0.5)
    
    def __del__(self):
        self.cam.release()
        cv2.destroyAllWindows()
    
    def get_frame(self):
        ret, img = self.cam.read()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img.flags.writeable = False

        # Body Detection
        results = self.pose.process(img)
        print(results.pose_landmarks)

        img.flags.writeable = True
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        return img
    
    def show_frame(self):
        cv2.imshow('raw cam', self.get_frame())
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False
        return True
    
    def run(self):
        while self.cam.isOpened():
            if not self.show_frame():
                break
        
        self.__del__()