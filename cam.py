import cv2
from mediapipe.python.solutions import drawing_utils as mp_drawing, pose as mp_pose
import asyncio
from websockets.asyncio.client import connect
import json
from rich import print
import os
from dotenv import load_dotenv, dotenv_values
load_dotenv()

class Camera:
    def __init__(self):
        self.websocket = None
        self.cam = cv2.VideoCapture(0)
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.direction = 0
        self.host = os.getenv('HOST')
        self.port = os.getenv('PORT')

    async def async_init(self):
        while self.websocket is None:
            try:
                self.websocket = await connect(f'ws://{self.host}:{self.port}')
                await self.websocket.send("Hello world!")
                message = await self.websocket.recv()
                print(message)
            except Exception as e:
                print(f"Connection failed: {e}. Retrying in 5 seconds...")
                await asyncio.sleep(5)
    
    def __del__(self):
        self.cam.release()
        cv2.destroyAllWindows()
    
    async def get_frame(self):
        ret, img = self.cam.read()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img.flags.writeable = False

        # Body Detection
        results = self.pose.process(img)
        if results.pose_landmarks:
            landmarks = {
                'landmarks': [
                    {
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z,
                        'visibility': landmark.visibility
                    } for landmark in results.pose_landmarks.landmark
                ]
            }
            try:
                await self.websocket.send(json.dumps(landmarks))
            except Exception as e:
                print(f"Failed to send data: {e}")
                self.websocket = None  # Force reconnection

        img.flags.writeable = True
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        return img
    
    async def show_frame(self):
        cv2.imshow('raw cam', await self.get_frame())
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False
        return True
    
    async def run(self):
        await self.async_init()  # Ensure websocket connection is established
        while self.cam.isOpened():
            if self.websocket is None:
                await self.async_init()  # Reconnect if websocket is None
            if not await self.show_frame():
                break
        
        self.__del__()
    
    def get_direction(self):
        cam = Camera()
        asyncio.run(cam.async_init())
        asyncio.run(cam.run())

if __name__ == '__main__':
    cam = Camera()
    asyncio.run(cam.run())