import cv2, numpy as np, tensorflow as tf, pika
from mediapipe.python.solutions import drawing_utils as mp_drawing, pose as mp_pose

pose_list = ['stop', 'forward', 'backward', 'left', 'right']
default_pose = pose_list[0]
credentials = pika.PlainCredentials('rabbitmq', 'password')
connection = pika.BlockingConnection(
			pika.ConnectionParameters(host="localhost", credentials=credentials),
		)
channel = connection.channel()

channel.queue_declare(queue="direction", auto_delete=True, durable=False)

_model = tf.keras.models.load_model('pose.keras')
cap = cv2.VideoCapture(0)  # this is for camera

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

with mp_pose.Pose(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as pose:
	while cap.isOpened():
		ret, img = cap.read()

		if (ret != True):
			break

		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img.flags.writeable = False

		# Body Detection
		results = pose.process(img)

		img.flags.writeable = True
		img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

		if (results.pose_landmarks):
			mp_drawing.draw_landmarks(img, results.pose_landmarks, MY_POSE_CONNECTIONS)
			np_output = _model.predict(getValueFromIMP(results.pose_landmarks.landmark))
			current_pose = pose_list[np.argmax(np_output)] if np.max(np_output) > 0.6 else default_pose
		else:
			current_pose = default_pose

		cv2.putText(img, str(f'{current_pose} {np.max(np_output):.2f}'), (50, 50), 1, 2, (0, 0, 255), 3, cv2.FONT_HERSHEY_SIMPLEX)
		cv2.imshow('raw cam', img)

		channel.basic_publish(exchange="amq.topic", routing_key="direction", body=current_pose)
		print(f" [x] Sent '{current_pose}'")

		if cv2.waitKey(1) & 0xFF == ord('e') or cv2.getWindowProperty('raw cam',cv2.WND_PROP_VISIBLE) < 1:
			break

	for i in range(5):
		channel.basic_publish(exchange="amq.topic", routing_key="direction", body=default_pose)
	channel.queue_delete(queue='direction')
	channel.close()
	connection.close()
	cap.release()
	cv2.destroyAllWindows()
