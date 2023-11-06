import numpy as np
import time
import cv2
import os
import matplotlib.pyplot as plt

labelsPath = os.path.join("yolo.names")
LABELS = open(labelsPath).read().strip().split("\n")

weightsPath = os.path.join("yolov3_custom_train_900.weights")
configPath = os.path.join("yolov3_custom_train.cfg")

boxe = []
frame_label = []

net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
def predict(image):
	np.random.seed(42)
	COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
	(H, W) = image.shape[:2]
	ln = net.getLayerNames()
	ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
	blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
	net.setInput(blob)
	layerOutputs = net.forward(ln)
	boxes = []
	confidences = []
	classIDs = []
	threshold = 0.2
	for output in layerOutputs:
		for detection in output:
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]
			if confidence > threshold:
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, threshold, 0.1)
	if len(idxs) > 0:
		for i in idxs.flatten():
			boxe.append(boxes[i])
			frame_label.append("{}".format(LABELS[classIDs[i]], confidences[i]))
	else:
		boxe.append([])
		frame_label.append("NO VAL")

cap = cv2.VideoCapture('vid1.mp4')

number_frame = 30.0
video_size = (1280, 720)
fourcc = cv2.VideoWriter_fourcc(*'DIVX')

cntt = 0

while True:
	ret, frame = cap.read() 
	if ret:
		frame = cv2.resize(frame, (1280,720))
		predict(frame)
		print('Frame number: ', cntt)
		cntt += 1
		if cv2.waitKey(1) & 0xff == ord("q"):
			break
	else:
		break

fake_count = frame_label.count('fake')
real_count = frame_label.count('real')
label_this = 'real' if real_count > fake_count else 'fake'

mean_height = 0
mean_width = 0
good_frames = 0
for val in boxe:
	if len(val) == 0: continue
	mean_width += val[2]
	mean_height += val[3]
	good_frames += 1

mean_height //= good_frames
mean_width //= good_frames

for i in range(len(boxe)):
	if len(boxe[i]) == 0: continue
	boxe[i][2] = (5 * mean_width + boxe[i][2]) // 6
	boxe[i][3] = (5 * mean_height + boxe[i][2]) // 6

out = cv2.VideoWriter('deepfake_detection.mp4', fourcc, number_frame, video_size)
cap = cv2.VideoCapture('vid1.mp4')

go_cnt = 0
while True:
	ret, frame = cap.read() 
	if ret:
		frame = cv2.resize(frame, (1280,720))
		if frame_label[go_cnt] != 'NO VAL':
			(x, y) = (boxe[go_cnt][0], boxe[go_cnt][1])
			(w, h) = (boxe[go_cnt][2], boxe[go_cnt][3])
			color = (30, 30, 255)
			text = label_this
			cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
			cv2.putText(frame, text, (x + 15, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
		out.write(frame)
	else:
		break
	go_cnt += 1
	print(' ', go_cnt)

cap.release()   
out.release()
cv2.destroyAllWindows()