# YOLO barcode detection
import cv2 as cv
import numpy as np
import time
import imutils
from yolo_barcode_functions import ScanImg, RotateImg

img = cv.imread("images/3.jpg")
# Load names of classes and get random colors
classes = open("yolov3-barcode/obj.names").read().strip().split('\n')
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')

# Give the configuration and weight files for the model and load the network.
net = cv.dnn.readNetFromDarknet("yolov3-barcode/yolov3-barcode.cfg", "yolov3-barcode/yolov3-barcode_last.weights")
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
# net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

# determine the output layer
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# construct a blob from the image
blob = cv.dnn.blobFromImage(img, 1 / 255.0, (608, 608), swapRB=True, crop=False)
r = blob[0, 0, :, :]

cv.imshow('blob', r)
text = f'Blob shape={blob.shape}'
# cv.displayOverlay('blob', text)
print(text)
cv.waitKey(1)

net.setInput(blob)
t0 = time.time()
outputs = net.forward(ln)
t = time.time()
print('time=', t - t0)

print(len(outputs))
for out in outputs:
    print(out.shape)


def trackbar2(x):
    confidence = x / 100
    r = r0.copy()
    for output in np.vstack(outputs):
        if output[4] > confidence:
            x, y, w, h = output[:4]
            p0 = int((x - w / 2) * 608), int((y - h / 2) * 608)
            p1 = int((x + w / 2) * 608), int((y + h / 2) * 608)
            cv.rectangle(r, p0, p1, 1, 1)
    cv.imshow('blob', r)
    text = f'Bbox confidence={confidence}'
    # cv.displayOverlay('blob', text)
    print(text)


r0 = blob[0, 0, :, :]
r = r0.copy()
cv.imshow('blob', r)
cv.createTrackbar('confidence', 'blob', 50, 101, trackbar2)
trackbar2(50)

boxes = []
confidences = []
classIDs = []
h, w = img.shape[:2]

for output in outputs:
    for detection in output:
        scores = detection[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]
        if confidence > 0.3:
            box = detection[:4] * np.array([w, h, w, h])
            (centerX, centerY, width, height) = box.astype("int")
            x = int(centerX - (width / 2))
            y = int(centerY - (height / 2))
            box = [x, y, int(width), int(height)]
            boxes.append(box)
            confidences.append(float(confidence))
            classIDs.append(classID)

indices = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

imgCopy = img.copy()
if len(indices) > 0:
    for i in indices.flatten():
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])

        if classes[classIDs[i]] == "Barcode":
            y1 = max(0, y - int(h / 8))
            y2 = min(img.shape[0] - 1, y + h + int(h / 8))
            x1 = max(0, x - int(w / 5))
            x2 = min(img.shape[1] - 1, x + w + int(w / 5))
        else:
            y1 = max(0, y - int(h / 8))
            y2 = min(img.shape[0] - 1, y + h + int(h / 8))
            x1 = max(0, x - int(w / 8))
            x2 = min(img.shape[1] - 1, x + w + int(w / 8))
        # Crop each bounding box create by yolo
        croppedImg = img[y1:y2, x1:x2]

        # Scan cropped image with ZBar library
        if ScanImg(croppedImg) == -1 and classes[classIDs[i]] == "Barcode":
            # Rotate image if ZBar cannot decode
            rotatedImg = RotateImg(croppedImg)
            if ScanImg(rotatedImg) == -1:
                # Resize image if ZBar still cannot decode
                RSImg = cv.resize(rotatedImg, (int(300 * w / h), 300))
                ScanImg(RSImg)

        # Draw Box detect by YOLO
        color = [int(c) for c in colors[classIDs[i]]]
        cv.rectangle(imgCopy, (x, y), (x + w, y + h), color, 2)
        text = "{}: {:.4f}".format(classes[classIDs[i]], confidences[i])
        cv.putText(imgCopy, text, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 1, color, 2)

# Show image with YOLO boxes
cv.imshow("Window", imgCopy)
cv.waitKey(0)
cv.destroyAllWindows()
