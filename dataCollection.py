import cv2

import numpy as np
from cvzone.HandTrackingModule import HandDetector
import math
import time
from tensorflow.keras.models import load_model
class_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
                'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

model = load_model("models/asl_model4.h5")

# accesses web cam
cap = cv2.VideoCapture(0)
#initiializes hand detector to deteft one hand
detector = HandDetector(maxHands = 1)
folder = "img/A"
counter = 0
#check if the camer
if not cap.isOpened():
    print("Cannot open camera")
    exit()

offset = 50
imgSize = 300

while True:

    success, img = cap.read()
    if not success or img is None:
        print("Failed to grab frame")
        break
    hands, img = detector.findHands(img)
    
    if not success or img is None:
        print("Failed to grab frame")
        break
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        #create white image
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255
        #crop the image
        imgHeight, imgWidth = img.shape[:2]


        x1 = max(0, x - offset)
        y1 = max(0, y - offset)
        x2 = min(imgWidth, x + w + offset)
        y2 = min(imgHeight, y + h + offset)


        imgCrop = img[y1:y2, x1:x2]
        # shape of the cropped image
        imgCropShape = imgCrop.shape

        aspectRatio = h/w
        if aspectRatio >1:
            k = imgSize / h
            wCal = min(math.ceil(k * w), imgSize)
            if imgCrop.size != 0:
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            else:
                print("Warning: imgCrop is empty, skipping this frame.")
                continue
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wGap + imgResize.shape[1]] = imgResize
        else:
            k = imgSize/h
            hCal = math.ceil(k*h)
            hCal = min(hCal, imgSize)
            if imgCrop.size != 0:
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            
            else:
                print("Warning: imgCrop is empty, skipping this frame.")
                continue
            imgResizeshape = imgResize.shape
           
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[ hGap:hCal + hGap, :] = imgResize


        
        if imgCrop.size != 0:
            cv2.imshow("ImageCrop", imgCrop)
        else:
            print("Warning: Empty crop, not displaying.")
        cv2.imshow("ImageWhite", imgWhite)
   
        imgGray = cv2.cvtColor(imgWhite, cv2.COLOR_BGR2GRAY)
        imgNormalized = imgGray / 255.0

        imgInput = imgNormalized.reshape(1, imgSize, imgSize, 1)

        # Predict class probabilities
        prediction = model.predict(imgInput)

        # Get index of highest confidence
        predicted_class_index = np.argmax(prediction)

        # Get corresponding label
        predicted_sign = class_labels[predicted_class_index]

        # Show cropped and white images
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

        # Display prediction on original image
        cv2.putText(img, f'Prediction: {predicted_sign}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                    2, (0, 255, 0), 3, cv2.LINE_AA)
     
    cv2.imshow("Image", img)
    key = cv2.waitKey(10)
    if key & 0xFF == ord('q'):
        break
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)
cap.release() 
cv2.destroyAllWindows()
