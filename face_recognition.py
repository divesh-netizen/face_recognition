# 1. Load the training data (numpy array of all the person)
        # x-value are stored in the numpy array
        # y-value we need to assign for each person
# 2. Read a video stream using opencv
# 3. Extact face of it
# 4. use knn to find the prediction of face (int)
# 5. map the prediction id to name of the user
# 6. Display the prediction on the screen- bounding box and name

import cv2
import numpy as np
import os

################## KNN - CODE #################

def distance(v1, v2):
    return np.sqrt(((v1 - v2)**2).sum())

def knn(train, test, k=5):
    dist = []

    for i in range(train.shape[0]):
        # get the vector and label
        ix = train[i, :-1]
        iy = train[i, -1]
        # compute the distance from the test point
        d = distance(test, ix)
        dist.append([d, iy])
    # sort based on distance and get top k 
    dk = sorted(dist, key=lambda x: x[0])[:k]
    # retrieve only the labels
    labels = np.array(dk)[:, -1]

    # get frequencies of each input
    output = np.unique(labels, return_counts=True)
    # find the max frequency and corresponding label
    index = np.argmax(output[1])
    return output[0][index]

###############################################

# Init Camera
cap = cv2.VideoCapture(0)

# Face detection
face_cascade = cv2.CascadeClassifier("E:\\courses\\python\\Project_python\\face_recognition\\hr.xml")

skip = 0
dataset_path = 'data/'

face_data =  []
labels = []

class_id = 0 # Labels for the given file
names = {}   # mapping b/w id - name

# Data preparation

for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):
        # create the mappping b/w the class id and name
        names[class_id] = fx[:-4]
        print("Loaded "+fx)
        data_item = np.load(dataset_path+fx)
        face_data.append(data_item)

        # create labels for the class
        target = class_id*np.ones((data_item.shape[0], ))
        class_id += 1
        labels.append(target)

face_dataset = np.concatenate(face_data, axis=0)
face_labels = np.concatenate(labels, axis = 0).reshape((-1, 1))

print(face_dataset.shape)
print(face_labels.shape)


trainset = np.concatenate((face_dataset, face_labels), axis=1)

print(trainset.shape)


# Testing

while True:
    ret, frame = cap.read()
    if ret==False:
        continue

    faces =  face_cascade.detectMultiScale(frame, 1.3, 5)

    for face in faces:
        x, y, w, h = face

        # get the face ROI
        offset = 10
        face_section = frame[y-offset: y+h+offset, x-offset:x+w+offset]
        face_section = cv2.resize(face_section, (100, 100))

        # predict label (out)
        out = knn(trainset, face_section.flatten())

        # Display on the screen the name and rectangle around it
        pred_name=  names[int(out)]
        cv2.putText(frame, pred_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

    cv2.imshow("Faces", frame)

    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break  

cap.release()
cv2.distroyAllWindows()
