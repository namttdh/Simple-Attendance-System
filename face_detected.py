import face_recognition
from sklearn import svm
import os
import cv2
import numpy as np
import time
from datetime import datetime


known_face_encodings = []
face_attendanced = set()
names = []
process_this_frame = True
face_locations = []
face_encodings = []

# Training directory
train_dir = os.listdir('./train')
for person in train_dir:
    pix = os.listdir("./train/" + person)

    for person_img in pix:
        face = face_recognition.load_image_file("./train/" + person + "/" + person_img)
        face_bounding_boxes = face_recognition.face_locations(face)

        #If training image contains exactly one face
        if len(face_bounding_boxes) == 1:
            face_enc = face_recognition.face_encodings(face)[0]
            known_face_encodings.append(face_enc)
            names.append(person)
        else:
            print(person + "/" + person_img + " Khong dung duoc")

video_capture = cv2.VideoCapture(0)

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    rgb_small_frame = small_frame[:, :, ::-1]

    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index] and names[best_match_index] not in face_attendanced:
                name = names[best_match_index]
                face_attendanced.add(name)
                now = datetime.now()
                current_time = now.strftime("%d/%m/%Y, %H:%M:%S")
                f = open("./attendance/diemdanh"+now.strftime("%d%m%Y")+".txt", "a")
                f.write(name+" " +current_time +"\n")
                f.close()
    process_this_frame = not process_this_frame


    # Display the results
    # for (top, right, bottom, left), name in zip(face_locations, face_names):
    #     # Scale back up face locations since the frame we detected in was scaled to 1/4 size
    #     top *= 4
    #     right *= 4
    #     bottom *= 4
    #     left *= 4

    #     # Draw a box around the face
    #     cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

    #     # Draw a label with a name below the face
    #     cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
    #     font = cv2.FONT_HERSHEY_DUPLEX
    #     cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
