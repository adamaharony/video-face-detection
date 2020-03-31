'''
        This is a BETA build of a face recognition algorithm.
'''

import cv2

face_classifier = cv2.CascadeClassifier("face-haar.xml")   # This is the face cascade for the model detection
eye_classifier = cv2.CascadeClassifier("eye-haar.xml")   # This is the eye cascade for the model detection
vid = cv2.VideoCapture(0)   # Getting a video feed
  

# STARTING WITH THE MAIN CLASSIFYING FUNCTION:
while True:
    r, frame = vid.read()   # Getting each frame from the video feed

    grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    # Converting each BGR frame to greyscale

    # Recognised faces:
    recognised_faces = face_classifier.detectMultiScale(
        grey_frame,     # Forwarding each greyscale frame to the classifier
        scaleFactor=1.2,
        minNeighbors=10, 
        minSize=(50, 50)    # Minimum size of a face (in pixels)
    )

    # Recognised eyes:
    recognised_eyes = eye_classifier.detectMultiScale(
        grey_frame,     # Forwarding each greyscale frame to the classifier
        scaleFactor=1.2,
        minNeighbors=15,
        minSize=(15, 15)    # Minimum size of an eye (in pixels)
    )


    # Drawing a recrangle around every face recognised:
    for (x, y, width, height) in recognised_faces:
        cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0), 1) # BGR format
        
    # Drawing a recrangle around every eye recognised: 
    for (x, y, width, height) in recognised_eyes:
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 1) # BGR format

    # Display everything recongised in the classifier:
    cv2.imshow("Live Video Feed - Face Detection", frame)


    # Waiting for the 'Q' key, then exiting the program:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break   # Leaving the main while loop


  
# After finishing everything, close everything:  
vid.release()
cv2.destroyAllWindows()
    