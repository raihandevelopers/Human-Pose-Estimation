
#import packages
import cv2
import mediapipe as mp
import numpy as np


mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose()


cap = cv2.VideoCapture("a.mp4")



while True:
    ret, img = cap.read()
    # resize image/frame so we can accommodate it on our screen
    img = cv2.resize(img, (1280, 720))

    # do Pose detection
    results = pose.process(img)
    # draw the detected pose on original video/ live stream
    mp_draw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                           mp_draw.DrawingSpec((255, 0, 0), 2, 2),
                           mp_draw.DrawingSpec((255, 0, 255), 2, 2)
                           )
    # Display pose on original video/live stream
    cv2.imshow("Pose Estimation", img)

    # Extract and draw pose on plain white image
    h, w, c = img.shape   # get shape of original frame
    opImg = np.zeros([h, w, c])  # create blank image with original frame size
    opImg.fill(255)  # set white background. put 0 if you want to make it black

    # draw extracted pose on black white image
    mp_draw.draw_landmarks(opImg, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                           mp_draw.DrawingSpec((255, 0, 0), 2, 2),
                           mp_draw.DrawingSpec((255, 0, 255), 2, 2)
                           )
  
    cv2.imshow("Extracted Pose", opImg)


    print(results.pose_landmarks)

    cv2.waitKey(1)
