import cv2
import numpy as np

video = cv2.VideoCapture(r"C:\Users\oluka\Desktop\Job Application 2025\Reach industries\code\AICandidateTest-FINAL.mp4")
TRACKER_TYPE = [cv2.TrackerGOTURN_create(),
                cv2.TrackerKCF_create(),
                cv2.TrackerMedianFlow_create(),
                cv2.TrackerMIL_create(),
                cv2.TrackerModel_create(),
                cv2.TrackerMOSSE_create(),
                cv2.TrackerTLD_create()
                ]

# create opencv kcf tracker
list_of_trackers = []
tracker_initialised = False

frame_count = 0
while video.isOpened():

    # iterate through the frame in the video
    ret, frame = video.read()
    frame_count += 1

    print(frame_count)

    # if unable to read frame from video break the while loop
    if not ret:
        print("Unable to read frame from video...")
        break
    
    resized_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    # initialise the tracker 
    if frame_count == 2400:
        for x in range(5):
            tracker = cv2.TrackerTLD_create()
            bbox = cv2.selectROI("Select objects to track", resized_frame, False)
            tracker.init(frame, bbox)
            list_of_trackers.append(tracker)
        tracker_initialised = True
    
    if tracker_initialised:
        for tracker in list_of_trackers:
            success, bbox = tracker.update(frame)
            if success:
                # Draw the bounding box
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(resized_frame, p1, p2, (255, 0, 0), 2, 1)

    cv2.imshow("frame", resized_frame)
    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()