import cv_detectors
import time
import cv2


video = cv_detectors.video_reader(r"AICandidateTest-FINAL.mp4")
video.get_frame()

for frame in video:
    frame = cv_detectors.image_preprocessing(frame)
 
    start_time = time.time()
    list_of_houghcircle_bbox = cv_detectors.houghcircle_detections(frame)
    list_of_color_based_bbox = cv_detectors.colour_based_detection(frame, 500)
    duration = time.time() - start_time

    if duration > 0:
        fps = 1/duration
        print(f"Detectors fps: {int(fps)} | frame processed: {video.frame_number}/{len(video)}",
        f"| number of detections: {len(list_of_houghcircle_bbox)+len(list_of_color_based_bbox)}")

    frame = cv_detectors.draw_bounding_box(frame, list_of_color_based_bbox, (255, 0, 0))
    frame = cv_detectors.draw_bounding_box(frame, list_of_houghcircle_bbox, (0, 255, 0))

    cv_detectors.find_chemical_in_houghcircle(frame, list_of_houghcircle_bbox)

    # Display result
    cv2.imshow("Lab technician object detection and tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
