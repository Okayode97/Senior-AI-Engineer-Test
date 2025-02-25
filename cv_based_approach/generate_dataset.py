import cv_detectors
import time
import cv2
import os
from pathlib import Path
from collections import defaultdict
import json


DATA_FOLDER = r"./dataset/"

categories = [{"id": 1, "name": "petri_dish"},
              {"id": 2, "name": "gloves"}]

# box in format xyhw

def experiment_with_automated_dataset():
    video = cv_detectors.video_reader(r"AICandidateTest-FINAL.mp4")
    video.get_frame(0)

    annotation = defaultdict(list)
    detection_id = 0

    if not os.path.exists(DATA_FOLDER):
        os.mkdir(DATA_FOLDER)

    for frame in video:
        if video.frame_number % 30 == 0:
            frame = cv_detectors.image_preprocessing(frame)
        
            h, w, _ = frame.shape

            start_time = time.time()
            list_of_houghcircle_bbox = cv_detectors.houghcircle_detections(frame)
            list_of_color_based_bbox = cv_detectors.colour_based_detection(frame, 500)
            duration = time.time() - start_time

            if duration > 0:
                fps = 1/duration
                print(f"Detectors fps: {int(fps)} | frame processed: {video.frame_number}/{len(video)}",
                f"| number of detections: {len(list_of_houghcircle_bbox)+len(list_of_color_based_bbox)}")

            img_path = os.path.join(DATA_FOLDER, f"frame_{int(video.frame_number)}.png")
            annotation["images"].append({"id": int(video.frame_number), "file_name": img_path, "width": w, "height": h})
            cv2.imwrite(img_path, frame)

            houghcircle_annotations = []
            for box in list_of_houghcircle_bbox:
                detection_id += 1
                box_area = int(box.height * box.width)
                label = {"id": detection_id, "image_id": int(video.frame_number), "category_id": 1, "bbox": [int(box.y), int(box.x), int(box.height), int(box.width)], "area": box_area, "iscrowded": 0}
                houghcircle_annotations.append(label)
                frame = box.draw_bbox_onto_frame(frame, (0, 255, 0))
                frame = cv2.putText(frame, f'id: {detection_id}', (int(box.x), int(box.y+box.height/2)), cv2.FONT_HERSHEY_SIMPLEX, 
                                    1, (0, 255, 0), 2, cv2.LINE_AA)

            color_based_annotations = []
            for box in list_of_color_based_bbox:
                detection_id += 1
                box_area = int(box.height * box.width)
                label = {"id": detection_id, "image_id": int(video.frame_number), "category_id": 2, "bbox": [int(box.y), int(box.x), int(box.height), int(box.width)], "area": box_area, "iscrowded": 0}
                color_based_annotations.append(label)
                frame = box.draw_bbox_onto_frame(frame, (255, 0, 0))

                #cv putText
                frame = cv2.putText(frame, f'id: {detection_id}', (int(box.x), int(box.y+box.height/2)), cv2.FONT_HERSHEY_SIMPLEX, 
                                    1, (255, 0, 0), 2, cv2.LINE_AA)

            annotation["annotations"] += houghcircle_annotations
            annotation["annotations"] += color_based_annotations

            img_path = os.path.join(DATA_FOLDER, f"Annotated_frame_{int(video.frame_number)}.png")
            cv2.imwrite(img_path, frame)

    annotation["categories"] = categories

    with open("annotations.json", "w") as outfile:
        json.dump(annotation, outfile)

if __name__ == "__main__":
    experiment_with_automated_dataset()
