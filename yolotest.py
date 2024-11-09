
from ultralytics import YOLO
import cv2
import supervision as sv
import torch    ## pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 ## gpu hızlandırması için
# print(torch.__version__)
# print(torch.cuda.is_available())


def main():
    torch.cuda.set_device(0)
    # device = torch.device('cpu')
    device = torch.device('cuda')
    # print(f'Using device: {device}')
    # cap = cv2.VideoCapture(r".\videos\testvideo.mp4")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  
    model = YOLO(r".\models\yolov8x.pt" ).to(device)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    count = 0
    while True:

        frame = cap.read()[1]       ## returns an array [success, frame]
        frame = cv2.flip(frame, 1)
        result = model(frame)[0]
        # result = model(frame, classes=[2,3,5,7])[0]
        # result = model.track(r".\videos\testvideo.mp4",classes=[2,3,5,7], show=True, tracker="bytetrack.yaml",persist=False, stream=False)

        annotated = sv.Detections.from_ultralytics(result)
        count = count + 1
        print(f"Frame Number {count}")
        for bbox, _, confidence, class_id ,_,_  in annotated:
            x1, y1, x2, y2 = bbox 
            print(f"Class: {model.model.names[class_id]}, Confidence: {confidence:.2f}, Coordinates: ({((x1 + x2)/2):0.2f}, ({((y1 + y2)/2):0.2f})")
        labels = [
            f"Class: {model.model.names[class_id]}, Confidence: {confidence:.2f}"
            for bbox, _, confidence, class_id , tracker_id,_
            in annotated
        ]
        frame = sv.BoxAnnotator(
            thickness=2,
            ).annotate(
            scene=frame,
            detections=annotated,
            )
        frame = sv.LabelAnnotator(
            text_thickness=1,
            text_scale=0.5
            ).annotate(
            scene=frame,
            detections=annotated,
            labels=labels,
        )

        cv2.imshow("yolov8",frame)

        if cv2.waitKey(1) == 27:
            break


if __name__ == "__main__":
    main()