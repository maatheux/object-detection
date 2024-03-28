import cv2
from ultralytics import YOLO


def running_inference(model: YOLO ):
    cap = cv2.VideoCapture(2)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = model(frame)
        annoted_frame = results[0].plot()

        # results = model.predict(frame, stream=True, conf=0.4)
        # for result in results:
        #     annoted_frame = result.plot()
        #
        cv2.imshow("YOLOv8 Inference", annoted_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
