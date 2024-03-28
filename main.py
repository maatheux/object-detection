from ultralytics import YOLO

from predicts_functions.predict_results import return_results
from predicts_functions.streaming_by_dipankar import running_inference_by_dipankar
from predicts_functions.streaming_predict import running_inference

yoloModel = YOLO("yolov8n.pt")

if __name__ == "__main__":
    # return_results(yoloModel)
    # running_inference(yoloModel)
    running_inference_by_dipankar(yoloModel)
