import torch
from ultralytics import YOLO


def return_results(model: YOLO):
    # run inference on images bus.jpg and cr7.png and return results
    # results = model(['bus.jpg', 'cr7.png'], stream=True)  # return a list of Results objects

    tensor = torch.zeros(2, 3, 320, 640)

    tensor[:, 0, :, :] = 0 # B
    tensor[:, 1, :, :] = 255 # G
    tensor[:, 2, :, :] = 0 # R

    results = model(tensor, stream=True)  # return a list of Results objects

    # Process results list
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        result.show()  # display to screen
        result.save(filename='result.jpg')  # save to disk
