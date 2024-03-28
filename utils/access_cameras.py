import cv2

if __name__ == "__main__":
    cams_test = 500
    for i in range(cams_test):
        try:
            cap = cv2.VideoCapture(i)
            test, frame = cap.read()
            if test:
                print(f"Camera {i} is working")
        except:
            pass
