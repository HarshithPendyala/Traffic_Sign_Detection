# import the necessary packages
import cv2
import pandas as pd

from PreProcessing import preprocessing
from Detection import detect
from Classification import CNN

classes = {
    0: "Speed Limit 20km/h", 1: "Speed Limit 30km/h", 2: "Speed Limit 50km/h", 3: "Speed Limit 60km/h",
    4: "Speed Limit 70km/h",
    5: "Speed Limit 80km/h", 6: "End of speed Limit 60km/h", 7: "Speed Limit 100km/h", 8: "Speed Limit 120km/h",
    9: "No passing",
    10: "No passing veh over 3.5 tons", 11: "Right-of-way at intersection", 12: "Priority Road", 13: "Yields",
    14: "Stop",
    15: "No Vehicles", 16: "Veh > 3.5 tons prohibited", 17: "No entry", 18: "General Caution",
    19: "Dangerous curve left",
    20: "Dangerous curve right", 21: "Double curve", 22: "Bumpy Road", 23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road Work", 26: "Traffic signals", 27: "Pedestrians", 28: "Children crossing", 29: "Bicycle crossing",
    30: "Beware of ice/snow", 31: "Wild animals crossing", 32: "End speed + passing limits", 33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead Only", 36: "Go straight or right", 37: "Go straight or left", 38: "Keep right", 39: "Keep left",
    40: "Roundabout mandatory", 41: "End of no passing", 42: "End no passing veh > 3.5 tons", 43: "background"
}

# Create a VideoCapture object and read from input file and get the fps
cap = cv2.VideoCapture('in_vid.mp4')
# cap.set(cv2.CAP_PROP_FPS, 120)
fps = int(cap.get(cv2.CAP_PROP_FPS))
count = 0

# Check if camera opened successfully
if not cap.isOpened():
    print("Error opening video  file")

# Read until video is completed
while cap.isOpened():
    # Capture frame-by-frame
    flag = 0
    ret, frame = cap.read()
    if ret:
        if count % 10 == 0:
            img = frame.copy()
            # cv2.imwrite("thumbnail " + str(count) + ".png", img)

            result_red, result_blue = preprocessing(img)
            detect(result_red, result_blue, img)

            fle = pd.read_csv('Single_Test_Demo.csv')
            if len(fle) > 0:
                predicted = CNN()
                itr = 0
                for cls in predicted:
                    if cls == 43:
                        continue
                    else:
                        if fle["type"][itr] == "circle":
                            cv2.circle(frame, (fle["x"][itr], fle["y"][itr]), fle["r"][itr], (0, 255, 0), 2)
                            cv2.rectangle(frame, (fle["x"][itr] - fle["r"][itr] - 10, fle["y"][itr] - fle["r"][itr] - 10),
                                            (fle["x"][itr] + fle["r"][itr] + 10, fle["y"][itr] + fle["r"][itr] + 10), (0, 128, 255), 2)
                            cv2.putText(frame, classes[cls], (100, 300+(itr*50)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)

                        if fle["type"][itr] == "triangle":
                            # cv2.drawContours(frame, int(fle["tri_cords"][itr]), 0, (0, 255, 0), 1)
                            cv2.rectangle(frame, (fle["x"][itr] - fle["r"][itr] - 10, fle["y"][itr] - fle["r"][itr] - 10),
                                          (fle["x"][itr] + fle["r"][itr] + 10, fle["y"][itr] + fle["r"][itr] + 10),
                                          (0, 128, 255), 0)
                            cv2.putText(frame, classes[cls], (100, 300+(itr*50)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
                    itr = itr + 1
                    flag = 1
                # cv2.imwrite("Detected " + str(count) + ".png", frame)

        # Display the resulting frame
        cv2.namedWindow('Live Feed', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Live Feed', 1440, 900)
        cv2.imshow('Live Feed', frame)
        if flag == 1:
            cv2.waitKey(1250)
        count += 1

        # Press Q on keyboard to  exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Break the loop
    else:
        break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
