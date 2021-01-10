import os
import sys
import questionary
import tensorflow as tf
import cv2
import numpy as np


LABELS = {
    0: "Speed limit (20km/h)",
    1: "Speed limit (30km/h)",
    2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)",
    4: "Speed limit (70km/h)",
    5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)",
    7: "Speed limit (100km/h)",
    8: "Speed limit (120km/h)",
    9: "No passing",
    10: "No passing for vehicles over 3.5 metric tons",
    11: "Right-of-way at the next intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles over 3.5 metric tons prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve to the left",
    20: "Dangerous curve to the right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End of all speed and passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End of no passing by vehicles over 3.5 metric tons",
}


def main():
    if len(sys.argv) not in [2]:
        sys.exit("Usage: python recognize.py data_directory")

    # Get image arrays and labels for all image files
    images = load_data(sys.argv[1])

    models_paths = os.listdir("models")
    models_paths.insert(-1, "Exit")
    if len(models_paths) != 0:
        answer = questionary.select(
            "Which old model you want to load?",
            choices=models_paths
                           ).ask()
        model = tf.keras.models.load_model(os.path.join("models", answer))

        if answer == "Exit":
            exit(1)

    y_pred = model.predict(images)

    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (25, 25)
    font_scale = 0.5
    color = (0, 255, 0)
    thickness = 2

    for i in range(10 if len(y_pred) > 10 else len(y_pred)/2):
        img = cv2.resize(images[i], (250, 250))
        img = cv2.putText(img, LABELS.get(np.argmax(y_pred[i])), org, font,
                          font_scale, color, thickness, cv2.LINE_AA)
        cv2.imshow("Prediction " + str(i), img)
        cv2.waitKey()


def load_data(data_dir):
    images = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if not file.split('.')[-1].lower() in ["ppm", "jpg", "jpeg", "png"]:
                continue
            img = cv2.imread(os.path.join(root, file), 1)
            img = cv2.resize(img, dsize=(30, 30))
            images.append(img / 255.)

    return np.array(images)


if __name__ == "__main__":
    main()
