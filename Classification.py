from keras.models import load_model
import pandas as pd
import numpy as np
import cv2
from PIL import Image

model = load_model('classification.hs')
bg = [43]


def CNN():
    final_test = pd.read_csv('Single_Test_Demo.csv')
    i = len(final_test)
    imgs = final_test["Path"].values
    data = []

    for img in imgs:
        image = Image.open(img, mode='r')
        image = image.resize((32, 32))
        data.append(np.array(image))

    def preprocessing(img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.equalizeHist(img)
        img = img / 255
        return img

    X_test = np.array(data)
    X_test = np.array(list(map(preprocessing, X_test)))
    X_test = X_test.reshape(i, 32, 32, 1)

    prob = model.predict(X_test)
    j = 0
    score = []
    for k in prob:
        if max(k) > 0.65:
            # print(max(k))
            score.append(model.predict_classes(X_test)[j])
        else:
            score.append(43)
        j = j + 1

    return score
