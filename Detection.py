import cv2
import math
import numpy as np
import pandas as pd
from csv import writer

df = pd.read_csv(r'Single_Test_Demo.csv')
df.truncate()
fields = ['Path', 'x', 'y', 'r', 'type']
filename = "Single_Test_Demo.csv"

# Function for opening the file
with open(filename, 'w') as writeobj:
    csvwriter = writer(writeobj)
    csvwriter.writerow(fields)

# Function for write
def write_list_as_row(filename, list):
    with open(filename, 'w', newline='') as writeobj:
        csvwriter = writer(writeobj)
        csvwriter.writerow(list)

# Function to append
def append_list_as_row(filename, list):
    with open(filename, 'a+', newline='') as writeobj:
        csvwriter = writer(writeobj)
        csvwriter.writerow(list)

# Function for finding circumcenter and circumradius for a triangle
def circum(approxim):
    ax = approxim.ravel()[0]
    ay = approxim.ravel()[1]
    bx = approxim.ravel()[2]
    by = approxim.ravel()[3]
    cx = approxim.ravel()[4]
    cy = approxim.ravel()[5]

    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    ux = int(
        ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by)) / d)
    uy = int(
        ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax)) / d)
    rad = int(math.sqrt(math.pow(ax - ux, 2) + math.pow(ay - uy, 2)))
    return ux, uy, rad

# Function for checking whether the triangle is according to the requirement
def dis(approxima):
    ax = approxima.ravel()[0]
    ay = approxima.ravel()[1]
    bx = approxima.ravel()[2]
    by = approxima.ravel()[3]
    cx = approxima.ravel()[4]
    cy = approxima.ravel()[5]
    ab = int(math.sqrt(math.pow(ax - bx, 2) + math.pow(ay - by, 2)))
    bc = int(math.sqrt(math.pow(cx - bx, 2) + math.pow(cy - by, 2)))
    ac = int(math.sqrt(math.pow(ax - cx, 2) + math.pow(ay - cy, 2)))

    if ab+bc > 1.75*ac and bc+ac > 1.75*ab and ab+ac > 1.75*bc:
        return True
    else:
        return False

# Function to find triangles in a frame
def detriangle(mask, initial):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    i = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Approximating neighbouring points to a single point
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if area > 200:
            if len(approx) == 3:
                if dis(approx):
                    x, y, radius = circum(approx)
                    crop = initial[y - radius - 30:y + radius + 30, x - radius - 30:x + radius + 30]
                    if crop.size > 0:
                        namet = "triangle thumbnail" + str(i) + ".png"
                        cv2.imwrite(namet, crop)
                        row = [namet, x, y, radius, "triangle"]
                        append_list_as_row('Single_Test_Demo.csv', row)
                    i = i + 1
                else:
                    continue

# Function for detecting circles in a frame
def decircle(mask, initial):
    res = cv2.bitwise_and(initial, initial, mask=mask)
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=130, param2=30, minRadius=0, maxRadius=0) # change this to 20 if required
    i = 0
    # ensure at least some circles were found
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # Cropping the image
            crop = initial[y - r - 30:y + r + 30, x - r - 30:x + r + 30]
            if crop.size > 0:
                namec = "circle thumbnail" + str(i) + ".png"
                cv2.imwrite(namec, crop)
                row = [namec, x, y, r, "circle"]
                append_list_as_row('Single_Test_Demo.csv', row)
            i = i + 1


def detect(mask_red, mask_blue, initial):
    write_list_as_row('Single_Test_Demo.csv', fields)
    # Sending masks for triangle and circle detection
    detriangle(mask_red, initial)
    decircle(mask_red, initial)
    detriangle(mask_blue, initial)
    decircle(mask_blue, initial)
