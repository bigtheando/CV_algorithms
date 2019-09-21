import cv2 as cv
import numpy as np
import pprint
import time
import sys
import os
sys.path.append("../../../image")
colors = [[0, 0, 0],
          [255, 0, 0],
          [0, 255, 0],
          [0, 0, 255],
          [128, 0, 0],
          [0, 128, 0],
          [0, 0, 128],
          [255, 255, 0],
          [255, 0, 255],
          [0, 255, 255]]


def mouse_callback(event, x, y, flags, param):
    image, label, label_col = param

    if x < 0 or y < 0 or image.shape[0] <= y or image.shape[1] <= x:
        return

    elif event == cv.EVENT_LBUTTONDOWN:
        image = cv.circle(image, (x, y), 3, colors[label_col], -1)
        label[y, x] = label_col


def labeling(image, win_name="image"):

    image_to_draw = image.copy()
    label = np.zeros(image.shape[:2])
    label_col = 1

    cv.namedWindow(win_name, cv.WINDOW_AUTOSIZE)
    
    while True:
        param = [image_to_draw, label, label_col]
        cv.setMouseCallback(win_name, mouse_callback, param)
        cv.imshow(win_name, image_to_draw)
        key = cv.waitKey(1)
        for i in range(10):
            if key == ord(str(i)):
                label_col = i

        # press Escape key
        if key == 27:
            cv.destroyAllWindows()
            break

    return label
        

def g(x):
    return 1 - (x / (255 * np.sqrt(3)))


def grow_cut(image, label, itr):

    # neighborhood
    dx = [-1, -1, -1, 0, 0, 1, 1, 1]
    dy = [-1, 0, 1, -1, 1, -1, 0, 1]

    # strength
    strength = np.where(label != 0, 1.0, 0.0)

    next_label = label.copy()
    next_strength = strength.copy()
    
    # iteration step
    for _ in range(itr):
        print("Iter : {}".format(_))
        label = next_label.copy()
        strength = next_strength.copy()
        for x in range(image.shape[0]):
            for y in range(image.shape[1]):
                for i in range(len(dx)):
                    nx, ny = x + dx[i], y + dy[i]
                    if nx < 0 or ny < 0 or \
                       image.shape[0] <= ny or image.shape[1] <= nx:
                       continue

                    norm_c = np.linalg.norm(image[y, x] - image[ny, nx])
                    if g(norm_c) * strength[ny, nx] > strength[y, x]:
                        next_label[y, x] = label[ny, nx]
                        next_strength[y, x] = g(norm_c) * strength[ny, nx]

    return next_label


def main():
    # image = cv.imread("/lena.png")
    image = cv.imread("../../../image/lena.png")
    label = labeling(image)
    start_t = time.time()
    itr = 100
    label = grow_cut(image, label, itr)
    
    result = np.zeros(image.shape)
    for y in range(label.shape[0]):
        for x in range(label.shape[1]):
            result[y, x] = colors[label[y, x]]
    elapsed_time = time.time() - start_t
    print("Elapsed time : {}".format(elapsed_time))
    cv.namedWindow("label", cv.WINDOW_AUTOSIZE)
    cv.imshow("label", result)
    cv.waitKey(0)


if __name__ == "__main__":
    main()