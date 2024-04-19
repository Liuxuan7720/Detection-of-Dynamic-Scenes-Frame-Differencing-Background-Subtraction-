import cv2
import numpy as np
import math
import os

LEARNING_RATE = 0.01
INITIAL_THRESH = 20

data_path_dynamic = 'E:\object_detection_exp\ode\exp2\data\dynamic'
data_path_still = 'E:\object_detection_exp\ode\exp2\data\still'
image_000 = cv2.imread('E:\object_detection_exp\ode\exp2\data\still\\000.bmp')
fps = 30
size = (352, 288)


# combine rect
def combine_rect(point_set, thresh=10):
    # find the smallest distance
    for i in range(len(point_set)):
        for j in range(i):
            x1, y1, w1, h1 = point_set[i]
            x2, y2, w2, h2 = point_set[j]
            # 如果重叠，则直接合成大矩形
            if h2 != -1:
                if x1 < x2 < x1 + w1 and y1 < y2 < y1 + h1:
                    x_lh, y_lh = min([x1, x2]), min([y1, y2])
                    x_rl, y_rl = max([x1 + w1, x2 + w2]), max([y1 + h1, y2 + h2])
                    point_set[i] = [x_lh, y_lh, x_rl - x_lh, y_rl - y_lh]
                    point_set[j] = [-1, -1, -1, -1]
                # 如果相距较近，直接合成大矩形
                else:
                    dis = min([abs(x1 - x2), abs(y1 - y2),
                               abs(x1 - x2 + w1), abs(x1 - x2 - w2),
                               abs(y1 - y2 + h1), abs(y1 - y2 - h2),
                               abs(x1 - x2 + w1 - w2), abs(y1 - y2 + h1 - h2)])
                    if dis < thresh:
                        x_lh, y_lh = min([x1, x2]), min([y1, y2])
                        x_rl, y_rl = max([x1 + w1, x2 + w2]), max([y1 + h1, y2 + h2])
                        point_set[i] = [x_lh, y_lh, x_rl - x_lh, y_rl - y_lh]
                        point_set[j] = [-1, -1, -1, -1]
    # 消除余下矩形
    result = []
    for i in range(len(point_set)):
        if point_set[i][0] != -1:
            result.append(point_set[i])
    return result


# 搜索矩形
def search_object(draw_img, process_img,  begin_x, begin_y, shape, widen):
    image_height, image_width = draw_img.shape
    h, w = shape
    left_end = min(0, begin_y - widen)
    high_end = min(0, begin_x - widen)
    right_end = max(image_width, begin_y + widen)
    low_end = max(image_height, begin_x + widen)
    count = np.zeros((low_end - high_end + 1, right_end - left_end + 1))
    for i in range(high_end, low_end):
        for j in range(left_end, right_end):
            patch = process_img[i:i + h, j:j + w]
            count[i, j] = np.sum(patch)
    index = np.unravel_index(count.argmax(), count.shape)
    return_x, return_y = index
    # print(return_x, return_y)
    img = cv2.rectangle(draw_img, (return_y, return_x), (return_y + w, return_x + h), (255, 0, 0), 1)
    return img, return_x, return_y


def gauss(mu, sig, x):
    a = np.exp(- ((x - mu) ** 2) / (2 * (sig ** 2)))
    b = (math.pi * 2) ** .5 * sig
    return a / b


def new_mu_sig(mu_old, sig_old, p):
    p = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY)
    # print(p.shape)
    mu_new = (1 - LEARNING_RATE) * mu_old + LEARNING_RATE * p
    sig_new = ((1 - LEARNING_RATE) * sig_old ** 2 + LEARNING_RATE * ((p - mu_new) ** 2)) ** .5
    return mu_new, sig_new


def initial_mu_sig(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_height, image_width = image.shape
    mu = np.float64(image)
    sig = np.float64(np.ones((image_height, image_width)) * INITIAL_THRESH)
    return mu, sig


def train():
    # select path
    data_path = data_path_still
    # initialize mu and sigma
    mu, sig = initial_mu_sig(image_000)
    for fn in os.listdir(data_path):
        filename = os.path.join(data_path, fn)
        with open(filename) as f:
            img = cv2.imread(filename)
            if img is None:
                print(filename + "is empty!")
                continue
            else:
                mu, sig = new_mu_sig(mu, sig, img)
                print("finished on " + fn)

    return mu, sig


def detect_single_img(img, mu, sig, lambda_thresh, x, y,  is_plt=True):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape
    binary = np.zeros((h, w), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            if abs(img[i, j] - mu[i, j]) > lambda_thresh * sig[i, j]:
                binary[i, j] = 255
    # morphology
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 定义矩形结构元素
    # result = cv2.erode(result, kernel, iterations=1)
    # result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel, iterations=1)
    binary_filtered = cv2.medianBlur(binary, 5)
    # binary_filtered = cv2.morphologyEx(binary_filtered, cv2.MORPH_DILATE, kernel, iterations=1)
    img, x, y = search_object(img, binary_filtered, x, y, (75, 30), 4)
    # cv2.imshow("result", img)
    # cv2.waitKey(0)
    if is_plt:
        cv2.imshow("1", binary_filtered)
        cv2.waitKey(0)
    return img, x, y


def detect_video(path, mu, sig, lambda_thresh):
    # select path
    data_path = path

    video = cv2.VideoWriter('E:\object_detection_exp\ode\exp2\\runs\\result.mp4', -1, fps, size)
    x, y = 0, 0
    for fn in os.listdir(data_path):
        filename = os.path.join(data_path, fn)
        with open(filename) as f:
            img = cv2.imread(filename)
            if img is None:
                print(filename + "is empty!")
                continue
            else:
                result, x, y = detect_single_img(img, mu, sig, lambda_thresh, x, y, is_plt=False)
                video.write(result)
                print("video write " + fn)
    video.release()
    print("done!")
    return
