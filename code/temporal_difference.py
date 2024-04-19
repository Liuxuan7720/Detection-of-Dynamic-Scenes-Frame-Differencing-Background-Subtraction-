import cv2
import numpy as np
import math
import os

data_path_dynamic = 'E:\object_detection_exp\ode\exp2\data\dynamic'
data_path_still = 'E:\object_detection_exp\ode\exp2\data\still'
image_0001 = cv2.imread('E:\object_detection_exp\ode\exp2\data\dynamic\\0001.bmp')
fps = 30
size = (352, 288)


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

def register_images(img1, img2):
    # 创建 SIFT 特征检测器
    sift = cv2.SIFT.create()

    # 检测两张图像中的特征点和描述符
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # 创建一个 BFMatcher 用于匹配特征点
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # 从匹配中提取关键点坐标
    img1_points = np.float32([kp1[m.queryIdx].pt for m in matches])
    img2_points = np.float32([kp2[m.trainIdx].pt for m in matches])

    # 估计单应性变换矩阵
    matrix, mask = cv2.findHomography(img1_points, img2_points, cv2.RANSAC, 5.0)

    # 使用单应性变换将 img1 配准到 img2
    h, w = img2.shape[:2]
    img1_registered = cv2.warpPerspective(img1, matrix, (w, h))

    return img1_registered


def temporal_difference(f_pre, f_cur, thresh, target):
    f_pre = np.float64(register_images(f_pre, target))
    f_cur = np.float64(register_images(f_cur, target))
    result = np.zeros(f_pre.shape, dtype=np.uint8)
    difference = np.abs((f_pre - f_cur))
    # index = np.where(difference > thresh)
    # result[index] = 255
    return difference


def detect_video(path, thresh, x, y):
    # select path
    data_path = path

    video = cv2.VideoWriter('E:\object_detection_exp\ode\exp2\\runs\\result_still_temporal_rect.mp4', -1, fps, size)
    for fn in os.listdir(data_path):
        filename = os.path.join(data_path, fn)
        with open(filename) as f:
            img = cv2.imread(filename)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_blurred = cv2.GaussianBlur(img, (5, 5), 4)
            x, y = 0, 0
            if img is None:
                print(filename + "is empty!")
                continue
            else:
                if fn == '0001.bmp':
                    f_pre_2 = img_blurred
                elif fn == '0002.bmp':
                    f_pre_1 = img_blurred
                elif fn == '0003.bmp':
                    f_pre_0 = img_blurred
                else:
                    f_cur = img_blurred
                    diff_1 = temporal_difference(f_pre_1, f_cur, thresh, f_cur)
                    diff_2 = temporal_difference(f_pre_0, f_pre_2, thresh, f_cur)
                    diff = diff_1 * diff_2
                    binary = np.zeros(diff.shape, dtype=np.uint8)
                    binary[np.where(diff > thresh)] = 255
                    binary_filtered = cv2.medianBlur(binary, 3)
                    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                    # result = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
                    img, x, y = search_object(img, binary_filtered, x, y, (75, 30), 4)
                    f_pre_2 = f_pre_1
                    f_pre_1 = f_pre_0
                    f_pre_0 = f_cur
                    video.write(img)
                    print("write " + fn)
    video.release()
    print("done!")
    return
