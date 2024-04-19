import os
import cv2
import numpy as np
import gauss_background_subtraction as bs
import temporal_difference as td

data_path_dynamic = 'E:\object_detection_exp\ode\exp2\data\dynamic'
data_path_still = 'E:\object_detection_exp\ode\exp2\data\still'
fps = 30
size = (352, 288)
video_len = 1108

test_img = cv2.imread('E:\object_detection_exp\ode\exp2\data\dynamic\\0002.bmp')
test_img_2 = cv2.imread('E:\object_detection_exp\ode\exp2\data\dynamic\\0004.bmp')
# cv2.imshow("1", test_img)
# cv2.waitKey(0)

mu, sig = bs.train()
# cv2.imshow("background", np.uint8(mu))
# cv2.waitKey(0)
# cv2.imwrite("E:\object_detection_exp\ode\exp2\\runs\\background.bmp", np.uint8(mu))
# 高斯差分检测单张图片
bs.detect_single_img(test_img, mu, sig, 1.7)
# 高斯差分检测视频
bs.detect_video(data_path_still, mu, sig, 1.85)
# 帧间差分
td.detect_video(data_path_dynamic, 110, 0, 0)
# 配准测试
# img = td.register_images(test_img, test_img_2)
# cv2.imshow('match',test_img)
# cv2.waitKey(0)

# video = cv2.VideoWriter('E:\object_detection_exp\ode\exp2\\runs\out_still.mp4', -1, fps, size)
#
# path = data_path_still
#
# for fn in os.listdir(path):
#     filename = os.path.join(path, fn)
#     with open(filename) as f:
#         img = cv2.imread(filename)
#         if img is None:
#             print(filename + "is empty!")
#             continue
#         video.write(img)
# video.release()
# print("video write end!")




