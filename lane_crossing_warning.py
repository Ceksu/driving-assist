import cv2
import numpy as np
from mss import mss
import time

def convert_hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)


def convert_hls(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2HLS)


def convert_gray_scale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def apply_smoothing(image, kernel_size=5):
    """
    kernel_size must be postivie and odd
    """
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def detect_edges(image, low_threshold=50, high_threshold=150):
    # type: (object, object, object) -> object
    return cv2.Canny(image, low_threshold, high_threshold)


def filter_region(image, vertices):
    """
    Create the mask using the vertices and apply it to the input image
    """
    mask = np.zeros_like(image)
    if len(mask.shape) == 2:
        cv2.fillPoly(mask, vertices, 255)
    else:
        cv2.fillPoly(mask, vertices, (255,) * mask.shape[2])  # in case, the input image has a channel dimension
    return cv2.bitwise_and(image, mask)


def select_region(image):
    """
    It keeps the region surrounded by the `vertices` (i.e. polygon).  Other area is set to 0 (black).
    """
    # first, define the polygon by vertices
    rows, cols = image.shape[:2]
    bottom_left = [cols * 0.35, rows * 0.8]
    top_left = [cols * 0.45, rows * 0.6]
    bottom_right = [cols * 0.55, rows * 0.8]
    top_right = [cols * 0.5, rows * 0.7]
    # the vertices are an array of polygons (i.e array of arrays) and the data type must be integer
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    return filter_region(image, vertices)


def process_img(image):
    calibration = 1180
    mod = select_region(image)
    mod = detect_edges(apply_smoothing(convert_gray_scale(mod)))
    c = cv2.countNonZero(mod) - calibration
    if c < 0:
        c = 0
    mod = cv2.cvtColor(mod, cv2.COLOR_GRAY2BGR)
    return image, mod, c

sct = mss()
last_time = time.time()

cap = cv2.VideoCapture('auto2-noaudio.mp4')
frames = 0

while(cap.isOpened()):
    ret, frame = cap.read()
    original_image, mod, count = process_img(frame)
    out = cv2.addWeighted(original_image, 1, mod, 0.5, 0)

    # Green rectangle
    cv2.rectangle(out, (1350, 1050), (1550, 950), (0, 255, 0), -1)
    # Warning rectangle
    cv2.rectangle(out, (1550, 1050), (1900, 950), (0, 0, 0), -1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(out, str(count), (1360, 1020), font, 2, (255, 255, 255), 4)

    if count > 200:
        cv2.putText(out, 'WARNING', (1580, 1020), font, 2, (0, 255, 255), 4)

    cv2.imshow('frame', out)
    #cv2.imwrite("frames/frame%d.jpg" % frames, out)  # save frame as JPEG file
    frames += 1

    print('Loop took {} seconds'.format(time.time() - last_time))
    last_time = time.time()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
