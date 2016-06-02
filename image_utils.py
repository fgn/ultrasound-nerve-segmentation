import matplotlib.pyplot as plt
import cv2

def cv_imshow(image):
    cv_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(cv_rgb)