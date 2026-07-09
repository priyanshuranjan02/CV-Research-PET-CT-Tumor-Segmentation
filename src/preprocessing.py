def preprocess_image(img):
    import cv2
    import numpy as np
    
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if img.dtype != 'uint8':
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img = img.astype('uint8')

    return img
print("Preprocessing function defined successfully.")