def segment_image(img):
    import cv2
    import numpy as np

    adaptive = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )

    kernel = np.ones((5,5), np.uint8)
    clean = cv2.morphologyEx(adaptive, cv2.MORPH_OPEN, kernel)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)

    return clean
print("Segmentation function defined successfully.")