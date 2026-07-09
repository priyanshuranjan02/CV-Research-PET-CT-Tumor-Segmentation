def extract_features(contour):
    import cv2
    
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    return area, perimeter
print("Feature extraction function defined successfully.")