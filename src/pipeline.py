import os
import cv2
import numpy as np
import pandas as pd

# -------------------------------
# PATH SETUP
# -------------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))

input_folder = os.path.join(base_dir, "..", "data", "2D Images")
output_folder = os.path.join(base_dir, "..", "results")

os.makedirs(output_folder, exist_ok=True)

print("Reading from:", input_folder)

# -------------------------------
# RESULTS STORAGE
# -------------------------------
results = []

# -------------------------------
# PROCESS ALL IMAGES
# -------------------------------
for filename in os.listdir(input_folder):
    if filename.endswith(".tif"):

        path = os.path.join(input_folder, filename)
        img = cv2.imread(path, -1)

        if img is None:
            print("Skipping:", filename)
            continue

        # -------------------------------
        # PREPROCESSING
        # -------------------------------
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if img.dtype != 'uint8':
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
            img = img.astype('uint8')

        # -------------------------------
        # SEGMENTATION (Adaptive Threshold)
        # -------------------------------
        adaptive_mask = cv2.adaptiveThreshold(
            img,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            15,   # tuned
            3     # tuned
        )

        # -------------------------------
        # MORPHOLOGY (BALANCED)
        # -------------------------------
        kernel = np.ones((5, 5), np.uint8)

        # Only closing (preserve region)
        clean_mask = cv2.morphologyEx(adaptive_mask, cv2.MORPH_CLOSE, kernel)

        # Controlled dilation (avoid overgrowth)
        clean_mask = cv2.dilate(clean_mask, kernel, iterations=1)

        # -------------------------------
        # REMOVE BORDERS
        # -------------------------------
        h, w = clean_mask.shape
        margin = 5

        clean_mask[:margin, :] = 0
        clean_mask[-margin:, :] = 0
        clean_mask[:, :margin] = 0
        clean_mask[:, -margin:] = 0

        # -------------------------------
        # CONTOUR DETECTION
        # -------------------------------
        contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        tumor_contour = None
        valid_contours = []

        for cnt in contours:
            # Validate contour before processing
            if cnt is None or len(cnt) < 3:
                continue
            
            try:
                area = cv2.contourArea(cnt)
            except Exception as e:
                print(f"⚠️ Error calculating area for contour in {filename}: {e}")
                continue

            # Balanced filtering
            if 1000 < area < 50000:
                valid_contours.append(cnt)

        # Selection logic
        if len(valid_contours) > 0:
            tumor_contour = max(valid_contours, key=cv2.contourArea)
        elif len(contours) > 0:
            tumor_contour = max(contours, key=cv2.contourArea)
        else:
            tumor_contour = None

        # -------------------------------
        # FEATURE EXTRACTION
        # -------------------------------
        area = 0
        perimeter = 0
        
        if tumor_contour is not None:
            try:
                # Validate contour before calculating features
                if len(tumor_contour) >= 3:
                    area = cv2.contourArea(tumor_contour)
                    perimeter = cv2.arcLength(tumor_contour, True)
                else:
                    print(f"⚠️ Invalid contour (insufficient points) in {filename}")
            except Exception as e:
                print(f"⚠️ Error extracting features for {filename}: {e}")
                area = 0
                perimeter = 0

        results.append({
            "image": filename,
            "area": area,
            "perimeter": perimeter
        })

        # -------------------------------
        # DRAW OUTPUT
        # -------------------------------
        output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        if tumor_contour is not None:
            cv2.drawContours(output, [tumor_contour], -1, (0, 255, 0), 2)

        save_path = os.path.join(output_folder, filename)
        cv2.imwrite(save_path, output)

# -------------------------------
# SAVE RESULTS (SAFE WRITE)
# -------------------------------
df = pd.DataFrame(results)

csv_path = os.path.join(output_folder, "results.csv")

if os.path.exists(csv_path):
    try:
        os.remove(csv_path)
    except PermissionError:
        print("⚠️ Close results.csv file and run again")
        exit()

df.to_csv(csv_path, index=False)

print("\nProcessing Done ✅")
print("Total images processed:", len(results))
print("Results saved at:", csv_path)