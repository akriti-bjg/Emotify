import os
import csv
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

CSV_FILE = "fer2013.csv"
OUTPUT_DIR = "dataset"
IMG_SIZE = 48 

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

emotion_map = {
    0: "angry",
    3: "happy",
    4: "sad",
    6: "neutral"
}

for split in ["train", "val", "test"]:
    for emotion in emotion_map.values():
        os.makedirs(os.path.join(OUTPUT_DIR, split, emotion), exist_ok=True)

images = []
labels = []

with open(CSV_FILE, "r") as file:
    reader = csv.DictReader(file)
    for row in reader:
        emotion = int(row["emotion"])
        if emotion not in emotion_map:
            continue
        pixels = np.array(row["pixels"].split(), dtype=np.uint8)
        image = pixels.reshape(IMG_SIZE, IMG_SIZE)
        images.append(image)
        labels.append(emotion_map[emotion])

X_train, X_temp, y_train, y_temp = train_test_split(
    images, labels, test_size=(1 - TRAIN_RATIO), random_state=42, stratify=labels
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

def save_images(images, labels, split):
    counter = {emotion: 0 for emotion in emotion_map.values()}
    for img, label in zip(images, labels):
        filename = f"{label}_{counter[label]}.jpg"
        path = os.path.join(OUTPUT_DIR, split, label, filename)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        cv2.imwrite(path, img_rgb)
        counter[label] += 1

save_images(X_train, y_train, "train")
save_images(X_val, y_val, "val")
save_images(X_test, y_test, "test")

print(f"Dataset preparation complete.")
print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
