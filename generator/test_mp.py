import cv2
import numpy as np
from pathlib import Path
import mediapipe as mp
import config

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "assets" / config.PORTRAIT_FILE

image = cv2.imread(str(INPUT), cv2.IMREAD_COLOR)
if image is None:
    raise RuntimeError("Image not loaded")

# Crop top
height, width = image.shape[:2]
top = max(0, int(height * 0.015))
image = image[top:height, :]

# Resize
OUTPUT_SIZE = (420, 520)
image = cv2.resize(image, OUTPUT_SIZE, interpolation=cv2.INTER_LANCZOS4)

# Mediapipe Segmentation
mp_selfie_segmentation = mp.solutions.selfie_segmentation
with mp_selfie_segmentation.SelfieSegmentation(model_selection=0) as selfie_segmentation:
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = selfie_segmentation.process(image_rgb)
    
    # Results is a mask in range [0.0, 1.0]
    mask = (results.segmentation_mask > 0.5).astype(np.uint8) * 255

    cv2.imwrite("test_mediapipe_mask.png", mask)
    print("Mediapipe mask saved.")
