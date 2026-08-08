import cv2
import numpy as np
from PIL import Image, ImageEnhance

# 1. Load image and mask
img = cv2.imread('assets/portrait_final.png')
# Cropping top space
height, width = img.shape[:2]
top = max(0, int(height * 0.015))
img = img[top:height, :]

mask = cv2.imread('data/portrait_mask.png', cv2.IMREAD_GRAYSCALE)

# 2. Resize
target_width, target_height = 420, 520
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
resized_img = cv2.resize(img_rgb, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
resized_mask = cv2.resize(mask, (target_width, target_height), interpolation=cv2.INTER_AREA)

# 3. Enhance with PIL
pil_img = Image.fromarray(resized_img)
enhancer = ImageEnhance.Contrast(pil_img)
pil_img = enhancer.enhance(1.2) # modest contrast
enhancer = ImageEnhance.Brightness(pil_img)
pil_img = enhancer.enhance(1.1)

# Convert to grayscale
gray_pil = pil_img.convert('L')
gray = np.array(gray_pil)

# 4. Apply mask
gray_masked = np.full_like(gray, 255)
foreground = resized_mask > 20
gray_masked[foreground] = gray[foreground]

# 5. Dither using Floyd-Steinberg on the clean masked image
# Dark pixels become black dots
img_d = gray_masked.astype(np.float32).copy()
h, w = img_d.shape
DITHER_THRESHOLD = 110 # adjust threshold to keep more facial features

for y in range(h):
    for x in range(w):
        old_pixel = img_d[y, x]
        new_pixel = 0.0 if old_pixel < DITHER_THRESHOLD else 255.0
        img_d[y, x] = new_pixel
        error = old_pixel - new_pixel
        
        if x + 1 < w: img_d[y, x + 1] += error * 7 / 16
        if y + 1 < h and x > 0: img_d[y + 1, x - 1] += error * 3 / 16
        if y + 1 < h: img_d[y + 1, x] += error * 5 / 16
        if y + 1 < h and x + 1 < w: img_d[y + 1, x + 1] += error * 1 / 16

dither_res = np.where(img_d < 128, 0, 255).astype(np.uint8)

Image.fromarray(dither_res).save('data/test_dither_fs.png')
print("Done. Check data/test_dither_fs.png")
