#!/usr/bin/env python3
"""
Compare results with and without crop_area side by side
"""

import cv2
import numpy as np

# Load images
img1 = cv2.imread('results/test_with_crop.jpg')
img2 = cv2.imread('results/test_without_crop.jpg')
template = cv2.imread('results/test_template_bboxes.jpg')

if img1 is None or img2 is None:
    print("Error: Images not found!")
    exit(1)

# Ensure same height
h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]
max_h = max(h1, h2)

# Resize if needed
if h1 < max_h:
    img1 = cv2.copyMakeBorder(img1, 0, max_h - h1, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
if h2 < max_h:
    img2 = cv2.copyMakeBorder(img2, 0, max_h - h2, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))

# Create comparison
comparison = np.hstack([img1, img2])

# Add labels
cv2.putText(comparison, "WITH crop_area", (50, 50),
           cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
cv2.putText(comparison, "WITHOUT crop_area", (w1 + 50, 50),
           cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

# Save
cv2.imwrite('results/comparison_crop_area.jpg', comparison)
print("✅ Saved: results/comparison_crop_area.jpg")

# Resize for display
scale = 1200 / comparison.shape[0]
comparison_display = cv2.resize(comparison, None, fx=scale, fy=scale)

cv2.imshow("Comparison", comparison_display)

# Also show template
if template is not None:
    scale_t = 1200 / template.shape[0]
    template_display = cv2.resize(template, None, fx=scale_t, fy=scale_t)
    cv2.imshow("Template Annotations", template_display)

cv2.waitKey(0)
cv2.destroyAllWindows()
