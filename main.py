import cv2
import numpy as np

student = cv2.imread("student.jpg")
raw_matte_img = cv2.imread("matte.png", cv2.IMREAD_GRAYSCALE)

if student is None:
    student = np.full((400, 300, 3), (180, 190, 200), dtype=np.uint8)
    cv2.rectangle(student, (90, 220), (210, 400), (40, 60, 120), -1)
    cv2.circle(student, (150, 140), 65, (140, 175, 220), -1)
    cv2.circle(student, (130, 130), 8, (30, 30, 30), -1)
    cv2.circle(student, (170, 130), 8, (30, 30, 30), -1)
    cv2.ellipse(student, (150, 160), (20, 10), 0, 0, 180, (60, 60, 150), 3)

if raw_matte_img is None:
    raw_matte_img = np.zeros((400, 300), dtype=np.uint8)
    cv2.rectangle(raw_matte_img, (90, 220), (210, 400), 255, -1)
    cv2.circle(raw_matte_img, (150, 140), 65, 255, -1)
    cv2.circle(raw_matte_img, (150, 140), 72, 128, 8)
    cv2.circle(raw_matte_img, (150, 140), 25, 110, -1)
    cv2.circle(raw_matte_img, (40, 60), 18, 255, -1)

h, w = student.shape[:2]
raw_matte_img = cv2.resize(raw_matte_img, (w, h))
raw_matte = raw_matte_img.astype(np.float32) / 255.0

sure_pixels = int(np.sum(raw_matte >= 0.9))
room_pixels = int(np.sum(raw_matte <= 0.1))
unsure_pixels = int(np.sum((raw_matte > 0.1) & (raw_matte < 0.9)))

print("Part 1 - Pixel Counts:")
print("Sure pixels (student):", sure_pixels)
print("Unsure pixels (edges/ghosts):", unsure_pixels)
print("Room pixels (background):", room_pixels)

binary_mask = (raw_matte > 0.5).astype(np.uint8)
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask)

if num_labels > 1:
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    biggest_patch = (labels == largest_label).astype(np.uint8) * 255
else:
    biggest_patch = binary_mask * 255

floodfill_mask = biggest_patch.copy()
mask_border = np.zeros((h + 2, w + 2), np.uint8)
cv2.floodFill(floodfill_mask, mask_border, (0, 0), 255)
holes = cv2.bitwise_not(floodfill_mask)
filled_patch = cv2.bitwise_or(biggest_patch, holes)

clean_matte = filled_patch.astype(np.float32) / 255.0

print("\nPart 2 - Matte Cleaned:")
print("Kept largest connected patch and filled internal holes.")

blue_card = cv2.imread("blue_card.jpg")
if blue_card is None:
    blue_card = np.full((h, w, 3), (210, 120, 30), dtype=np.uint8)
else:
    blue_card = cv2.resize(blue_card, (w, h))

student_f = student.astype(np.float32)
blue_card_f = blue_card.astype(np.float32)

clean_matte_3ch = np.stack([clean_matte] * 3, axis=-1)
clean_id_card = (clean_matte_3ch * student_f + (1.0 - clean_matte_3ch) * blue_card_f).astype(np.uint8)
cv2.imwrite("id_card_clean.png", clean_id_card)

print("\nPart 3 - Blending Complete:")
print("Saved final ID card as id_card_clean.png")

raw_matte_3ch = np.stack([raw_matte] * 3, axis=-1)
raw_id_card = (raw_matte_3ch * student_f + (1.0 - raw_matte_3ch) * blue_card_f).astype(np.uint8)

y1, y2 = int(h * 0.1), int(h * 0.65)
x1, x2 = int(w * 0.2), int(w * 0.8)

face_raw = raw_id_card[y1:y2, x1:x2]
face_clean = clean_id_card[y1:y2, x1:x2]

comparison = np.hstack([face_raw, face_clean])
cv2.imwrite("face_comparison.png", comparison)

print("\nPart 4 - Face Comparison:")
print("Saved side-by-side comparison as face_comparison.png (Left: Raw Matte with Ghosting, Right: Cleaned Matte)")
