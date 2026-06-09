"""
画像内のテキストが実際に何ピクセルの大きさかを確認するスクリプト。
黒いピクセルの塊をテキスト候補として検出し、サイズを表示する。
"""

import sys
import cv2
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "samples/フロー.png"
img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)

# 黒いテキストだけを残す（青い図形を除去）
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
blue_mask = cv2.inRange(hsv, np.array([90, 50, 50]), np.array([130, 255, 255]))
cleaned = img.copy()
cleaned[blue_mask == 255] = (255, 255, 255)

# グレースケール→二値化（黒テキストを検出）
gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

# 輪郭を検出してサイズを確認
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(f"{'高さ(px)':>8}  {'幅(px)':>8}  {'面積':>8}  位置(x,y)")
print("-" * 50)
for cnt in sorted(contours, key=lambda c: cv2.boundingRect(c)[1]):
    x, y, w, h = cv2.boundingRect(cnt)
    area = cv2.contourArea(cnt)
    if area < 1:
        continue
    print(f"{h:>8}  {w:>8}  {area:>8.0f}  ({x}, {y})")
