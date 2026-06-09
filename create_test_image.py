"""
テスト用フローチャート画像を生成する。
フロー.png と同じような構成（青枠・日本語ラベル・大小混在）を
OpenCV/Pillow で描画する。
"""

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2

CANVAS_W, CANVAS_H = 1920, 1080
FONT_PATH = r"C:\Windows\Fonts\meiryo.ttc"

# フロー.png に合わせた文字サイズ
FONT_LARGE  = ImageFont.truetype(FONT_PATH, 28)   # スタート・終了
FONT_MEDIUM = ImageFont.truetype(FONT_PATH, 18)   # 処理実行
FONT_SMALL  = ImageFont.truetype(FONT_PATH, 13)   # 初期化・検証（問題箇所と同条件）

BORDER_COLOR = (70, 130, 200)   # 青
TEXT_COLOR   = (20, 20, 20)     # ほぼ黒
BG_COLOR     = (255, 255, 255)  # 白
ARROW_COLOR  = (80, 80, 80)     # グレー


def draw_box(draw, cx, cy, w, h, text, font, border_color=BORDER_COLOR,
             border_width=2, shape="rect"):
    """中心座標(cx, cy)にテキスト入り矩形を描画する。"""
    x0, y0 = cx - w // 2, cy - h // 2
    x1, y1 = cx + w // 2, cy + h // 2

    if shape == "diamond":
        pts = [(cx, y0), (x1, cy), (cx, y1), (x0, cy)]
        draw.polygon(pts, fill=BG_COLOR, outline=border_color)
        for i in range(border_width - 1):
            shrink = i + 1
            pts2 = [(cx, y0 + shrink), (x1 - shrink, cy),
                    (cx, y1 - shrink), (x0 + shrink, cy)]
            draw.polygon(pts2, outline=border_color)
    else:
        for i in range(border_width):
            draw.rectangle([x0 + i, y0 + i, x1 - i, y1 - i],
                           fill=BG_COLOR if i == 0 else None,
                           outline=border_color)

    # テキストを中央寄せ
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.text((cx - tw // 2, cy - th // 2), text, font=font, fill=TEXT_COLOR)

    return (x0, y0, x1, y1)


def draw_arrow(draw, x, y_top, y_bottom, color=ARROW_COLOR):
    """垂直の矢印を描画する。"""
    draw.line([(x, y_top), (x, y_bottom - 10)], fill=color, width=2)
    draw.polygon([(x, y_bottom), (x - 7, y_bottom - 14), (x + 7, y_bottom - 14)],
                 fill=color)


def main():
    img = Image.new("RGB", (CANVAS_W, CANVAS_H), BG_COLOR)
    draw = ImageDraw.Draw(img)

    cx = CANVAS_W // 2  # 中央x座標

    # ノードの定義: (y中心, 幅, 高さ, テキスト, フォント, 形状)
    nodes = [
        (120,  200, 60,  "スタート",     FONT_LARGE,  "rect"),
        (250,  160, 50,  "初期化",       FONT_SMALL,  "rect"),
        (370,  180, 50,  "データ読み込み", FONT_SMALL,  "rect"),
        (490,  160, 50,  "検証",         FONT_SMALL,  "diamond"),
        (620,  180, 55,  "処理実行",     FONT_MEDIUM, "rect"),
        (740,  160, 50,  "結果出力",     FONT_SMALL,  "rect"),
        (860,  200, 60,  "終了",         FONT_LARGE,  "rect"),
    ]

    boxes = []
    for cy, w, h, text, font, shape in nodes:
        box = draw_box(draw, cx, cy, w, h, text, font, shape=shape)
        boxes.append((cy, box))

    # 矢印を描画
    for i in range(len(boxes) - 1):
        cy_curr, box_curr = boxes[i]
        cy_next, box_next = boxes[i + 1]
        draw_arrow(draw, cx, box_curr[3], box_next[1])

    # 保存
    out_path = "samples/test_flowchart.png"
    img.save(out_path)
    print(f"Saved: {out_path}")
    print(f"Size: {CANVAS_W}x{CANVAS_H}")
    print(f"Nodes: {len(nodes)}")
    print("Small text nodes (FONT_SMALL=13px): 初期化, データ読み込み, 検証, 結果出力")
    print("Large text nodes (FONT_LARGE=28px): スタート, 終了")


if __name__ == "__main__":
    main()
