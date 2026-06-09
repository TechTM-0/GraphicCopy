"""
test_ocr.py - OCR処理の動作確認スクリプト

使い方:
    # 引数なし → samples/ の画像を選択 or ダミー画像
    .\.venv\Scripts\python.exe tests\test_ocr.py

    # 画像パスを指定
    .\.venv\Scripts\python.exe tests\test_ocr.py samples\myimage.jpg

出力:
    - コンソールに検出されたテキストと信頼度を表示
    - tests/debug_original.png : 元画像
    - tests/debug_mask.png     : マスク画像（テキスト領域=白）
    - tests/debug_masked.png   : マスク適用後（テキスト領域を白塗り）
"""

import sys
import cv2
import numpy as np

sys.path.insert(0, "src")
from graphiccopy.ocr import run_ocr_with_preprocess


def make_dummy_image() -> np.ndarray:
    """引数なし時に使うダミー画像（テキスト2つ＋横線）"""
    img = np.ones((200, 400, 3), dtype=np.uint8) * 255
    cv2.putText(img, "Hello", (50, 80),   cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    cv2.putText(img, "World", (200, 160), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    cv2.line(img, (0, 100), (400, 100), (0, 0, 0), 2)
    return img


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # 引数でパスを明示指定した場合
        path = sys.argv[1]
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            print(f"ERROR: 画像を読み込めませんでした: {path}")
            sys.exit(1)
        print(f"入力: {path}  ({img.shape[1]}x{img.shape[0]}px)")
    else:
        # 引数なし → samples/ フォルダの画像を一覧表示して選択
        import glob
        candidates = glob.glob("samples/*.*")
        image_files = [p for p in candidates if p.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp"))]

        if image_files:
            print("=== samples/ の画像 ===")
            for i, p in enumerate(image_files):
                print(f"  {i}: {p}")
            print("  d: ダミー画像を使う")

            choice = input("\n番号を入力: ").strip()
            if choice == "d":
                img = make_dummy_image()
                print("入力: ダミー画像")
            elif choice.isdigit() and int(choice) < len(image_files):
                path = image_files[int(choice)]
                # cv2.imread は日本語パスを読めないため np.fromfile + imdecode を使う
                img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    print(f"ERROR: 画像を読み込めませんでした: {path}")
                    sys.exit(1)
                print(f"入力: {path}  ({img.shape[1]}x{img.shape[0]}px)")
            else:
                print("ERROR: 無効な入力です")
                sys.exit(1)
        else:
            img = make_dummy_image()
            print("入力: ダミー画像（samples/ に画像が見つからなかったため）")

    result = run_ocr_with_preprocess(img, verbose=True)

    print("\n=== text_blocks ===")
    if result["text_blocks"]:
        for block in result["text_blocks"]:
            print(f"  text={block['text']!r}  bbox={block['bbox']}  conf={block['confidence']:.1f}")
    else:
        print("  （テキスト検出なし）")

    mask = result["mask"]
    white_pixels = int(np.sum(mask == 255))
    total_pixels = mask.size
    print(f"\nmask: shape={mask.shape}  "
          f"white_pixels={white_pixels} ({white_pixels / total_pixels * 100:.1f}%)")

    masked = img.copy()
    masked[mask == 255] = [255, 255, 255]

    bbox_img = img.copy()
    for block in result["text_blocks"]:
        x, y, w, h = block["bbox"]
        cv2.rectangle(bbox_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        label = f"{block['text']} {block['confidence']:.0f}%"
        cv2.putText(bbox_img, label, (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    cv2.imencode(".png", img)[1].tofile("tests/debug_original.png")
    cv2.imencode(".png", mask)[1].tofile("tests/debug_mask.png")
    cv2.imencode(".png", masked)[1].tofile("tests/debug_masked.png")
    cv2.imencode(".png", bbox_img)[1].tofile("tests/debug_bbox.png")
    print("\n保存完了:")
    print("  tests/debug_original.png : 元画像")
    print("  tests/debug_mask.png     : マスク（白=テキスト領域）")
    print("  tests/debug_masked.png   : マスク適用後")
    print("  tests/debug_bbox.png     : bbox描画")
