"""
preprocess.py - 前処理モジュール

OCR用前処理と図形抽出用前処理を分離する。

- preprocess_for_ocr : 文字を読みやすくする（拡大・二値化）
- preprocess_for_shapes: 図形の線を保持する（Phase 2 で実装予定）

色除去は行わない。テキストが黒とは限らず、色で図形線とテキストを
区別することが原理的にできないため。
枠線干渉が問題になった場合の代替案はロジック解説.md「候補B」を参照。
"""

import cv2
import numpy as np


def resize_image(image: np.ndarray, scale: float) -> np.ndarray:
    """画像を scale 倍に拡大する（INTER_CUBIC補間）。"""
    h, w = image.shape[:2]
    return cv2.resize(
        image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC
    )


def binarize_otsu(gray: np.ndarray) -> np.ndarray:
    """Otsu法でしきい値を自動決定して二値化する。"""
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def binarize_adaptive(gray: np.ndarray) -> np.ndarray:
    """
    Adaptive Threshold による二値化。照明ムラ・影がある画像向け。
    Otsu の信頼度が低い場合の fallback として使う。
    """
    return cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2,
    )


def preprocess_for_ocr(
    image: np.ndarray,
    scale: float = 2.0,
    method: str = "otsu",
) -> tuple[np.ndarray, float]:
    """
    OCR用前処理を適用してグレースケール二値画像を返す。

    Args:
        image : BGR形式の入力画像
        scale : 拡大倍率（デフォルト2倍）
        method: "otsu"（デフォルト）または "adaptive"

    Returns:
        (binary_gray, scale): 二値化済みグレースケール画像と適用した倍率。
        倍率は run_ocr に渡してバウンディングボックス座標を元スケールに戻すために使う。
    """
    resized = resize_image(image, scale)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    if method == "adaptive":
        binary = binarize_adaptive(gray)
    else:
        binary = binarize_otsu(gray)

    return binary, scale
