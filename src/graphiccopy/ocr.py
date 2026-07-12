"""
ocr.py - OCR処理とテキストマスク生成

処理パイプライン上の役割:
    INPUT IMAGE → [このモジュール] → text_blocks + mask → 図形抽出

このモジュールがやること:
    1. 画像内のテキスト領域を検出してテキストを抽出する（OCR）
    2. テキストが存在する領域を白く塗ったマスク画像を生成する

マスクを生成する理由:
    次の図形抽出ステップで「ここはテキストなので図形として扱わない」と
    判断するために使う。マスクがないと文字の輪郭を図形として誤検出する。
"""

import cv2
import numpy as np
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

CONFIDENCE_THRESHOLD = 0

# MSERによる候補検出時の拡大倍率。
# 元画像で10〜14pxの文字が4倍で40〜56pxになりMSERが検出しやすくなる。
_DETECT_SCALE = 4.0


def load_image(path: str) -> np.ndarray:
    """
    日本語・全角文字を含むパスでも画像を読み込む。
    cv2.imread は Windows で非ASCII パスを読めないため imdecode を使う。
    """
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def run_ocr(image: np.ndarray, scale: float = 1.0, config: str = "--psm 11",
            debug: dict | None = None) -> dict:
    """
    画像に対してOCRを実行し、テキスト情報とマスク画像を返す。

    Args:
        image : BGR または グレースケール画像（np.ndarray）
        scale : preprocess_for_ocr で適用した拡大倍率。
                バウンディングボックスとマスクを元画像の座標系に戻すために使う。
        config: Tesseract設定文字列。
                全体画像は "--psm 11"（疎なテキスト）、
                切り抜きは "--psm 7"（1行テキスト）。
        debug : dict を渡すと Tesseract の生出力（空文字・低信頼度を含む）を
                debug["raw"] に格納する。通常の処理では捨てられる情報のため、
                「切り抜きは作られたがOCRが空を返した」ケースの追跡に使う。

    Returns:
        {
            "text_blocks": [{"text": str, "bbox": [x, y, w, h], "confidence": float}],
            "mask": np.ndarray  # 元画像サイズ・テキスト領域が白(255)
        }
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    data = pytesseract.image_to_data(
        gray,
        lang="jpn",
        output_type=pytesseract.Output.DICT,
        config=config,
    )

    if debug is not None:
        debug["raw"] = [
            {"text": data["text"][i], "conf": float(data["conf"][i])}
            for i in range(len(data["text"]))
            if float(data["conf"][i]) >= 0  # 単語レベルの行のみ（-1はページ/段落等の階層行）
        ]

    h, w = gray.shape[:2]
    orig_h = int(round(h / scale))
    orig_w = int(round(w / scale))
    mask = np.zeros((orig_h, orig_w), dtype=np.uint8)

    text_blocks = []

    for i, text in enumerate(data["text"]):
        text = text.strip()
        if not text:
            continue

        conf = float(data["conf"][i])
        if conf < CONFIDENCE_THRESHOLD:
            continue

        x  = int(round(data["left"][i]  / scale))
        y  = int(round(data["top"][i]   / scale))
        bw = int(round(data["width"][i] / scale))
        bh = int(round(data["height"][i]/ scale))

        text_blocks.append({"text": text, "bbox": [x, y, bw, bh], "confidence": conf})
        cv2.rectangle(mask, (x, y), (x + bw, y + bh), 255, -1)

    return {"text_blocks": text_blocks, "mask": mask}


def _iou(bbox1: list, bbox2: list) -> float:
    """2つの bbox [x, y, w, h] の IoU（重複面積の割合）を返す。"""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    ix = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    iy = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    intersection = ix * iy
    union = w1 * h1 + w2 * h2 - intersection
    return intersection / union if union > 0 else 0.0


def _deduplicate(blocks: list, iou_threshold: float = 0.3) -> list:
    """
    同一テキストが重複検出された場合に除去する。
    信頼度の高い順にソートし、IoU が閾値を超える後発のブロックを捨てる。
    """
    sorted_blocks = sorted(blocks, key=lambda b: b["confidence"], reverse=True)
    kept = []
    for block in sorted_blocks:
        if not any(_iou(block["bbox"], k["bbox"]) > iou_threshold for k in kept):
            kept.append(block)
    return kept


def _mser_detect_regions(gray: np.ndarray) -> list:
    """
    MSERで文字候補のbbox [x, y, w, h] を検出する。
    _DETECT_SCALE 倍に拡大したグレースケール画像を受け取ることを前提とする。
    """
    mser = cv2.MSER_create(5, 150, 14400)
    regions, _ = mser.detectRegions(gray)
    return [list(cv2.boundingRect(pts)) for pts in regions]


def _filter_by_size(bboxes: list,
                    min_h: int, max_h: int,
                    min_w: int, max_w: int) -> list:
    """サイズが文字らしくない領域を除外する。"""
    return [
        [x, y, w, h] for x, y, w, h in bboxes
        if min_h <= h <= max_h and min_w <= w <= max_w
    ]


def _nms_regions(bboxes: list, iou_threshold: float = 0.3) -> list:
    """
    MSERが同一文字に対して複数の重複領域を生成するため、
    IoUが閾値を超える重複を除去する。面積の大きい領域を優先して残す。
    """
    if not bboxes:
        return []
    sorted_boxes = sorted(bboxes, key=lambda b: b[2] * b[3], reverse=True)
    kept = []
    for box in sorted_boxes:
        x, y, w, h = box
        duplicate = False
        for kx, ky, kw, kh in kept:
            ix = max(0, min(x + w, kx + kw) - max(x, kx))
            iy = max(0, min(y + h, ky + kh) - max(y, ky))
            inter = ix * iy
            union = w * h + kw * kh - inter
            iou = inter / union if union > 0 else 0.0
            if iou > iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(box)
    return kept


def _merge_horizontal(bboxes: list,
                       height_ratio: float = 0.8,
                       y_overlap_ratio: float = 0.7,
                       verbose: bool = False) -> list:
    """
    以下の3条件を全て満たす場合のみ横方向にマージする。

      1. 横ギャップ <= max(h1, h2)  ← 文字高さ基準の相対値
      2. 高さ比 >= height_ratio     ← min(h1,h2)/max(h1,h2)
      3. 縦重なり >= y_overlap_ratio * min(h1, h2)

    条件2・3により枠線（文字と縦重なりしない / 高さが大きく異なる）は結合しない。
    推移的マージをサポートするため変化がなくなるまで繰り返す。

    verbose=True でマージ判定の詳細ログを出力する。
    """
    if not bboxes:
        return []
    boxes = [[x, y, x + w, y + h] for x, y, w, h in bboxes]

    changed = True
    while changed:
        changed = False
        merged = [False] * len(boxes)

        for i in range(len(boxes)):
            if merged[i]:
                continue
            ax1, ay1, ax2, ay2 = boxes[i]
            ah = ay2 - ay1

            for j in range(i + 1, len(boxes)):
                if merged[j]:
                    continue
                bx1, by1, bx2, by2 = boxes[j]
                bh = by2 - by1

                # 各条件の計算
                h_ratio  = min(ah, bh) / max(ah, bh)
                y_overlap = min(ay2, by2) - max(ay1, by1)
                y_ol_ratio = y_overlap / min(ah, bh) if min(ah, bh) > 0 else 0
                h_gap    = max(bx1 - ax2, ax1 - bx2, 0)
                gap_limit = max(ah, bh)

                ok_ratio   = h_ratio   >= height_ratio
                ok_overlap = y_ol_ratio >= y_overlap_ratio
                ok_gap     = h_gap     <= gap_limit
                do_merge   = ok_ratio and ok_overlap and ok_gap

                if verbose:
                    reason = "OK" if do_merge else (
                        "height_ratio" if not ok_ratio else
                        "y_overlap"    if not ok_overlap else
                        "gap"
                    )
                    print(
                        f"merge? h_ratio={h_ratio:.2f} overlap={y_ol_ratio:.2f} "
                        f"gap={h_gap}(lim={gap_limit}) → {'True' if do_merge else f'False ({reason})'}"
                    )

                if not do_merge:
                    continue

                # マージ: j を i に吸収
                boxes[i] = [min(ax1, bx1), min(ay1, by1),
                             max(ax2, bx2), max(ay2, by2)]
                ax1, ay1, ax2, ay2 = boxes[i]
                ah = ay2 - ay1
                merged[j] = True
                changed = True

        boxes = [b for i, b in enumerate(boxes) if not merged[i]]

    return [[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2 in boxes]


def _filter_by_aspect(bboxes: list,
                       min_h: int = 15,
                       max_wh_ratio: float = 10.0) -> list:
    """
    「高さが極端に小さく かつ 幅/高さ比が大きすぎる」領域を枠線とみなして除外する。
    幅/高さ比の単独判定にしない理由: 長いラベル（「処理開始前チェック」等）も
    幅が長くなるが高さはそれなりにあるため、単独判定だと誤除外されるリスクがある。
    """
    result = []
    for x, y, w, h in bboxes:
        if h == 0:
            continue
        is_thin = h < min_h
        is_wide = (w / h) > max_wh_ratio
        if is_thin and is_wide:
            continue  # 細くて横長 → 枠線とみなす
        result.append([x, y, w, h])
    return result


def run_ocr_with_preprocess(image: np.ndarray, verbose: bool = False,
                            debug: dict | None = None) -> dict:
    """
    MSERで文字候補領域を検出し、各領域を切り抜いてOCRする。
    全体画像では見つけられない小さいテキスト（10〜14px）に対応する。

    処理フロー:
        1. _DETECT_SCALE 倍に拡大してMSER検出
        2. サイズフィルタ
        3. NMS（MSERの重複除去・IoU > 0.3）
        4. 横方向限定マージ（高さ比・縦重なり・横ギャップの3条件）
        5. アスペクト比フィルタ
        6. 各領域を切り抜いてPSM 7でOCR
        7. 座標を元画像スケールに変換して統合

    Args:
        verbose: True でMSER検出数・マージ判定ログを出力する
        debug  : dict を渡すと各段階のスナップショットを格納する（tests/eval_ocr.py 用）。
                 None のときは何も記録せず、処理内容は debug 引数がなかった頃と同一。
                 stages のbboxは _DETECT_SCALE 倍スケール座標のまま格納する
                 （元画像座標へ丸めると段階間の同一性判定がブレるため）。

                 debug = {
                     "scale":   float,                       # _DETECT_SCALE
                     "stages":  {段階名: [[x,y,w,h], ...]},   # 4倍スケール座標
                     "crops":   [{"bbox": [x,y,w,h],         # 4倍スケール座標
                                  "image": np.ndarray,       # OCRに渡した二値画像
                                  "raw": [{"text","conf"}],  # Tesseract生出力
                                  "accepted": int}],         # 非空で採用された数
                     "final":   [text_block, ...],           # 元画像座標
                     "ocr_calls": int,
                 }
    """
    from .preprocess import preprocess_for_ocr

    h, w = image.shape[:2]
    scale = _DETECT_SCALE
    sh, sw = int(h * scale), int(w * scale)
    scaled = cv2.resize(image, (sw, sh), interpolation=cv2.INTER_CUBIC)
    gray_scaled = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)

    if debug is not None:
        debug["scale"] = scale
        debug["stages"] = {}
        debug["crops"] = []

    def _snapshot(name: str, boxes: list) -> None:
        if debug is not None:
            debug["stages"][name] = [list(b) for b in boxes]
        if verbose:
            print(f"[{name}] remaining: {len(boxes)} regions")

    # Step 1: MSER で候補検出
    bboxes = _mser_detect_regions(gray_scaled)
    _snapshot("mser", bboxes)

    # Step 2: サイズフィルタ（_DETECT_SCALE 倍スケール基準）
    # 元画像 10px → 4x で 40px、元画像 80px → 4x で 320px
    bboxes = _filter_by_size(bboxes, min_h=20, max_h=320, min_w=10, max_w=800)
    _snapshot("size_filter", bboxes)

    # Step 3: NMS（MSERの重複除去）
    bboxes = _nms_regions(bboxes, iou_threshold=0.3)
    _snapshot("nms", bboxes)

    # Step 4: 横方向限定マージ
    bboxes = _merge_horizontal(
        bboxes, height_ratio=0.8, y_overlap_ratio=0.7, verbose=verbose
    )
    _snapshot("merge", bboxes)

    # Step 5: アスペクト比フィルタ（細くて横長の枠線を除外）
    bboxes = _filter_by_aspect(bboxes, min_h=15, max_wh_ratio=10.0)
    _snapshot("aspect_filter", bboxes)

    all_blocks = []
    PAD = 10  # 切り抜き時のパディング（_DETECT_SCALE 倍スケール上）

    # Step 6: 各領域を切り抜いてOCR
    for x, y, bw, bh in bboxes:
        x1 = max(0, x - PAD)
        y1 = max(0, y - PAD)
        x2 = min(sw, x + bw + PAD)
        y2 = min(sh, y + bh + PAD)

        crop = scaled[y1:y2, x1:x2]
        # 既に _DETECT_SCALE 倍済みなので追加拡大なし（scale=1.0）
        preprocessed, _ = preprocess_for_ocr(crop, scale=1.0)
        # 切り抜き = 1行テキスト想定なので PSM 7（single text line）
        ocr_debug = {} if debug is not None else None
        result = run_ocr(preprocessed, scale=1.0, config="--psm 7", debug=ocr_debug)

        if debug is not None:
            debug["crops"].append({
                "bbox": [x, y, bw, bh],
                "image": preprocessed,
                "raw": ocr_debug.get("raw", []),
                "accepted": len(result["text_blocks"]),
            })

        for block in result["text_blocks"]:
            tx, ty, tw, th = block["bbox"]
            # crop内座標 → scaled座標 → 元画像座標
            all_blocks.append({
                "text": block["text"],
                "bbox": [
                    int((tx + x1) / scale),
                    int((ty + y1) / scale),
                    int(tw / scale),
                    int(th / scale),
                ],
                "confidence": block["confidence"],
            })

    unique_blocks = _deduplicate(all_blocks)

    if debug is not None:
        debug["final"] = [dict(b) for b in unique_blocks]
        debug["ocr_calls"] = len(bboxes)

    mask = np.zeros((h, w), dtype=np.uint8)
    for block in unique_blocks:
        bx, by, bw, bh = block["bbox"]
        cv2.rectangle(mask, (bx, by), (bx + bw, by + bh), 255, -1)

    return {"text_blocks": unique_blocks, "mask": mask}
