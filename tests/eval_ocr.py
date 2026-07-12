"""
eval_ocr.py - OCRパイプラインの評価ハーネス

「フロー.png 1枚を目視で見て手調整」を卒業し、変更のたびに全サンプルの数値で
良し悪しを判断するための物差し。CC パイプラインへの切替はこの数値で決める
（全サンプルで現行以上のときのみ MSER を削除する — fable 5提案.md 節6）。

使い方:
    .\.venv\Scripts\python.exe tests\eval_ocr.py

    # ベースラインとして tests/baseline.json に固定する
    .\.venv\Scripts\python.exe tests\eval_ocr.py --save-baseline

評価対象:
    samples/**/<名前>.expected.json があるサンプル画像のみ。
    期待値ファイルは lines（行テキスト＋bbox）と chars（文字＋bbox）を持つ。

出力:
    - コンソール: 指標テーブル ＋ 文字別の段階追跡テーブル
    - tests/debug/<名前>/stage_*.png : 段階別bboxオーバーレイ
    - tests/debug/<名前>/crops/      : OCRに渡した切り抜きとTesseract生出力
    - tests/baseline.json            : --save-baseline 指定時のみ
"""

import argparse
import glob
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, "src")
from graphiccopy.ocr import run_ocr_with_preprocess

# 段階追跡で表示する順序。パイプラインの実行順と一致させること。
STAGES = ["mser", "size_filter", "nms", "merge", "aspect_filter", "final"]

# 文字別追跡の列。final までは bbox の被覆、recognized は最終テキストへの出現。
TRACK_COLUMNS = STAGES + ["recognized"]

TIMING_RUNS = 3  # 1回だとキャッシュの影響を受けるため複数回測る

DEBUG_DIR = "tests/debug"
BASELINE_PATH = "tests/baseline.json"


# --- 補助 -------------------------------------------------------------------

def load_image(path: str) -> np.ndarray:
    """cv2.imread は Windows で日本語パスを読めないため imdecode を使う。"""
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def save_image(path: str, img: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imencode(".png", img)[1].tofile(path)


def levenshtein(a: str, b: str) -> int:
    """編集距離。認識一致率を「完全一致/不一致」ではなく部分点で測るために使う。"""
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def similarity(pred: str, gt: str) -> float:
    """1 - 正規化編集距離。「スート」と「」を同じ0点にしないため。"""
    if not gt:
        return 1.0 if not pred else 0.0
    return max(0.0, 1.0 - levenshtein(pred, gt) / max(len(pred), len(gt)))


def intersects(a: list, b: list) -> bool:
    """bbox [x, y, w, h] 同士が1px でも重なるか。"""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return not (ax + aw <= bx or bx + bw <= ax or ay + ah <= by or by + bh <= ay)


def contains_center(box: list, target: list) -> bool:
    """
    target の中心が box の内側にあるか（段階追跡の被覆判定）。

    IoU 閾値を使わない理由: MSER は1文字を複数の部分領域に分割することがあり、
    IoU では「文字の一部を検出できている」状態を未検出と誤判定してしまう。
    """
    bx, by, bw, bh = box
    tx, ty, tw, th = target
    cx, cy = tx + tw / 2, ty + th / 2
    return bx <= cx <= bx + bw and by <= cy <= by + bh


# --- 評価 -------------------------------------------------------------------

def evaluate(image: np.ndarray, expected: dict, debug: dict) -> dict:
    """1サンプルの指標を計算する。debug は run_ocr_with_preprocess が埋めたもの。"""
    gt_lines = expected["lines"]
    predicted = debug["final"]

    # 予測bboxを、重なるGT行に割り当てる（どのGT行とも重ならないものは誤検出）
    assigned = {i: [] for i in range(len(gt_lines))}
    false_positives = []
    for block in predicted:
        hit = [i for i, line in enumerate(gt_lines)
               if intersects(block["bbox"], line["bbox"])]
        if hit:
            assigned[hit[0]].append(block)
        else:
            false_positives.append(block)

    detected = sum(1 for i in assigned if assigned[i])
    detection_recall = detected / len(gt_lines) if gt_lines else 0.0

    # 認識一致率: 行内の予測テキストをx順に連結してGTと比較する
    per_line = []
    for i, line in enumerate(gt_lines):
        blocks = sorted(assigned[i], key=lambda b: b["bbox"][0])
        pred_text = "".join(b["text"] for b in blocks)
        sim = similarity(pred_text, line["text"])
        per_line.append({"gt": line["text"], "pred": pred_text, "similarity": sim})

    recognition = (sum(p["similarity"] for p in per_line) / len(per_line)
                   if per_line else 0.0)

    empty_crops = sum(1 for c in debug["crops"] if c["accepted"] == 0)

    return {
        "detection_recall": detection_recall,
        "recognition_rate": recognition,
        "false_positives": len(false_positives),
        "ocr_calls": debug["ocr_calls"],
        "empty_crops": empty_crops,
        "stage_counts": {s: len(debug["stages"][s]) for s in STAGES if s != "final"},
        "per_line": per_line,
        "false_positive_texts": [b["text"] for b in false_positives],
    }


def track_chars(expected: dict, debug: dict) -> list:
    """
    各GT文字が「どの段階まで生き残ったか」を追跡する。
    最初に × になった段階が消失の犯人（H1〜H4の決着）。

    最後の "recognized" 列は bbox の被覆ではなく、文字が最終テキストに現れたかを見る。
    bbox 被覆だけでは足りない: Tesseract が隣接文字を含む広い bbox を返した場合、
    「bbox は覆っているのにテキストには出ていない」文字を見逃す（実際に「タ」で起きた）。
    """
    scale = debug["scale"]
    output_text = "".join(b["text"] for b in debug["final"])
    rows = []
    for ch in expected.get("chars", []):
        x, y, w, h = ch["bbox"]
        scaled_box = [x * scale, y * scale, w * scale, h * scale]

        alive = {}
        for stage in STAGES:
            if stage == "final":
                # final の bbox は元画像座標なので元スケールで判定する
                boxes = [b["bbox"] for b in debug["final"]]
                target = ch["bbox"]
            else:
                boxes = debug["stages"][stage]
                target = scaled_box
            alive[stage] = any(contains_center(b, target) for b in boxes)

        alive["recognized"] = ch["char"] in output_text
        rows.append({"char": ch["char"], "alive": alive})
    return rows


# --- 出力 -------------------------------------------------------------------

def dump_debug_images(name: str, image: np.ndarray, expected: dict, debug: dict) -> None:
    """段階別オーバーレイと切り抜き画像を tests/debug/<name>/ に書き出す。"""
    scale = debug["scale"]
    out_dir = os.path.join(DEBUG_DIR, name)

    for idx, stage in enumerate(STAGES, 1):
        overlay = image.copy()

        # 期待値（GT行）を緑で先に描いて、検出結果と重ねて見えるようにする
        for line in expected["lines"]:
            x, y, w, h = line["bbox"]
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 160, 0), 1)

        if stage == "final":
            boxes = [b["bbox"] for b in debug["final"]]
        else:
            # 4倍スケール座標 → 元画像座標（描画のためだけの変換）
            boxes = [[int(x / scale), int(y / scale), int(w / scale), int(h / scale)]
                     for x, y, w, h in debug["stages"][stage]]

        for x, y, w, h in boxes:
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), 1)

        cv2.putText(overlay, f"{stage}: {len(boxes)}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        save_image(os.path.join(out_dir, f"stage_{idx}_{stage}.png"), overlay)

    # 切り抜きとTesseract生出力（空文字を返したものも残す = H4の証拠）
    lines = []
    for i, crop in enumerate(debug["crops"]):
        save_image(os.path.join(out_dir, "crops", f"crop_{i:02d}.png"), crop["image"])
        raw = " | ".join(f"{r['text']!r}({r['conf']:.0f})" for r in crop["raw"]) or "(空)"
        lines.append(f"crop_{i:02d}.png  bbox4x={crop['bbox']}  accepted={crop['accepted']}  raw={raw}")

    with open(os.path.join(out_dir, "crops", "raw_ocr.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def print_report(name: str, metrics: dict, timing: dict, chars: list) -> None:
    print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")

    print("\n[指標]")
    print(f"  検出再現率      : {metrics['detection_recall'] * 100:5.1f}%")
    print(f"  認識一致率      : {metrics['recognition_rate'] * 100:5.1f}%")
    print(f"  誤検出数        : {metrics['false_positives']}  {metrics['false_positive_texts']}")
    print(f"  OCR呼び出し回数 : {metrics['ocr_calls']}")
    print(f"  空振り回数      : {metrics['empty_crops']}")
    print(f"  処理時間        : 平均 {timing['mean']:.1f}s / 最大 {timing['max']:.1f}s")

    print("\n[行ごとの認識結果]")
    for line in metrics["per_line"]:
        print(f"  期待={line['gt']!r:12} 実際={line['pred']!r:14} 一致率={line['similarity'] * 100:5.1f}%")

    print("\n[段階別 bbox 件数]")
    print("  " + " → ".join(f"{s}:{n}" for s, n in metrics["stage_counts"].items()))

    if chars:
        print("\n[文字別 段階追跡]  final までは bbox の被覆・recognized は最終テキストへの出現")
        header = "  文字 | " + " | ".join(f"{s[:6]:>6}" for s in TRACK_COLUMNS)
        print(header)
        print("  " + "-" * (len(header) - 2))
        for row in chars:
            marks = " | ".join(f"{'○' if row['alive'][s] else '×':>6}" for s in TRACK_COLUMNS)
            lost = next((s for s in TRACK_COLUMNS if not row["alive"][s]), None)
            note = f"  ← {lost} で消失" if lost else ""
            print(f"   {row['char']}   | {marks}{note}")


# --- メイン -----------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-baseline", action="store_true",
                        help="結果を tests/baseline.json に固定する")
    args = parser.parse_args()

    expected_files = sorted(glob.glob("samples/**/*.expected.json", recursive=True))
    if not expected_files:
        print("ERROR: samples/**/*.expected.json が見つかりません")
        sys.exit(1)

    report = {}
    for exp_path in expected_files:
        with open(exp_path, encoding="utf-8") as f:
            expected = json.load(f)

        stem = os.path.basename(exp_path).replace(".expected.json", "")
        img_dir = os.path.dirname(exp_path)
        img_paths = [p for p in glob.glob(os.path.join(img_dir, stem + ".*"))
                     if not p.endswith(".json")]
        if not img_paths:
            print(f"WARNING: {stem} の画像が見つかりません（スキップ）")
            continue

        image = load_image(img_paths[0])
        print(f"\n実行中: {img_paths[0]}  ({image.shape[1]}x{image.shape[0]}px)")

        # 1回目は計装つき、残りは時間計測のみ（計装のオーバーヘッドを混ぜない）
        debug = {}
        durations = []
        for i in range(TIMING_RUNS):
            start = time.perf_counter()
            if i == 0:
                run_ocr_with_preprocess(image, debug=debug)
            else:
                run_ocr_with_preprocess(image)
            durations.append(time.perf_counter() - start)

        timing = {"mean": sum(durations) / len(durations), "max": max(durations),
                  "runs": [round(d, 2) for d in durations]}

        metrics = evaluate(image, expected, debug)
        chars = track_chars(expected, debug)
        dump_debug_images(stem, image, expected, debug)
        print_report(stem, metrics, timing, chars)

        report[stem] = {
            **metrics,
            "timing": timing,
            "char_tracking": {r["char"]: next((s for s in TRACK_COLUMNS if not r["alive"][s]), None)
                              for r in chars},
        }

    print(f"\nデバッグ画像: {DEBUG_DIR}/<名前>/")

    if args.save_baseline:
        with open(BASELINE_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"ベースラインを保存: {BASELINE_PATH}")


if __name__ == "__main__":
    main()
