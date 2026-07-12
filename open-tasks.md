# open-tasks

**正本は GitHub Issues。** このファイルはIssueを開かずに全体を見るためのダッシュボードで、内容は書かない（リンクと状態のみ）。

- 不変の情報（経緯・計測結果・なぜそう決めたか）→ **Issueのコメント**
- 現在有効な設計判断・既知の課題 → **[#9 設計判断ログ](https://github.com/TechTM-0/GraphicCopy/issues/9) の本文**（上書き型）
- 現在のアルゴリズム → **`docs/ロジック解説.md`**（コードと同じコミットで更新）

最終更新: 2026-07-12

---

## 現在地

段階1完了。「タ」「ー」消失の原因が確定した（H4 / H5）。CC刷新の方針は変更なし。

- ベースライン固定済み: 検出再現率100% / 認識一致率66.7% / 誤検出1 / 6.0秒（`tests/baseline.json`）
- 「ー」= H4: 単独クロップでTesseractが空文字を返す
- 「タ」= H5: `_deduplicate` が隣接文字を重複と誤認して除去

## 次の一手

**[#3 段階2: `detect.py` 新規実装（A〜D）](https://github.com/TechTM-0/GraphicCopy/issues/3)**

インク抽出・CC分析・2パス分類・行グルーピングを実装する。完了条件はフロー.pngで全9文字のbbox取得。着手前に実装方針の説明と承認。

## 残タスク

| # | タスク | 状態 |
|---|---|---|
| [#2](https://github.com/TechTM-0/GraphicCopy/issues/2) | CC検出パイプラインへの刷新（親） | open |
| [#3](https://github.com/TechTM-0/GraphicCopy/issues/3) | 段階2: `detect.py` 新規実装（A〜D） | **次の一手** |
| [#4](https://github.com/TechTM-0/GraphicCopy/issues/4) | 段階3: E〜G接続（行認識・信頼度3帯域ゲート） | open |
| [#5](https://github.com/TechTM-0/GraphicCopy/issues/5) | 段階4: ピクセルマスク化 + empty_frame警告 | open |
| [#6](https://github.com/TechTM-0/GraphicCopy/issues/6) | 段階5: MSER系コード削除 | open |
| [#7](https://github.com/TechTM-0/GraphicCopy/issues/7) | 評価サンプルの追加（紙の写真を必ず1枚） | open |
| [#9](https://github.com/TechTM-0/GraphicCopy/issues/9) | 設計判断ログ（正本・**クローズしない**） | 常設 |

## Phase 別ロードマップ

- **Phase 1（進行中）**: Pythonプロトタイプ — 前処理・テキスト検出（CC刷新中 #2）・マスク生成・図形抽出（直線・円）・SVG出力
- **Phase 2**: 空間関係推定 — テキスト→図形の所属判定・包含/接触関係
- **Phase 3**: 複雑図形対応 — 形状ヒント・楕円/多角形/自由曲線・矢印/有向エッジ
- **Phase 4**: 展開 — Web版（WASM）・モバイル版

## 完了

| # | タスク | 完了日 |
|---|---|---|
| [#1](https://github.com/TechTM-0/GraphicCopy/issues/1) | 段階1: 評価ハーネス＋段階別計装＋ベースライン計測 | 2026-07-12 |
| [#8](https://github.com/TechTM-0/GraphicCopy/issues/8) | マスク生成の検証 | 2026-06-10 |

## 保留

- ドキュメント自動更新フック（git commit の PostToolUse フックで更新漏れを検知する仕組み）
