"""
大津の二値化 — 全プロセス可視化スクリプト
資料用途（PNG出力）
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm
from matplotlib.patches import FancyArrowPatch

# ── フォント ──────────────────────────────────────────────────────────────────
for fname in ["Yu Gothic", "MS Gothic", "Meiryo"]:
    if any(fname.lower() in f.name.lower() for f in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = fname
        break

# ── スタイル ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "grid.linewidth":    0.6,
    "xtick.direction":   "out",
    "ytick.direction":   "out",
})

# ── カラー定義 ────────────────────────────────────────────────────────────────
C_BLK   = "#34495e"   # 黒グループ
C_WHT   = "#e67e22"   # 白グループ
C_T     = "#e74c3c"   # T線
C_M0    = "#c0392b"   # m₀
C_M1    = "#2980b9"   # m₁
C_SIGMA = "#8e44ad"   # σ帯
C_DIST  = "#6c3483"   # 距離矢印

# ── 合成データ生成（説明用：黒・白グループを均等に配置）────────────────────
np.random.seed(42)
N_EACH = 50_000   # 各グループの画素数

# 黒グループ: 平均80, 標準偏差20 / 白グループ: 平均190, 標準偏差20
dark_vals  = np.clip(np.random.normal(80,  20, N_EACH), 0, 255).astype(np.uint8)
light_vals = np.clip(np.random.normal(190, 20, N_EACH), 0, 255).astype(np.uint8)

# 合成グレースケール画像（316×316）
side = 316
all_vals = np.concatenate([dark_vals, light_vals])
np.random.shuffle(all_vals)
img_gray = all_vals[:side*side].reshape(side, side)

# 大津法でしきい値計算
otsu_thresh, img_otsu = cv2.threshold(
    img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
)
T = int(otsu_thresh)

# 元画像（説明用に左半分=黒・右半分=白の合成画像）
img_display = np.zeros((side, side), dtype=np.uint8)
img_display[:, :side//2] = dark_vals[:side*(side//2)].reshape(side, side//2)
img_display[:, side//2:] = light_vals[:side*(side//2)].reshape(side, side//2)

# ── ヒストグラム & 統計量 ─────────────────────────────────────────────────────
hist, _ = np.histogram(img_gray.ravel(), bins=256, range=(0, 256))
total   = img_gray.size
bins    = np.arange(256)

n0  = hist[:T+1].sum();  n1  = hist[T+1:].sum()
P0  = n0 / total;        P1  = n1 / total
m0  = np.dot(np.arange(T+1),     hist[:T+1])    / n0
m1  = np.dot(np.arange(T+1,256), hist[T+1:])    / n1
std0 = np.sqrt(np.dot(hist[:T+1],  (np.arange(T+1)     - m0)**2) / n0)
std1 = np.sqrt(np.dot(hist[T+1:],  (np.arange(T+1,256) - m1)**2) / n1)
sigma_B2 = P0 * P1 * (m0 - m1)**2

# ── 全 T でクラス間分散 ───────────────────────────────────────────────────────
sigma_all = np.zeros(256)
for t in range(1, 255):
    hn0 = hist[:t+1].sum();  hn1 = hist[t+1:].sum()
    if hn0 == 0 or hn1 == 0:
        continue
    hm0 = np.dot(np.arange(t+1),    hist[:t+1])    / hn0
    hm1 = np.dot(np.arange(t+1,256), hist[t+1:])   / hn1
    sigma_all[t] = (hn0/total) * (hn1/total) * (hm0 - hm1)**2

# ── ガウス平滑化（曲線用）────────────────────────────────────────────────────
def gauss_smooth(h, sigma=2.0):
    k = int(sigma * 4)
    x = np.arange(-k, k+1)
    g = np.exp(-x**2 / (2*sigma**2)); g /= g.sum()
    return np.convolve(h.astype(float), g, mode="same")

hist_smooth = gauss_smooth(hist)
h0 = hist.copy().astype(float); h0[T+1:] = 0
h1 = hist.copy().astype(float); h1[:T+1] = 0
curve0 = gauss_smooth(h0)
curve1 = gauss_smooth(h1)

# ── レイアウト ────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 13))
fig.suptitle("大津の二値化 — 全プロセス可視化", fontsize=16,
             fontweight="bold", y=0.99)

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.38,
                       top=0.95, bottom=0.06, left=0.07, right=0.97)

# ════════════════════════════════════════════════════════
# 行 1 ： 画像 3 枚
# ════════════════════════════════════════════════════════
for col, (img, title) in enumerate([
    (img_display, "① 合成入力画像\n（左=黒グループ / 右=白グループ）"),
    (img_gray,    "② ランダム配置（実際の計算対象）"),
    (img_otsu,    f"③ 大津法 二値化  (T = {T})"),
]):
    ax = fig.add_subplot(gs[0, col])
    kw = dict(cmap="gray") if img.ndim == 2 else {}
    ax.imshow(img, **kw)
    ax.set_title(title, fontsize=11, pad=6)
    ax.axis("off")

# ════════════════════════════════════════════════════════
# 行 2 ： ④ ヒストグラム  +  ⑤ σ²_B カーブ
# ════════════════════════════════════════════════════════

# ── ④ ────────────────────────────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 0:2])
ax4.bar(bins, hist, width=1.0, color="#aed6f1", alpha=0.7, zorder=1)
ax4.plot(bins, hist_smooth, color="#1a5276", linewidth=1.6, zorder=2, label="平滑化曲線")
ax4.fill_between(bins, hist_smooth, where=(bins <= T),
                 alpha=0.30, color=C_BLK, zorder=3, label=f"黒グループ  P₀={P0:.3f}")
ax4.fill_between(bins, hist_smooth, where=(bins >  T),
                 alpha=0.25, color=C_WHT, zorder=3, label=f"白グループ  P₁={P1:.3f}")
ax4.axvline(T, color=C_T, linewidth=2.2, linestyle="--", zorder=4, label=f"最適 T = {T}")

ymax = hist.max()
# m₀ 注釈（黒グループ内のピーク付近）
ax4.annotate(f"m₀ = {m0:.0f}", xy=(m0, hist[int(m0)]),
             xytext=(m0 - 45, ymax * 0.75),
             fontsize=10, color=C_M0, fontweight="bold",
             arrowprops=dict(arrowstyle="->", color=C_M0, lw=1.5))
# m₁ 注釈（255ピーク）
ax4.annotate(f"m₁ = {m1:.0f}", xy=(255, hist[255]),
             xytext=(220, ymax * 0.75),
             fontsize=10, color=C_M1, fontweight="bold",
             arrowprops=dict(arrowstyle="->", color=C_M1, lw=1.5))

ax4.set_title("④ ヒストグラム — しきい値 T で黒グループ・白グループに分割",
              fontsize=11, pad=7)
ax4.set_xlabel("明るさ  (0 = 黒, 255 = 白)", fontsize=10)
ax4.set_ylabel("画素数", fontsize=10)
ax4.set_xlim(-2, 257)
ax4.legend(fontsize=8.5, loc="upper left", framealpha=0.9)

# ── ⑤ ────────────────────────────────────────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 2])
ax5.plot(sigma_all, color="#2471a3", linewidth=1.8, zorder=2)
ax5.fill_between(np.arange(256), sigma_all, alpha=0.15, color="#2471a3", zorder=1)
ax5.axvline(T, color=C_T, linewidth=2.2, linestyle="--", zorder=3, label=f"最大  T = {T}")
ax5.scatter([T], [sigma_all[T]], color=C_T, s=70, zorder=4)
ax5.annotate(f"最大値\n{sigma_all[T]:.1f}",
             xy=(T, sigma_all[T]), xytext=(T - 60, sigma_all[T] * 0.80),
             fontsize=9, color=C_T, fontweight="bold",
             arrowprops=dict(arrowstyle="->", color=C_T, lw=1.3))
ax5.set_title("⑤ クラス間分散  σ²_B(T)\n全 T を試して最大を探す", fontsize=10, pad=7)
ax5.set_xlabel("しきい値  T", fontsize=10)
ax5.set_ylabel("σ²_B(T)", fontsize=10)
ax5.legend(fontsize=9)

# ════════════════════════════════════════════════════════
# 行 3 ： ⑥ 2 グループの分布曲線（破断軸）
# ════════════════════════════════════════════════════════
gs_bot = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=gs[2, :], width_ratios=[2, 1], wspace=0.07
)
ax_L = fig.add_subplot(gs_bot[0])
ax_R = fig.add_subplot(gs_bot[1])

# ── 左パネル：黒グループ ──────────────────────────────────────────────────────
x0 = np.arange(T + 1)
c0 = curve0[:T+1]
yL_max = c0.max()

ax_L.fill_between(x0, c0, alpha=0.35, color=C_BLK, zorder=2)
ax_L.plot(x0, c0, color=C_BLK, linewidth=2.2, zorder=3,
          label=f"黒グループ   {n0:,} 画素  ({P0*100:.1f}%)")
# σ₀ 帯
ax_L.axvspan(max(0, m0-std0), min(T, m0+std0),
             alpha=0.15, color=C_SIGMA, zorder=1, label=f"±σ₀ = ±{std0:.0f}")
# m₀ 線
ax_L.axvline(m0, color=C_M0, linewidth=2.0, linestyle="--", zorder=4)
ax_L.text(m0 + 3, yL_max * 0.88,
          f"m₀ = {m0:.0f}\nσ₀ = {std0:.0f}",
          fontsize=10, color=C_M0, fontweight="bold", va="top",
          bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                    edgecolor=C_M0, alpha=0.9))
# T=175 境界線（右端）
ax_L.axvline(T, color=C_T, linewidth=1.5, linestyle=":", zorder=4, alpha=0.8)
ax_L.text(T - 3, yL_max * 0.35, f"T = {T}\n(境界)",
          fontsize=8.5, color=C_T, ha="right", va="center",
          bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                    edgecolor=C_T, alpha=0.8))

ax_L.set_xlim(-3, T + 5)
ax_L.set_ylim(0, yL_max * 1.28)
ax_L.spines["right"].set_visible(False)
ax_L.tick_params(right=False)
ax_L.set_xlabel("明るさ  (0 = 黒)", fontsize=10)
ax_L.set_ylabel("画素数", fontsize=10)
ax_L.legend(fontsize=9, loc="upper right", framealpha=0.9)

# 式ボックス
formula = (
    "σ²_B = P₀ × P₁ × (m₀ − m₁)²\n"
    f"     = {P0:.3f} × {P1:.3f} × {int((m0-m1)**2):,}\n"
    f"     = {sigma_B2:.1f}"
)
ax_L.text(0.02, 0.46, formula, transform=ax_L.transAxes, fontsize=9,
          va="top", family="monospace",
          bbox=dict(boxstyle="round,pad=0.5", facecolor="#fef9e7",
                    edgecolor="#d4ac0d", linewidth=1.2))

# ── 右パネル：白グループ（ズームイン）────────────────────────────────────────
x_lo = max(T + 1, int(m1 - 6 * max(std1, 3)))
x1   = np.arange(x_lo, 256)
c1   = curve1[x_lo:]
yR_max = c1.max()

ax_R.fill_between(x1, c1, alpha=0.35, color=C_WHT, zorder=2)
ax_R.plot(x1, c1, color=C_WHT, linewidth=2.2, zorder=3,
          label=f"白グループ   {n1:,} 画素  ({P1*100:.1f}%)")
# σ₁ 帯
ax_R.axvspan(max(x_lo, m1-std1), min(255, m1+std1),
             alpha=0.20, color=C_SIGMA, zorder=1, label=f"±σ₁ = ±{std1:.1f}")
# m₁ 線
ax_R.axvline(m1, color=C_M1, linewidth=2.0, linestyle="--", zorder=4)
ax_R.text(m1 - 1, yR_max * 0.88,
          f"m₁ = {m1:.0f}\nσ₁ = {std1:.1f}",
          fontsize=10, color=C_M1, fontweight="bold", va="top", ha="right",
          bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                    edgecolor=C_M1, alpha=0.9))
# T=175 の境界は画面外→注釈で示す
ax_R.text(x_lo + 0.5, yR_max * 0.15,
          f"← T={T} の境界より右",
          fontsize=8.5, color=C_T, va="center",
          bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                    edgecolor=C_T, alpha=0.8))

ax_R.set_xlim(x_lo - 1, 256)
ax_R.set_ylim(0, yR_max * 1.28)
ax_R.spines["left"].set_visible(False)
ax_R.yaxis.set_label_position("right")
ax_R.yaxis.tick_right()
ax_R.tick_params(left=False)
ax_R.set_xlabel(f"明るさ  ({x_lo}〜255 ズームイン)", fontsize=10)
ax_R.set_ylabel("画素数", fontsize=10, rotation=270, labelpad=16)
ax_R.legend(fontsize=9, loc="upper left", framealpha=0.9)

# ── 破断マーク ────────────────────────────────────────────────────────────────
d = 0.018
for ax_side, xp in [(ax_L, 1), (ax_R, 0)]:
    kw = dict(transform=ax_side.transAxes, color="k",
              clip_on=False, lw=1.8, zorder=10)
    ax_side.plot((xp-d, xp+d), (-d, +d), **kw)
    ax_side.plot((xp-d, xp+d), (1-d, 1+d), **kw)

# ── 距離矢印（figure座標で m₀→m₁ を跨いで描画）────────────────────────────
fig.canvas.draw()  # bbox を確定させる

def to_fig(ax, x_data, y_frac):
    """データ座標 x_data, 軸内比率 y_frac → figure 座標"""
    xlim = ax.get_xlim()
    x_frac = (x_data - xlim[0]) / (xlim[1] - xlim[0])
    pos = ax.get_position()
    return (pos.x0 + x_frac * pos.width,
            pos.y0 + y_frac * pos.height)

x_m0_fig, y_arr_fig = to_fig(ax_L, m0, 1.10)
x_m1_fig, _         = to_fig(ax_R, m1, 1.10)

ax_arr = fig.add_axes([0, 0, 1, 1], facecolor="none")
ax_arr.set_xlim(0, 1); ax_arr.set_ylim(0, 1); ax_arr.axis("off")
ax_arr.annotate(
    "", xy=(x_m1_fig, y_arr_fig), xytext=(x_m0_fig, y_arr_fig),
    arrowprops=dict(arrowstyle="<->", color=C_DIST, lw=2.2),
    xycoords="figure fraction", textcoords="figure fraction"
)
ax_arr.text((x_m0_fig + x_m1_fig) / 2, y_arr_fig + 0.012,
            f"距離  (m₁ − m₀)  =  {m1-m0:.0f}",
            ha="center", va="bottom", fontsize=11,
            color=C_DIST, fontweight="bold",
            transform=ax_arr.transAxes)

# ── ⑥ タイトル ───────────────────────────────────────────────────────────────
ax_L.set_title(
    "⑥ 2グループの分布曲線（左: 黒グループ  /  右: 白グループ ズームイン）",
    fontsize=11, pad=7, loc="left"
)

# ════════════════════════════════════════════════════════
# 出力
# ════════════════════════════════════════════════════════
plt.savefig("otsu_visualization.png", dpi=150, bbox_inches="tight")
print("保存完了: otsu_visualization.png")
print(f"T={T}, m0={m0:.1f}(σ={std0:.1f}), m1={m1:.1f}(σ={std1:.1f})")
print(f"P0={P0:.4f}, P1={P1:.4f}, sigma_B2={sigma_B2:.2f}")
