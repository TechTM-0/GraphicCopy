import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

# 日本語フォント設定
for fname in ["Yu Gothic", "MS Gothic", "Meiryo"]:
    if any(fname.lower() in f.name.lower() for f in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = fname
        break

# --- データ準備 ---
img_bgr = cv2.imdecode(
    np.frombuffer(open(r"C:\Users\t-mur\GraphicCopy\samples\フロー.png", "rb").read(), np.uint8),
    cv2.IMREAD_COLOR
)
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
otsu_thresh, _ = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
T = int(otsu_thresh)

hist, _ = np.histogram(img_gray.ravel(), bins=256, range=(0, 256))
total = img_gray.size
bins = np.arange(256)

# 2グループの計算
n0 = hist[:T+1].sum()
n1 = hist[T+1:].sum()
P0 = n0 / total
P1 = n1 / total
m0 = np.dot(np.arange(T+1),     hist[:T+1])     / n0
m1 = np.dot(np.arange(T+1, 256), hist[T+1:])    / n1

# 各グループの標準偏差
vals0 = np.arange(T+1)
vals1 = np.arange(T+1, 256)
std0 = np.sqrt(np.dot(hist[:T+1],   (vals0 - m0)**2) / n0)
std1 = np.sqrt(np.dot(hist[T+1:],   (vals1 - m1)**2) / n1)

sigma_B2 = P0 * P1 * (m0 - m1)**2

# 全Tでσ²_B計算（③用）
sigma_all = np.zeros(256)
for t in range(1, 255):
    hn0 = hist[:t+1].sum()
    hn1 = hist[t+1:].sum()
    if hn0 == 0 or hn1 == 0:
        continue
    hm0 = np.dot(np.arange(t+1),    hist[:t+1])    / hn0
    hm1 = np.dot(np.arange(t+1,256), hist[t+1:])   / hn1
    sigma_all[t] = (hn0/total) * (hn1/total) * (hm0 - hm1)**2

# ============================
# Figure 1: 2グループの分布
# ============================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(f"大津の二値化 — 2グループの正体 (T={T})", fontsize=14, fontweight="bold")

# --- ① 黒グループの分布 ---
ax = axes[0]
ax.bar(np.arange(T+1), hist[:T+1], color="dimgray", alpha=0.85, width=1.0)
ax.axvline(m0, color="red", linewidth=2.5, linestyle="-", label=f"m₀ = {m0:.0f}")
ax.axvspan(max(0, m0-std0), min(255, m0+std0), alpha=0.15, color="red", label=f"±σ₀ = ±{std0:.0f}")
ax.set_xlim(0, 255)
ax.set_title(f"① 黒グループ (0〜{T})\n画素数: {n0:,}個 ({P0*100:.1f}%)", fontsize=11)
ax.set_xlabel("明るさ")
ax.set_ylabel("画素数")
ax.legend(fontsize=10)
ax.text(0.05, 0.90, f"平均 m₀ = {m0:.1f}", transform=ax.transAxes,
        fontsize=12, color="red", fontweight="bold")
ax.text(0.05, 0.80, f"標準偏差 σ₀ = {std0:.1f}", transform=ax.transAxes,
        fontsize=10, color="red")

# --- ② 白グループの分布 ---
ax = axes[1]
ax.bar(np.arange(T+1, 256), hist[T+1:], color="darkorange", alpha=0.85, width=1.0)
ax.axvline(m1, color="blue", linewidth=2.5, linestyle="-", label=f"m₁ = {m1:.0f}")
ax.axvspan(max(0, m1-std1), min(255, m1+std1), alpha=0.15, color="blue", label=f"±σ₁ = ±{std1:.0f}")
ax.set_xlim(0, 255)
ax.set_title(f"② 白グループ ({T+1}〜255)\n画素数: {n1:,}個 ({P1*100:.1f}%)", fontsize=11)
ax.set_xlabel("明るさ")
ax.set_ylabel("画素数")
ax.legend(fontsize=10)
ax.text(0.05, 0.90, f"平均 m₁ = {m1:.1f}", transform=ax.transAxes,
        fontsize=12, color="blue", fontweight="bold")
ax.text(0.05, 0.80, f"標準偏差 σ₁ = {std1:.1f}", transform=ax.transAxes,
        fontsize=10, color="blue")

# --- ③ 両グループ重ねて + 距離の可視化 ---
ax = axes[2]
ax.bar(np.arange(T+1),     hist[:T+1], color="dimgray",   alpha=0.75, width=1.0, label="黒グループ")
ax.bar(np.arange(T+1, 256), hist[T+1:], color="darkorange", alpha=0.75, width=1.0, label="白グループ")
ax.axvline(T, color="red", linewidth=2.0, linestyle="--", alpha=0.7, label=f"T={T}")
ax.axvline(m0, color="black",  linewidth=2.5, linestyle="-")
ax.axvline(m1, color="saddlebrown", linewidth=2.5, linestyle="-")

ymax = hist.max()
# m₀〜m₁ の距離を両矢印で表示
ax.annotate("", xy=(m1, ymax*0.92), xytext=(m0, ymax*0.92),
            arrowprops=dict(arrowstyle="<->", color="crimson", lw=2.5))
ax.text((m0+m1)/2, ymax*0.95, f"距離 = {m1-m0:.0f}",
        ha="center", fontsize=11, color="crimson", fontweight="bold")
ax.text(m0, ymax*0.70, f"m₀\n{m0:.0f}", ha="center", fontsize=10, color="black", fontweight="bold")
ax.text(m1, ymax*0.70, f"m₁\n{m1:.0f}", ha="center", fontsize=10, color="saddlebrown", fontweight="bold")

# σ²_B の値を表示
ax.text(0.03, 0.55,
        f"P₀={P0:.3f}, P₁={P1:.3f}\n"
        f"(m₀−m₁)² = {(m0-m1)**2:.0f}\n"
        f"σ²_B = {sigma_B2:.1f}",
        transform=ax.transAxes, fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", edgecolor="gray"),
        verticalalignment="center")

ax.set_xlim(0, 255)
ax.set_title("③ 2グループを重ねて比較\n「距離」がクラス間分散の核心", fontsize=11)
ax.set_xlabel("明るさ")
ax.set_ylabel("画素数")
ax.legend(fontsize=9, loc="upper left")

plt.tight_layout()
plt.savefig("otsu_groups.png", dpi=150, bbox_inches="tight")
print("保存完了: otsu_groups.png")
print(f"T={T}, m0={m0:.1f}, m1={m1:.1f}, P0={P0:.3f}, P1={P1:.3f}")
print(f"距離(m0-m1)={m0-m1:.1f}, sigma_B2={sigma_B2:.2f}")
plt.show()
