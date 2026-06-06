"""
第3章 多孔位接触力学机理与六维风险建模 — 数据处理与图表生成
=================================================================
生成四个核心图：
  图3-1: 9针充电口布局俯视图（基于 charging_port_model.csv）
  图3-2: 三类针型楔紧/卡滞临界参数对比
  图3-3: 接触集合演化与六维风险分量递进
  图3-4: 安全多面体二维截面

用法: .yolo_env/Scripts/python.exe docs/chapter3/chapter3_figures.py
输出: docs/chapter3/figures/ 目录
"""

from __future__ import annotations

import json
import csv
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter

# ---- 中文支持 ----
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "WenQuanYi Micro Hei"]
plt.rcParams["axes.unicode_minus"] = False


ROOT_DIR = Path(__file__).resolve().parents[2]
FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

MODEL_CSV = ROOT_DIR / "config" / "charging_port_model.csv"

DPI = 200
FIGSIZE_STANDARD = (8, 5)
FIGSIZE_WIDE = (10, 5.5)


# ============================================================
# 数据加载
# ============================================================

@dataclass
class PinSpec:
    """单根插针的几何与类别参数"""
    name: str           # 模型点名
    x: float            # 系下 x (mm)
    y: float            # 系下 y (mm)
    diameter: float     # 针径 (mm)
    length_eff: float   # 有效配合长度 (mm)
    clearance: float    # 径向间隙 c (mm)
    pin_type: str       # "大针" / "中针" / "小针"
    z_order: int        # 耦合顺序（按 GB/T 20234.3）
    stage: int          # 首次入孔阶段 S1~S5
    tray: str           # 所属触头组

    @property
    def radius(self) -> float:
        return self.diameter / 2.0

    @property
    def theta_crit_deg(self) -> float:
        """楔紧临界倾角 (°) — 小角度近似"""
        return float(np.degrees(np.arctan(2 * self.clearance / self.length_eff)))

    @property
    def mu_crit(self) -> float:
        """卡滞临界摩擦系数"""
        return 2 * self.clearance / self.length_eff

    @property
    def position_vec(self) -> np.ndarray:
        return np.array([self.x, self.y, 0.0], dtype=np.float64)


def load_pin_specs(csv_path: Path) -> List[PinSpec]:
    """从 charging_port_model.csv 加载针位，并补充 GB/T 20234.3 参数"""
    # 各触头的 GB/T 20234.3 参数
    spec_table = {
        "DC_neg":  dict(d=10.3, L=29, c=0.090, ptype="大针", z=2, stage=2, tray="DC"),
        "DC_pos":  dict(d=10.3, L=29, c=0.090, ptype="大针", z=3, stage=2, tray="DC"),
        "A_neg":   dict(d=6.0,  L=35, c=0.075, ptype="中针", z=7, stage=4, tray="A"),
        "A_pos":   dict(d=6.0,  L=35, c=0.075, ptype="中针", z=8, stage=4, tray="A"),
        "PE":      dict(d=6.0,  L=35, c=0.075, ptype="中针", z=1, stage=1, tray="PE"),
        "S_neg":   dict(d=3.0,  L=40, c=0.060, ptype="小针", z=5, stage=5, tray="S"),
        "S_pos":   dict(d=3.0,  L=40, c=0.060, ptype="小针", z=6, stage=5, tray="S"),
        "CC1":     dict(d=3.0,  L=40, c=0.060, ptype="小针", z=9, stage=5, tray="CC"),
        "CC2":     dict(d=3.0,  L=40, c=0.060, ptype="小针", z=4, stage=3, tray="CC"),
    }

    pins = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["label"].strip()
            x = float(row["x_mm"])
            y = float(row["y_mm"])
            spec = spec_table.get(name)
            if spec is None:
                print(f"  [WARN] {name} 无 GB/T 参数，跳过")
                continue
            pins.append(PinSpec(
                name=name, x=x, y=y,
                diameter=spec["d"],
                length_eff=spec["L"],
                clearance=spec["c"],
                pin_type=spec["ptype"],
                z_order=spec["z"],
                stage=spec["stage"],
                tray=spec["tray"],
            ))

    pins.sort(key=lambda p: p.z_order)
    return pins


# ============================================================
# 图 3-1: 9 针充电口布局俯视图
# ============================================================

def fig_3_1_pin_layout(pins: List[PinSpec]):
    """充电插头 9 针布局俯视图，按针型着色并标注坐标系"""
    fig, ax = plt.subplots(figsize=(7, 7))

    type_colors = {"大针": "#E63946", "中针": "#457B9D", "小针": "#2A9D8F"}
    type_edge = {"大针": "#B71C1C", "中针": "#1D3557", "小针": "#1B5E20"}

    # 画插头外轮廓（示意）
    body = Circle((0, 0), 28, fill=False, edgecolor="#666", linewidth=2,
                  linestyle="--", zorder=0)
    ax.add_patch(body)

    # 画各针
    for p in pins:
        color = type_colors[p.pin_type]
        edge = type_edge[p.pin_type]
        # 画间隙圆（内圈）
        clearance_circle = Circle((p.x, p.y), p.clearance, fill=True,
                                  facecolor=color, alpha=0.15, edgecolor="none", zorder=1)
        ax.add_patch(clearance_circle)
        # 画针截面
        pin_circle = Circle((p.x, p.y), p.radius / 2.0, fill=True,
                            facecolor=color, edgecolor=edge, linewidth=1.5,
                            alpha=0.85, zorder=2)
        ax.add_patch(pin_circle)
        # 标注
        offset = 2.2 if p.y >= 0 else -3.0
        ax.annotate(p.name.replace("_", "\n"), (p.x, p.y),
                    textcoords="offset points", xytext=(0, offset),
                    ha="center", fontsize=7, fontweight="bold",
                    color=edge, zorder=5)

    # 坐标系标注
    ax.arrow(-22, -24, 8, 0, head_width=0.8, head_length=1.2, fc="black", lw=2, zorder=6)
    ax.arrow(-22, -24, 0, 8, head_width=0.8, head_length=1.2, fc="black", lw=2, zorder=6)
    ax.text(-13, -24.5, "x", fontsize=11, fontweight="bold")
    ax.text(-23, -15,   "y", fontsize=11, fontweight="bold")

    # 插入方向标注
    ax.annotate("插入方向 (z轴)\n垂直纸面向内", xy=(0, 26), ha="center", fontsize=9,
                color="#555", style="italic")

    # 图例
    legend_handles = [
        mpatches.Patch(color=type_colors[t], alpha=0.85, label=f"{t} (Φ={d}mm)")
        for t, d in [("大针", 10.3), ("中针", 6.0), ("小针", 3.0)]
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9,
              title="针型分类", title_fontsize=10)

    ax.set_xlim(-28, 28)
    ax.set_ylim(-28, 28)
    ax.set_aspect("equal")
    ax.set_xlabel("x / mm", fontsize=11)
    ax.set_ylabel("y / mm", fontsize=11)
    ax.set_title("充电插头 9 针布局俯视图 (插座坐标系 {S})", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_1_pin_layout.png", dpi=DPI, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig3_1_pin_layout.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[OK] 图3-1 已保存: pin_layout")


# ============================================================
# 图 3-2: 三类针型楔紧/卡滞临界参数对比
# ============================================================

def fig_3_2_critical_params(pins: List[PinSpec]):
    """双轴柱状图 + 参考线，展示 θ_crit 与 μ_crit"""
    # 按类型聚合
    types_order = ["大针", "中针", "小针"]
    type_labels = {"大针": "大针\n(DC+, DC−)", "中针": "中针\n(A+, A−, PE)", "小针": "小针\n(S+, S−, CC1, CC2)"}

    agg = {}
    for pt in types_order:
        group = [p for p in pins if p.pin_type == pt]
        agg[pt] = {
            "label": type_labels[pt],
            "theta_avg": float(np.mean([p.theta_crit_deg for p in group])),
            "mu_avg": float(np.mean([p.mu_crit for p in group])),
            "diameter": group[0].diameter,
            "length": group[0].length_eff,
            "clearance": group[0].clearance,
            "pins": [p.name for p in group],
        }

    fig, ax1 = plt.subplots(figsize=FIGSIZE_WIDE)

    x = np.arange(len(types_order))
    width = 0.35
    colors_theta = ["#E63946", "#457B9D", "#2A9D8F"]

    theta_vals = [agg[t]["theta_avg"] for t in types_order]
    mu_vals = [agg[t]["mu_avg"] for t in types_order]

    # 左轴: θ_crit
    bars1 = ax1.bar(x - width / 2, theta_vals, width, color=colors_theta,
                    edgecolor="white", linewidth=0.8, zorder=3)
    ax1.set_ylabel("楔紧临界倾角 θ_crit / (°)", fontsize=11, color="#333")
    ax1.set_ylim(0, max(theta_vals) * 1.35)
    ax1.tick_params(axis="y", labelcolor="#333")

    # 数值标注
    for bar, val in zip(bars1, theta_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                 f"{val:.3f}°", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # 右轴: μ_crit
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width / 2, mu_vals, width, color=["#F4A261", "#A8DADC", "#E9C46A"],
                    edgecolor="white", linewidth=0.8, zorder=3)
    ax2.set_ylabel("卡滞临界摩擦系数 μ_crit", fontsize=11, color="#333")
    ax2.tick_params(axis="y", labelcolor="#333")

    for bar, val in zip(bars2, mu_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.00015,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # 实际摩擦系数参考线
    mu_actual = 0.15
    ax2.axhline(y=mu_actual, color="#D62828", linewidth=2, linestyle="--", zorder=4)
    ax2.text(2.35, mu_actual + 0.005, f"实测 μ ≈ {mu_actual}", color="#D62828",
             fontsize=10, fontweight="bold", ha="right")

    # 量级差异标注
    ax2.annotate("", xy=(1.0, mu_actual), xytext=(1.0, max(mu_vals) * 1.3),
                 arrowprops=dict(arrowstyle="<->", color="#D62828", lw=1.5))
    ax2.text(1.15, (mu_actual + max(mu_vals) * 1.3) / 2,
             "≈ 1.5 个数量级", fontsize=9, color="#D62828",
             va="center", fontstyle="italic")

    ax1.set_xticks(x)
    ax1.set_xticklabels([agg[t]["label"] for t in types_order], fontsize=10)
    ax1.set_xlabel("针型分类", fontsize=11)

    # 图例
    l1 = mpatches.Patch(color="#888", label=f"θ_crit (楔紧临界倾角)")
    l2 = mpatches.Patch(color="#BBB", label=f"μ_crit (卡滞临界 μ)")
    ax1.legend(handles=[l1, l2], loc="upper left", fontsize=9)

    ax1.set_title("GB/T 20234.3 三类针型楔紧—卡滞临界参数对比", fontsize=13, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3, zorder=0)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_2_critical_params.png", dpi=DPI, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig3_2_critical_params.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[OK] 图3-2 已保存: critical_params")


# ============================================================
# 图 3-3: 接触集合演化与风险分量递进
# ============================================================

def fig_3_3_risk_evolution(pins: List[PinSpec]):
    """按 S0-S5 阶段，展示接触集合增长与六维风险分量递进"""

    stages = ["S0", "S1", "S2", "S3", "S4", "S5"]
    stage_labels = ["S0\n自由", "S1\nPE入孔", "S2\nDC对入孔", "S3\nCC2入孔", "S4\nA对入孔", "S5\n全针耦合"]

    # 每阶段的激活针位
    stage_pins = {
        "S0": [],
        "S1": ["PE"],
        "S2": ["PE", "DC_neg", "DC_pos"],
        "S3": ["PE", "DC_neg", "DC_pos", "CC2"],
        "S4": ["PE", "DC_neg", "DC_pos", "CC2", "A_neg", "A_pos"],
        "S5": ["PE", "DC_neg", "DC_pos", "CC2", "A_neg", "A_pos", "S_neg", "S_pos", "CC1"],
    }

    def compute_risk_for_stage(active_names: List[str]) -> np.ndarray:
        """基于激活针位计算六维风险向量（模拟典型残余误差场景）"""
        active = [p for p in pins if p.name in active_names]
        n = len(active)
        if n == 0:
            return np.zeros(6)

        # 构造残余位姿误差（模拟：+0.3mm 横向 + 0.15° 倾斜）
        dp_mm = np.array([0.3, -0.2, 0.0], dtype=np.float64)
        dtheta_deg = np.array([0.10, -0.15, 0.08], dtype=np.float64)
        dtheta_rad = np.deg2rad(dtheta_deg)

        # 计算各方向上的风险
        risks = np.zeros(6)

        # r1, r2: 横向力风险（与 x/y 平移误差 + 力臂效应相关）
        max_x = max(abs(p.x) for p in active) if active else 1
        max_y = max(abs(p.y) for p in active) if active else 1
        risks[0] = min(1.0, abs(dp_mm[0]) / (0.8 * max(p.clearance for p in active)) * (1 + n / 9))
        risks[1] = min(1.0, abs(dp_mm[1]) / (0.8 * max(p.clearance for p in active)) * (1 + n / 9))

        # r3: 轴向力风险（随接触数目增大）
        risks[2] = min(1.0, 0.05 + 0.12 * n)

        # r4, r5: 力矩风险（力臂 × 倾角）
        risks[3] = min(1.0, abs(dtheta_rad[0]) / np.deg2rad(min(p.theta_crit_deg for p in active)) * (1 + max_x / 20))
        risks[4] = min(1.0, abs(dtheta_rad[1]) / np.deg2rad(min(p.theta_crit_deg for p in active)) * (1 + max_y / 20))

        # r6: 绕 z 轴扭转风险
        risks[5] = min(1.0, abs(dtheta_rad[2]) / np.deg2rad(min(p.theta_crit_deg for p in active)) * n / 9)

        return risks

    risk_matrix = np.array([compute_risk_for_stage(stage_pins[s]) for s in stages])

    # ---- 子图 A: 接触集合增长 ----
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 5.2),
                                      gridspec_kw={"width_ratios": [1, 1.3]})

    # 针位激活热图
    all_names = [p.name for p in pins]
    n_stages = len(stages)
    heatmap = np.zeros((len(all_names), n_stages))
    for j, s in enumerate(stages):
        for i, name in enumerate(all_names):
            heatmap[i, j] = 1 if name in stage_pins[s] else 0

    im = ax_a.imshow(heatmap, cmap="Blues", aspect="auto", vmin=0, vmax=1, alpha=0.85)
    ax_a.set_xticks(range(n_stages))
    ax_a.set_xticklabels(stage_labels, fontsize=9)
    ax_a.set_yticks(range(len(all_names)))
    ax_a.set_yticklabels(all_names, fontsize=9)
    ax_a.set_title("接触集合 A(z, δp) 演化", fontsize=12, fontweight="bold")

    for i in range(len(all_names)):
        for j in range(n_stages):
            if heatmap[i, j] == 1:
                ax_a.text(j, i, "●", ha="center", va="center", fontsize=8, color="#B71C1C")

    # 激活数量曲线
    ax_a2 = ax_a.twiny()
    active_counts = [len(stage_pins[s]) for s in stages]
    ax_a2.plot(range(n_stages), active_counts, "o-", color="#E63946", lw=2, markersize=8, zorder=5)
    ax_a2.set_xlim(-0.5, n_stages - 0.5)
    ax_a2.set_xticks([])
    for j, cnt in enumerate(active_counts):
        ax_a2.text(j, cnt + 0.35, f"|A|={cnt}", ha="center", fontsize=9,
                   color="#E63946", fontweight="bold")

    # ---- 子图 B: 六维风险分量递进 ----
    colors_risk = ["#E63946", "#F4A261", "#2A9D8F", "#457B9D", "#6D597A", "#D62828"]
    labels_risk = ["r1 (Fx)", "r2 (Fy)", "r3 (Fz)", "r4 (Mx)", "r5 (My)", "r6 (Mz)"]
    linestyles = ["-", "-", "--", "-.", "-.", ":"]

    x_idx = np.arange(n_stages)
    for k in range(6):
        ax_b.plot(x_idx, risk_matrix[:, k], marker="o", markersize=7,
                  color=colors_risk[k], linestyle=linestyles[k],
                  linewidth=2, label=labels_risk[k], zorder=3)

    # r6 突出标注
    ax_b.annotate("DC对力臂最大\nr6 陡增",
                  xy=(2, risk_matrix[2, 5]), xytext=(2.8, risk_matrix[2, 5] + 0.15),
                  fontsize=9, color="#D62828", fontweight="bold",
                  arrowprops=dict(arrowstyle="->", color="#D62828", lw=1.5))

    ax_b.axhline(y=0.7, color="#999", linestyle="--", linewidth=1, alpha=0.6)
    ax_b.text(4.8, 0.72, "警戒线 r=0.7", fontsize=8, color="#999", ha="right")

    ax_b.set_xticks(x_idx)
    ax_b.set_xticklabels(stage_labels, fontsize=10)
    ax_b.set_ylim(0, 1.05)
    ax_b.set_ylabel("归一化风险分量 r", fontsize=11)
    ax_b.set_xlabel("插装阶段", fontsize=11)
    ax_b.set_title("六维风险向量 r 分量递进曲线", fontsize=12, fontweight="bold")
    ax_b.legend(loc="upper left", fontsize=8, ncol=2)
    ax_b.grid(True, alpha=0.3)

    fig.suptitle("接触集合演化与六维风险分量递进", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_3_risk_evolution.png", dpi=DPI, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig3_3_risk_evolution.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[OK] 图3-3 已保存: risk_evolution")


# ============================================================
# 图 3-4: 安全多面体二维截面
# ============================================================

def fig_3_4_safety_polyhedron(pins: List[PinSpec]):
    """安全多面体在 (Fx, Fy) 和 (Fz, Mx) 截面上的投影"""

    # 选取 S2 阶段（DC 对入孔后）的接触构型
    active = [p for p in pins if p.name in {"PE", "DC_neg", "DC_pos"}]

    # 摩擦系数
    mu = 0.15

    def build_safety_polygon_2d(active_pins: List[PinSpec], dof_pair: Tuple[int, int]):
        """
        构造安全多面体在 (dof_a, dof_b) 二维截面上的投影多边形。
        算法：对各接触点的摩擦锥约束，投影到二维力/力矩子空间，
        取所有半平面的交集。
        """
        # 构建简化抓取矩阵 (对 2D 投影)
        polys = []
        n_pts = 120
        theta = np.linspace(0, 2 * np.pi, n_pts)

        for p in active_pins:
            # 每个接触点贡献的摩擦锥约束半空间
            r = np.array([p.x, p.y, 0.0])
            # 法向力上限归一化
            F_max = 1.0

            # 生成摩擦锥在二维截面的边界
            if dof_pair == (0, 1):  # Fx-Fy 截面
                # 法向力方向（x-y 平面内的径向）
                r_norm = np.linalg.norm(r[:2])
                if r_norm < 1e-6:
                    continue
                n_dir = r[:2] / r_norm
                t_dir = np.array([-n_dir[1], n_dir[0]])

                # 摩擦锥投影：F_n * n_dir + F_t * t_dir, |F_t| <= mu * F_n
                # 在 2D 中就是两个半平面的交集
                for sign in [-1, 1]:
                    a = -sign * mu * n_dir + t_dir
                    b = sign * mu * n_dir - t_dir
                    # 合并为半空间
                    normal = np.array([a[1] - b[1], b[0] - a[0]])
                    normal = normal / (np.linalg.norm(normal) + 1e-9)
                    polys.append((normal, 0.8))

            elif dof_pair == (2, 3):  # Fz-Mx 截面
                # Fz 受所有接触点法向摩擦约束
                # Mx 受 y 方向力臂约束
                polys.append((np.array([1.0, 0.0]), 1.0))   # Fz <= 1
                polys.append((np.array([-1.0, 0.0]), 0.0))  # Fz >= 0 (法向力非负)

                if abs(p.y) > 1e-6:
                    # Mx = Fz * y → |Mx| <= Fz * |y|
                    polys.append((np.array([-abs(p.y), 1.0]), 0.0))
                    polys.append((np.array([-abs(p.y), -1.0]), 0.0))

        return polys

    # ---- 子图 A: (Fx, Fy) 力截面 ----
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 5.2))

    # 构建 (Fx, Fy) 安全多边形
    polys_fxfy = build_safety_polygon_2d(active, (0, 1))

    # 生成多边形区域的采样点
    n_grid = 200
    fx_range = np.linspace(-1.2, 1.2, n_grid)
    fy_range = np.linspace(-1.2, 1.2, n_grid)
    Fx, Fy = np.meshgrid(fx_range, fy_range)
    mask = np.ones_like(Fx, dtype=bool)

    for normal, offset in polys_fxfy:
        mask &= (normal[0] * Fx + normal[1] * Fy <= offset)

    # 可视化
    ax_a.contourf(Fx, Fy, mask.astype(float), levels=[0.5, 1.0],
                  colors=["#A8DADC"], alpha=0.6)

    # 边界线
    # 用极坐标追踪安全边界
    angles = np.linspace(0, 2 * np.pi, 360)
    boundary = []
    for ang in angles:
        r_max = 1.5
        r_min = 0.0
        for _ in range(40):  # 二分查找边界
            r_mid = (r_min + r_max) / 2
            pt = np.array([r_mid * np.cos(ang), r_mid * np.sin(ang)])
            feasible = all(normal[0] * pt[0] + normal[1] * pt[1] <= offset
                          for normal, offset in polys_fxfy)
            if feasible:
                r_min = r_mid
            else:
                r_max = r_mid
        boundary.append([r_min * np.cos(ang), r_min * np.sin(ang)])

    boundary = np.array(boundary)
    ax_a.plot(boundary[:, 0], boundary[:, 1], color="#1D3557", linewidth=2.5, zorder=4)

    # 各接触点的法向方向
    for p in active:
        r = np.array([p.x, p.y])
        r_norm = np.linalg.norm(r)
        if r_norm < 1e-6:
            continue
        n_dir = r / r_norm
        ax_a.arrow(0, 0, n_dir[0] * 0.5, n_dir[1] * 0.5,
                   head_width=0.04, head_length=0.06, fc="#E63946", ec="#E63946", lw=1.5)
        ax_a.text(n_dir[0] * 0.58, n_dir[1] * 0.58, p.name, fontsize=7, color="#E63946")

    # 标注区域
    ax_a.text(0.2, 0.2, "安全\n可行域", fontsize=10, ha="center", va="center",
              color="#1D3557", fontweight="bold")
    ax_a.text(0.85, 0.85, "卡滞\n风险域", fontsize=10, ha="center", va="center",
              color="#D62828", fontstyle="italic")

    ax_a.axhline(y=0, color="#CCC", linewidth=0.5)
    ax_a.axvline(x=0, color="#CCC", linewidth=0.5)
    ax_a.set_xlim(-1.2, 1.2)
    ax_a.set_ylim(-1.2, 1.2)
    ax_a.set_xlabel("Fx / 归一化", fontsize=11)
    ax_a.set_ylabel("Fy / 归一化", fontsize=11)
    ax_a.set_title("(Fx, Fy) 力截面 (S2 阶段: PE+DC对)", fontsize=11, fontweight="bold")
    ax_a.set_aspect("equal")
    ax_a.grid(True, alpha=0.3)

    # ---- 子图 B: (Fz, Mx) 力-力矩截面 ----
    polys_fzmx = build_safety_polygon_2d(active, (2, 3))

    fz_range = np.linspace(0, 1.2, n_grid)
    mx_range = np.linspace(-0.8, 0.8, n_grid)
    Fz, Mx = np.meshgrid(fz_range, mx_range)
    mask2 = np.ones_like(Fz, dtype=bool)
    for normal, offset in polys_fzmx:
        mask2 &= (normal[0] * Fz + normal[1] * Mx <= offset)

    ax_b.contourf(Fz, Mx, mask2.astype(float), levels=[0.5, 1.0],
                  colors=["#E9C46A"], alpha=0.5)

    # (Fz, Mx) 边界 – 手工绘制梯形区域
    y_pe = abs([p for p in active if p.name == "PE"][0].y)  # 21
    y_dc = abs([p for p in active if p.name == "DC_neg"][0].y)  # 1.2

    fz_vals = np.linspace(0, 1.0, 100)
    mx_upper = fz_vals * y_pe * 0.8
    mx_lower = -fz_vals * y_pe * 0.8

    ax_b.fill_between(fz_vals, mx_lower, mx_upper, color="#A8DADC", alpha=0.5)
    ax_b.plot(fz_vals, mx_upper, color="#1D3557", linewidth=2.5)
    ax_b.plot(fz_vals, mx_lower, color="#1D3557", linewidth=2.5)
    ax_b.axvline(x=1.0, ymin=0.2, ymax=0.8, color="#1D3557", linewidth=2.5)

    ax_b.text(0.5, 0.35, "安全\n可行域", fontsize=10, ha="center", va="center",
              color="#1D3557", fontweight="bold")
    ax_b.text(0.85, 0.55, "卡滞", fontsize=9, ha="center", va="center",
              color="#D62828", fontstyle="italic")

    ax_b.set_xlim(0, 1.2)
    ax_b.set_ylim(-0.8, 0.8)
    ax_b.set_xlabel("Fz / 归一化", fontsize=11)
    ax_b.set_ylabel("Mx / 归一化", fontsize=11)
    ax_b.set_title("(Fz, Mx) 力—力矩截面 (S2 阶段: PE+DC对)", fontsize=11, fontweight="bold")
    ax_b.grid(True, alpha=0.3)

    fig.suptitle("无卡滞安全多面体二维截面示意", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_4_safety_polyhedron.png", dpi=DPI, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig3_4_safety_polyhedron.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[OK] 图3-4 已保存: safety_polyhedron")


# ============================================================
# 补充: 参数计算表输出
# ============================================================

def export_param_table(pins: List[PinSpec]):
    """导出表3-? 的完整参数计算到 CSV"""
    rows = []
    for p in pins:
        rows.append({
            "触头名": p.name,
            "针型": p.pin_type,
            "针径 Φ/mm": p.diameter,
            "有效长度 L/mm": p.length_eff,
            "径向间隙 c/mm": p.clearance,
            "2c/L": f"{2 * p.clearance / p.length_eff:.6f}",
            "θ_crit/°": f"{p.theta_crit_deg:.4f}",
            "μ_crit": f"{p.mu_crit:.6f}",
            "耦合顺序": p.z_order,
            "入孔阶段": f"S{p.stage}",
        })

    csv_path = FIG_DIR / "table_critical_params.csv"
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"[OK] 参数表已导出: {csv_path}")

    # 打印关键结论
    print("\n===== 结论 3.1 验证 =====")
    for ptype in ["大针", "中针", "小针"]:
        group = [p for p in pins if p.pin_type == ptype]
        avg_mu = float(np.mean([p.mu_crit for p in group]))
        print(f"  {ptype}: μ_crit = {avg_mu:.6f},  实测 μ ≈ 0.15,  比值 = {0.15/avg_mu:.0f}x")


# ============================================================
# 主入口
# ============================================================

def main():
    print("=" * 60)
    print("第3章 数据处理与图表生成")
    print("=" * 60)

    # 加载数据
    pins = load_pin_specs(MODEL_CSV)
    print(f"\n已加载 {len(pins)} 根插针:")
    for p in pins:
        print(f"  {p.name:<8s}  {p.pin_type}  Φ={p.diameter}mm  L={p.length_eff}mm  "
              f"c={p.clearance}mm  θ_crit={p.theta_crit_deg:.4f}°  μ_crit={p.mu_crit:.6f}  "
              f"z={p.z_order}  S{p.stage}")

    # 输出参数表
    export_param_table(pins)

    # 生成图表
    print(f"\n生成图表 (输出: {FIG_DIR})...\n")
    fig_3_1_pin_layout(pins)
    fig_3_2_critical_params(pins)
    fig_3_3_risk_evolution(pins)
    fig_3_4_safety_polyhedron(pins)

    print(f"\n全部完成! 共输出 {len(list(FIG_DIR.glob('*')))} 个文件到 {FIG_DIR}")


if __name__ == "__main__":
    main()
