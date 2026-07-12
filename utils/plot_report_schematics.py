from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent

INK = "#222222"
MUTED = "#666666"
BLUE = "#0B4F8A"
BLUE_LIGHT = "#E8F1FA"
BLUE_MID = "#D6E6F5"
GREEN = "#2E8B57"
GREEN_LIGHT = "#E2E8DD"
ORANGE = "#D55E00"
ORANGE_LIGHT = "#FAD7CD"
YELLOW_LIGHT = "#FFF2CC"
YELLOW = "#FFE599"
RED = "#C62828"
RED_LIGHT = "#F6A391"
PURPLE_LIGHT = "#E9E3F5"
GREY_LIGHT = "#F3F3F3"


def find_thesis_dir() -> Path:
    matches = sorted(
        path
        for path in WORKSPACE_ROOT.glob("DATN_20252_full_rewrite*")
        if path.is_dir() and (path / "main.tex").is_file()
    )
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one thesis directory, found: {matches}")
    return matches[0]


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Serif",
            "font.size": 9.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return [png_path, pdf_path]


def rounded_box(
    axis: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    facecolor: str,
    edgecolor: str = INK,
    fontsize: float = 8.8,
    weight: str = "normal",
    color: str = INK,
    linewidth: float = 0.95,
    radius: float = 0.06,
    linespacing: float = 1.12,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.035,rounding_size={radius}",
        linewidth=linewidth,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    axis.add_patch(patch)
    axis.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color=color,
        linespacing=linespacing,
    )
    return patch


def plain_box(
    axis: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    facecolor: str,
    edgecolor: str = INK,
    fontsize: float = 8.8,
    weight: str = "normal",
    color: str = INK,
    linewidth: float = 0.95,
) -> Rectangle:
    patch = Rectangle((x, y), width, height, linewidth=linewidth, edgecolor=edgecolor, facecolor=facecolor)
    axis.add_patch(patch)
    axis.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color=color,
        linespacing=1.12,
    )
    return patch


def arrow(
    axis: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    text: str | None = None,
    text_xy: tuple[float, float] | None = None,
    color: str = INK,
    linewidth: float = 1.1,
    connectionstyle: str = "arc3,rad=0.0",
    linestyle: str = "-",
) -> None:
    axis.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(
            arrowstyle="-|>",
            linewidth=linewidth,
            color=color,
            linestyle=linestyle,
            shrinkA=2,
            shrinkB=2,
            connectionstyle=connectionstyle,
        ),
    )
    if text and text_xy:
        axis.text(
            text_xy[0],
            text_xy[1],
            text,
            ha="center",
            va="center",
            fontsize=7.7,
            color=color,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.6),
            linespacing=1.05,
        )


def draw_prb_bar(axis: plt.Axes, x: float, y: float, width: float, height: float, emergency_ratio: float) -> None:
    axis.add_patch(Rectangle((x, y), width, height, linewidth=0.8, edgecolor=INK, facecolor="white"))
    emergency_width = width * emergency_ratio
    axis.add_patch(Rectangle((x, y), emergency_width, height, linewidth=0, facecolor=RED_LIGHT))
    axis.add_patch(Rectangle((x + emergency_width, y), width - emergency_width, height, linewidth=0, facecolor=GREEN_LIGHT))
    axis.text(x + emergency_width / 2, y + height / 2, "E", ha="center", va="center", fontsize=6.8)
    axis.text(x + emergency_width + (width - emergency_width) / 2, y + height / 2, "O", ha="center", va="center", fontsize=6.8)


def plot_problem_motivation(output_dir: Path) -> list[Path]:
    fig, axis = plt.subplots(figsize=(7.4, 3.15))
    axis.set_xlim(0, 12)
    axis.set_ylim(0, 5.0)
    axis.axis("off")

    panels = [
        (0.20, "(a) Bình thường", "Lưu lượng khẩn cấp thấp", 0.30, "Hai lát cắt hoạt động ổn định", GREEN),
        (4.15, "(b) Cấp phát tĩnh khi khẩn cấp", "Xe cứu thương xuất hiện", 0.30, "Nguy cơ tăng trễ và vi phạm SLA", RED),
        (8.10, "(c) DQN phân bổ động", "Quan sát KPI và trạng thái", 0.70, "Tăng PRB cho Emergency Slice", BLUE),
    ]
    for x, header, subtitle, ratio, outcome, color in panels:
        rounded_box(axis, x, 0.25, 3.55, 4.45, "", "white", edgecolor="#BBBBBB", linewidth=0.9, radius=0.04)
        axis.text(x + 1.775, 4.35, header, ha="center", va="center", fontsize=8.8, fontweight="bold")
        rounded_box(axis, x + 0.35, 3.15, 2.85, 0.62, subtitle, BLUE_LIGHT if color == BLUE else GREY_LIGHT, fontsize=7.8)
        if x < 8:
            rounded_box(axis, x + 0.35, 2.15, 1.22, 0.58, "Emergency\ntraffic", ORANGE_LIGHT, fontsize=7.2)
            rounded_box(axis, x + 1.98, 2.15, 1.22, 0.58, "Ordinary\ntraffic", GREEN_LIGHT, fontsize=7.2)
        else:
            rounded_box(axis, x + 0.35, 2.15, 1.22, 0.58, "DQN\nxApp", BLUE_MID, fontsize=7.2, weight="bold")
            rounded_box(axis, x + 1.98, 2.15, 1.22, 0.58, "RAN\nKPI", BLUE_LIGHT, fontsize=7.2)
            arrow(axis, (1.98 + x, 2.44), (1.57 + x, 2.44), color=BLUE, linewidth=1.0)
        draw_prb_bar(axis, x + 0.47, 1.32, 2.62, 0.40, ratio)
        rounded_box(axis, x + 0.38, 0.55, 2.80, 0.50, outcome, "white", edgecolor=color, fontsize=7.3, color=color, linewidth=1.05)

    axis.plot([3.92, 3.92], [0.55, 4.35], color="#CCCCCC", linewidth=0.8)
    axis.plot([7.87, 7.87], [0.55, 4.35], color="#CCCCCC", linewidth=0.8)
    fig.tight_layout(pad=0.1)
    return save_figure(fig, output_dir, "fig_problem_motivation")


def plot_oran_dqn_xapp_overview(output_dir: Path) -> list[Path]:
    fig, axis = plt.subplots(figsize=(7.4, 4.05))
    axis.set_xlim(0, 12)
    axis.set_ylim(0, 7.2)
    axis.axis("off")

    rounded_box(axis, 0.45, 5.72, 8.65, 0.96, "", BLUE_LIGHT, linewidth=1.0)
    axis.text(4.78, 6.47, "Service Management and Orchestration / SMO", ha="center", va="center", fontsize=8.9, fontweight="bold")
    rounded_box(axis, 0.85, 5.88, 3.30, 0.40, "Non-RT RIC", "white", fontsize=8.2)
    rounded_box(axis, 4.85, 5.88, 3.80, 0.40, "Chính sách dài hạn, huấn luyện mô hình", "white", fontsize=7.7)

    rounded_box(axis, 0.45, 3.62, 8.65, 1.24, "", BLUE_LIGHT, linewidth=1.0)
    axis.text(4.78, 4.56, "Near-RT RIC", ha="center", va="center", fontsize=9.4, fontweight="bold")
    rounded_box(axis, 1.00, 3.86, 3.55, 0.52, "DQN-based RAN Slicing xApp", BLUE_MID, edgecolor=BLUE, fontsize=8.5, weight="bold", color=BLUE)
    rounded_box(axis, 4.95, 3.86, 1.70, 0.52, "KPM\ncollection", "white", fontsize=7.3)
    rounded_box(axis, 6.95, 3.86, 1.70, 0.52, "Policy\ncontrol", "white", fontsize=7.3)

    rounded_box(axis, 0.45, 0.75, 8.65, 1.85, "", YELLOW_LIGHT, linewidth=1.0)
    axis.text(4.78, 2.30, "RAN nodes và phương tiện kết nối", ha="center", va="center", fontsize=9.0, fontweight="bold")
    plain_box(axis, 0.90, 1.35, 1.20, 0.55, "O-CU", "white", fontsize=8.2)
    plain_box(axis, 2.62, 1.35, 1.20, 0.55, "O-DU", "white", fontsize=8.2)
    plain_box(axis, 4.34, 1.35, 1.20, 0.55, "O-RU", "white", fontsize=8.2)
    rounded_box(axis, 6.15, 1.12, 2.35, 0.42, "Emergency vehicles", ORANGE_LIGHT, fontsize=7.2)
    rounded_box(axis, 6.15, 1.72, 2.35, 0.42, "Ordinary vehicles", GREEN_LIGHT, fontsize=7.2)

    arrow(axis, (4.78, 5.72), (4.78, 4.86), "A1", (5.12, 5.28), color=BLUE)
    arrow(axis, (2.78, 3.62), (2.78, 2.60), "E2 control:\nPRB ratio", (2.05, 3.08), color=BLUE)
    arrow(axis, (5.72, 2.60), (5.72, 3.62), "KPM feedback", (6.62, 3.08), color=GREEN)
    arrow(axis, (2.10, 1.63), (2.62, 1.63), "F1", (2.36, 1.90), color=INK)
    arrow(axis, (3.82, 1.63), (4.34, 1.63), "Fronthaul", (4.08, 1.90), color=INK)
    arrow(axis, (5.54, 1.63), (6.15, 1.63), "Radio\naccess", (5.86, 1.18), color=INK)

    plain_box(axis, 9.55, 0.75, 1.95, 5.93, "", "white", edgecolor="#777777", linewidth=0.9)
    axis.plot([9.55, 11.50], [4.95, 4.95], color="#777777", linewidth=0.8)
    axis.plot([9.55, 11.50], [3.15, 3.15], color="#777777", linewidth=0.8)
    axis.text(10.52, 6.05, "> 1 s\nNon-RT control", ha="center", va="center", fontsize=7.8)
    axis.text(10.52, 4.05, "10 ms--1 s\nNear-RT control", ha="center", va="center", fontsize=7.8, color=BLUE)
    axis.text(10.52, 1.95, "< 10 ms\nRAN scheduling", ha="center", va="center", fontsize=7.8)

    fig.tight_layout(pad=0.1)
    return save_figure(fig, output_dir, "fig_oran_dqn_xapp_overview")


def plot_system_model_closed_loop(output_dir: Path) -> list[Path]:
    fig, axis = plt.subplots(figsize=(7.4, 3.9))
    axis.set_xlim(0, 12)
    axis.set_ylim(0, 6.3)
    axis.axis("off")

    rounded_box(axis, 3.55, 4.90, 4.90, 0.82, "Near-RT RIC / DQN xApp\nchọn tỷ lệ PRB cho Emergency Slice", BLUE_LIGHT, edgecolor=BLUE, fontsize=8.5, weight="bold", color=BLUE)
    rounded_box(axis, 0.55, 2.30, 2.05, 1.20, "Traffic input\nEmergency\nOrdinary", GREY_LIGHT, fontsize=8.0)
    rounded_box(axis, 3.30, 2.20, 2.45, 1.35, "gNB / RAN Env\nhàng đợi, CQI,\ndung lượng, SLA", YELLOW_LIGHT, fontsize=8.0, weight="bold")
    rounded_box(axis, 6.55, 2.35, 1.65, 1.05, "PRB pool\nchia 2 slice", YELLOW, fontsize=8.0, weight="bold")
    rounded_box(axis, 9.05, 3.10, 2.25, 0.70, "Emergency Slice\nưu tiên độ trễ", RED_LIGHT, fontsize=8.0, weight="bold")
    rounded_box(axis, 9.05, 1.70, 2.25, 0.70, "Ordinary Traffic Slice\nthông lượng, hàng đợi", GREEN_LIGHT, fontsize=7.6, weight="bold")
    rounded_box(axis, 0.85, 0.65, 10.10, 0.50, "Phản hồi: trạng thái $s_t$, KPI và phần thưởng $r_t$", BLUE_LIGHT, edgecolor=GREEN, fontsize=8.0, color=GREEN)

    arrow(axis, (2.60, 2.90), (3.30, 2.90))
    arrow(axis, (5.75, 2.88), (6.55, 2.88))
    arrow(axis, (8.20, 3.15), (9.05, 3.45), "PRB", (8.58, 3.50))
    arrow(axis, (8.20, 2.60), (9.05, 2.05), "PRB", (8.58, 2.20))
    arrow(axis, (6.00, 4.90), (6.95, 3.40), "action $a_t$", (7.36, 4.12), color=BLUE, linewidth=1.25)
    arrow(axis, (4.35, 2.20), (4.35, 1.15), color=GREEN, linewidth=1.15)
    arrow(axis, (5.95, 1.15), (5.95, 4.90), color=GREEN, linewidth=1.15, connectionstyle="arc3,rad=0.0")
    arrow(axis, (1.45, 1.15), (3.55, 4.90), color=GREEN, linewidth=1.0, connectionstyle="arc3,rad=-0.23")

    fig.tight_layout(pad=0.1)
    return save_figure(fig, output_dir, "fig_system_model_closed_loop")


def plot_mdp_decision_cycle(output_dir: Path) -> list[Path]:
    fig, axis = plt.subplots(figsize=(7.1, 4.25))
    axis.set_xlim(0, 10)
    axis.set_ylim(0, 6.7)
    axis.axis("off")

    rounded_box(axis, 0.55, 4.55, 2.25, 0.95, "Trạng thái $s_t$\ntải, hàng đợi, trễ,\nhiệu suất phổ, cờ khẩn cấp", BLUE_LIGHT, fontsize=7.45)
    rounded_box(axis, 3.85, 4.55, 2.25, 0.95, "Tác nhân DQN\nước lượng $Q(s,a)$", BLUE_MID, edgecolor=BLUE, fontsize=8.0, weight="bold", color=BLUE)
    rounded_box(axis, 7.05, 4.55, 2.25, 0.95, "Hành động $a_t$\ntỷ lệ PRB cho\nEmergency Slice", YELLOW, fontsize=7.8)
    rounded_box(axis, 5.45, 1.95, 2.40, 1.02, "Môi trường RAN\nlưu lượng, kênh,\nhàng đợi, SLA", YELLOW_LIGHT, fontsize=7.8, weight="bold")
    rounded_box(axis, 1.85, 1.95, 2.45, 1.02, "Reward $r_t$ và\ntrạng thái $s_{t+1}$", GREEN_LIGHT, fontsize=8.0)

    arrow(axis, (2.80, 5.03), (3.85, 5.03), color=BLUE)
    arrow(axis, (6.10, 5.03), (7.05, 5.03), color=BLUE)
    arrow(axis, (8.18, 4.55), (6.90, 2.97), "cấp PRB", (7.85, 3.65), color=INK)
    arrow(axis, (5.45, 2.45), (4.30, 2.45), color=GREEN)
    arrow(axis, (3.08, 2.97), (1.67, 4.55), "vòng lặp\nquyết định", (1.10, 3.60), color=GREEN, connectionstyle="arc3,rad=0.18")

    fig.tight_layout(pad=0.1)
    return save_figure(fig, output_dir, "fig_mdp_decision_cycle")


def plot_dqn_training_pipeline(output_dir: Path) -> list[Path]:
    fig, axis = plt.subplots(figsize=(7.4, 4.15))
    axis.set_xlim(0, 12)
    axis.set_ylim(0, 6.6)
    axis.axis("off")

    rounded_box(axis, 0.65, 4.25, 2.00, 0.78, "RAN Env\nsinh $s_t$", YELLOW_LIGHT, fontsize=8.0, weight="bold")
    rounded_box(axis, 3.35, 4.25, 2.15, 0.78, "Q-network\nchọn hành động", BLUE_MID, edgecolor=BLUE, fontsize=8.0, weight="bold", color=BLUE)
    rounded_box(axis, 6.15, 4.25, 2.00, 0.78, "Action $a_t$\ntỷ lệ PRB", YELLOW, fontsize=8.0)
    rounded_box(axis, 8.95, 4.25, 2.05, 0.78, "Env step\n$r_t, s_{t+1}$", YELLOW_LIGHT, fontsize=8.0, weight="bold")

    rounded_box(axis, 8.15, 2.15, 2.55, 0.86, "Replay Buffer\n$(s_t,a_t,r_t,s_{t+1})$", PURPLE_LIGHT, fontsize=8.0)
    rounded_box(axis, 4.75, 2.15, 2.30, 0.86, "Mini-batch\nlấy mẫu ngẫu nhiên", GREY_LIGHT, fontsize=8.0)
    rounded_box(axis, 1.05, 2.15, 2.55, 0.86, "Cập nhật Q-network\nBellman target", BLUE_LIGHT, edgecolor=BLUE, fontsize=8.0, color=BLUE)
    rounded_box(axis, 1.05, 0.75, 2.55, 0.76, "Target Network\ncập nhật định kỳ", BLUE_LIGHT, fontsize=8.0)

    arrow(axis, (2.65, 4.64), (3.35, 4.64), "$s_t$", (3.00, 4.90), color=INK)
    arrow(axis, (5.50, 4.64), (6.15, 4.64), "epsilon-greedy", (5.84, 5.18), color=BLUE)
    arrow(axis, (8.15, 4.64), (8.95, 4.64), color=INK)
    arrow(axis, (9.98, 4.25), (9.42, 3.01), "transition", (10.42, 3.52), color=GREEN)
    arrow(axis, (8.15, 2.58), (7.05, 2.58), color=INK)
    arrow(axis, (4.75, 2.58), (3.60, 2.58), color=INK)
    arrow(axis, (2.32, 2.15), (2.32, 1.51), color=BLUE)
    arrow(axis, (3.60, 1.13), (4.52, 4.25), color=BLUE, connectionstyle="arc3,rad=-0.16")
    arrow(axis, (1.65, 2.15), (0.95, 4.25), color=GREEN, connectionstyle="arc3,rad=0.18")

    fig.tight_layout(pad=0.1)
    return save_figure(fig, output_dir, "fig_dqn_training_pipeline")


def plot_oran_dqn_xapp_overview(output_dir: Path) -> list[Path]:
    """Cleaner O-RAN overview: layered boxes and no labels on crowded arrows."""
    fig, axis = plt.subplots(figsize=(7.4, 4.0))
    axis.set_xlim(0, 12)
    axis.set_ylim(0, 7.0)
    axis.axis("off")

    rounded_box(axis, 0.45, 5.40, 8.70, 1.20, "", BLUE_LIGHT, linewidth=1.0)
    axis.text(4.80, 6.35, "Service Management and Orchestration / SMO", ha="center", va="center", fontsize=8.8, fontweight="bold")
    rounded_box(axis, 0.90, 5.62, 3.00, 0.42, "Non-RT RIC", "white", fontsize=8.0)
    rounded_box(axis, 4.70, 5.62, 3.95, 0.42, "Chính sách dài hạn và huấn luyện mô hình", "white", fontsize=7.5)

    rounded_box(axis, 0.45, 3.35, 8.70, 1.38, "", BLUE_LIGHT, linewidth=1.0)
    axis.text(4.80, 4.35, "Near-RT RIC (vị trí logic)", ha="center", va="center", fontsize=8.8, fontweight="bold")
    rounded_box(axis, 0.95, 3.60, 3.75, 0.55, "Bộ điều khiển DQN mô phỏng\n(định vị logic xApp)", BLUE_MID, edgecolor=BLUE, fontsize=7.7, weight="bold", color=BLUE)
    rounded_box(axis, 5.00, 3.60, 1.60, 0.55, "KPM tham chiếu\n(không thu thập thực)", "white", fontsize=6.35)
    rounded_box(axis, 7.00, 3.60, 1.60, 0.55, "Chính sách tham chiếu\n(không gửi E2 thực)", "white", fontsize=6.2)

    rounded_box(axis, 0.45, 0.70, 8.70, 1.70, "", YELLOW_LIGHT, linewidth=1.0)
    axis.text(4.80, 2.12, "Các thành phần RAN tham chiếu", ha="center", va="center", fontsize=8.7, fontweight="bold")
    plain_box(axis, 0.85, 1.20, 1.12, 0.52, "O-CU", "white", fontsize=7.8)
    plain_box(axis, 2.55, 1.20, 1.12, 0.52, "O-DU", "white", fontsize=7.8)
    plain_box(axis, 4.25, 1.20, 1.12, 0.52, "O-RU", "white", fontsize=7.8)
    rounded_box(axis, 6.05, 1.48, 2.35, 0.40, "Ordinary vehicles", GREEN_LIGHT, fontsize=7.0)
    rounded_box(axis, 6.05, 0.93, 2.35, 0.40, "Emergency vehicles", ORANGE_LIGHT, fontsize=7.0)

    arrow(axis, (4.80, 5.40), (4.80, 4.73), "A1", (5.08, 5.06), color=BLUE)
    arrow(axis, (2.80, 3.35), (2.80, 2.40), "Luồng E2\ntham chiếu", (2.25, 2.90), color=BLUE)
    arrow(axis, (5.70, 2.40), (5.70, 3.35), "KPM\ntham chiếu", (6.30, 2.90), color=GREEN)
    arrow(axis, (1.97, 1.46), (2.55, 1.46), color=INK)
    arrow(axis, (3.67, 1.46), (4.25, 1.46), color=INK)
    arrow(axis, (5.37, 1.46), (6.05, 1.46), color=INK)
    axis.text(2.26, 1.77, "F1", ha="center", va="center", fontsize=6.6, color=MUTED)
    axis.text(3.96, 1.77, "Fronthaul", ha="center", va="center", fontsize=6.6, color=MUTED)
    axis.text(5.72, 1.76, "Radio access", ha="center", va="center", fontsize=6.6, color=MUTED)

    plain_box(axis, 9.55, 0.70, 1.95, 5.90, "", "white", edgecolor="#777777", linewidth=0.9)
    axis.plot([9.55, 11.50], [4.85, 4.85], color="#777777", linewidth=0.8)
    axis.plot([9.55, 11.50], [3.10, 3.10], color="#777777", linewidth=0.8)
    axis.text(10.52, 5.78, "> 1 s\nNon-RT control", ha="center", va="center", fontsize=7.5)
    axis.text(10.52, 3.98, "10 ms--1 s\nNear-RT control", ha="center", va="center", fontsize=7.5, color=BLUE)
    axis.text(10.52, 1.90, "< 10 ms\nRAN scheduling", ha="center", va="center", fontsize=7.5)
    axis.text(4.80, 0.26, "Đồ án chỉ mô phỏng chính sách phân bổ PRB; không triển khai E2, KPM hay xApp thực.", ha="center", va="center", fontsize=6.5, color=MUTED)

    fig.tight_layout(pad=0.1)
    return save_figure(fig, output_dir, "fig_oran_dqn_xapp_overview")


def plot_system_model_closed_loop(output_dir: Path) -> list[Path]:
    """Cleaner closed loop: feedback travels through an explicit bottom box."""
    fig, axis = plt.subplots(figsize=(7.4, 3.75))
    axis.set_xlim(0, 12)
    axis.set_ylim(0, 6.0)
    axis.axis("off")

    rounded_box(axis, 3.55, 4.68, 4.90, 0.92, "Bộ điều khiển DQN mô phỏng\n(định vị logic tại Near-RT RIC)\nchọn tỷ lệ PRB cho lát cắt khẩn cấp", BLUE_LIGHT, edgecolor=BLUE, fontsize=7.55, weight="bold", color=BLUE)
    rounded_box(axis, 0.55, 2.45, 2.05, 1.05, "Lưu lượng vào\nkhẩn cấp\nvà thông thường", GREY_LIGHT, fontsize=7.7)
    rounded_box(axis, 3.25, 2.38, 2.35, 1.18, "Môi trường RAN mô phỏng\nhàng đợi, hiệu suất phổ,\ndung lượng, SLA", YELLOW_LIGHT, fontsize=7.25, weight="bold")
    rounded_box(axis, 6.45, 2.52, 1.65, 0.92, "Tập PRB\nchia hai lát cắt", YELLOW, fontsize=7.7, weight="bold")
    rounded_box(axis, 9.05, 3.08, 2.25, 0.66, "Lát cắt khẩn cấp\n(Emergency Slice)", RED_LIGHT, fontsize=7.45, weight="bold")
    rounded_box(axis, 9.05, 1.82, 2.25, 0.66, "Lát cắt xe thông thường\n(Ordinary Traffic Slice)", GREEN_LIGHT, fontsize=6.8, weight="bold")
    rounded_box(axis, 3.15, 0.72, 5.70, 0.56, "Phản hồi: trạng thái $s_t$, KPI và phần thưởng $r_t$", BLUE_LIGHT, edgecolor=GREEN, fontsize=7.8, color=GREEN)

    arrow(axis, (2.60, 2.98), (3.25, 2.98))
    arrow(axis, (5.60, 2.98), (6.45, 2.98))
    arrow(axis, (8.10, 3.18), (9.05, 3.41), "PRB", (8.55, 3.55))
    arrow(axis, (8.10, 2.77), (9.05, 2.15), "PRB", (8.55, 2.38))
    arrow(axis, (7.27, 4.68), (7.27, 3.44), "hành động $a_t$", (7.92, 4.04), color=BLUE, linewidth=1.2)
    arrow(axis, (4.43, 2.38), (4.43, 1.28), color=GREEN, linewidth=1.1)
    axis.plot([3.15, 0.35, 0.35], [1.00, 1.00, 4.98], color=GREEN, linewidth=1.1)
    arrow(axis, (0.35, 4.98), (3.55, 4.98), color=GREEN, linewidth=1.1)

    fig.tight_layout(pad=0.1)
    return save_figure(fig, output_dir, "fig_system_model_closed_loop")


def plot_dqn_training_pipeline(output_dir: Path) -> list[Path]:
    """Cleaner DQN training pipeline with a two-row, non-crossing flow."""
    fig, axis = plt.subplots(figsize=(7.4, 3.85))
    axis.set_xlim(0, 12)
    axis.set_ylim(0, 6.1)
    axis.axis("off")

    rounded_box(axis, 0.65, 4.15, 2.00, 0.76, "RAN Env\nsinh $s_t$", YELLOW_LIGHT, fontsize=7.8, weight="bold")
    rounded_box(axis, 3.35, 4.15, 2.15, 0.76, "Q-network\nchọn hành động", BLUE_MID, edgecolor=BLUE, fontsize=7.8, weight="bold", color=BLUE)
    rounded_box(axis, 6.15, 4.15, 2.00, 0.76, "Action $a_t$\ntỷ lệ PRB", YELLOW, fontsize=7.8)
    rounded_box(axis, 8.95, 4.15, 2.05, 0.76, "Env step\n$r_t, s_{t+1}, d_t$", YELLOW_LIGHT, fontsize=7.8, weight="bold")

    rounded_box(axis, 8.70, 2.18, 2.45, 0.80, "Replay Buffer\n$(s_t,a_t,r_t,s_{t+1},d_t)$", PURPLE_LIGHT, fontsize=7.15)
    rounded_box(axis, 5.95, 2.18, 2.20, 0.80, "Mini-batch\nlấy mẫu", GREY_LIGHT, fontsize=7.8)
    rounded_box(axis, 3.15, 2.18, 2.25, 0.80, "Cập nhật Q-network\nBellman target", BLUE_LIGHT, edgecolor=BLUE, fontsize=7.8, color=BLUE)
    rounded_box(axis, 3.15, 0.86, 2.25, 0.66, "Target Network\ncập nhật định kỳ", BLUE_LIGHT, fontsize=7.7)

    arrow(axis, (2.65, 4.53), (3.35, 4.53), "$s_t$", (3.00, 4.80), color=INK)
    arrow(axis, (5.50, 4.53), (6.15, 4.53), "epsilon-greedy", (5.84, 5.06), color=BLUE)
    arrow(axis, (8.15, 4.53), (8.95, 4.53), color=INK)
    arrow(axis, (9.98, 4.15), (9.98, 2.98), "transition\n+ cờ kết thúc", (10.66, 3.55), color=GREEN)
    arrow(axis, (8.70, 2.58), (8.15, 2.58), color=INK)
    arrow(axis, (5.95, 2.58), (5.40, 2.58), color=INK)
    arrow(axis, (4.28, 2.98), (4.28, 4.15), "update", (4.78, 3.55), color=BLUE)
    arrow(axis, (4.28, 1.52), (4.28, 2.18), color=BLUE, linestyle="--")

    fig.tight_layout(pad=0.1)
    return save_figure(fig, output_dir, "fig_dqn_training_pipeline")


def main() -> None:
    configure_style()
    output_dir = find_thesis_dir() / "Images"
    output_files: list[Path] = []
    output_files.extend(plot_problem_motivation(output_dir))
    output_files.extend(plot_oran_dqn_xapp_overview(output_dir))
    output_files.extend(plot_system_model_closed_loop(output_dir))
    output_files.extend(plot_mdp_decision_cycle(output_dir))
    output_files.extend(plot_dqn_training_pipeline(output_dir))
    print("Saved schematic figures:")
    for path in output_files:
        print(path)


if __name__ == "__main__":
    main()
