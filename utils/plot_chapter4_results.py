from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
RESULTS_ROOT = PROJECT_ROOT / "results"
SEEDS = (42, 43, 44)

COLOR_DQN = "#0B4F8A"
COLOR_STATIC = "#7F7F7F"
COLOR_GLB = "#2E8B57"
COLOR_SLA = "#C62828"
COLOR_GRID = "#D9D9D9"
SEED_COLORS = {
    42: "#0B4F8A",
    43: "#D55E00",
    44: "#6A3D9A",
}


def require_columns(frame: pd.DataFrame, columns: set[str], source: Path) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise KeyError(f"Thiếu cột trong {source}: {missing}")


def require_finite(frame: pd.DataFrame, columns: list[str], source: Path) -> None:
    values = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy()
    if not np.isfinite(values).all():
        raise ValueError(f"Có giá trị thiếu hoặc không hữu hạn trong {source}: {columns}")


def find_unique_file(filename: str) -> Path:
    matches = sorted(RESULTS_ROOT.rglob(filename))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Cần đúng một file {filename}, tìm thấy {len(matches)}: {matches}"
        )
    return matches[0]


def find_thesis_dir() -> Path:
    matches = sorted(
        path
        for path in WORKSPACE_ROOT.glob("DATN_20252_full_rewrite*")
        if path.is_dir() and (path / "Chuong4.tex").is_file()
    )
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Cần đúng một thư mục luận văn chứa Chuong4.tex, tìm thấy: {matches}"
        )
    return matches[0]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Serif",
            "font.size": 9.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_axis(axis: plt.Axes) -> None:
    axis.grid(axis="y", color=COLOR_GRID, linewidth=0.65, alpha=0.75)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    return [png_path, pdf_path]


def plot_system_model_overview_legacy(thesis_dir: Path) -> list[Path]:
    output_dir = thesis_dir / "Images"
    fig, axis = plt.subplots(figsize=(8.0, 4.6))
    axis.set_xlim(0, 10)
    axis.set_ylim(0, 7)
    axis.axis("off")

    def box(
        xy: tuple[float, float],
        width: float,
        height: float,
        text: str,
        facecolor: str,
        edgecolor: str = "#1A1A1A",
        fontsize: float = 10.0,
        weight: str = "normal",
    ) -> None:
        patch = matplotlib.patches.FancyBboxPatch(
            xy,
            width,
            height,
            boxstyle="round,pad=0.03,rounding_size=0.08",
            linewidth=1.0,
            edgecolor=edgecolor,
            facecolor=facecolor,
        )
        axis.add_patch(patch)
        axis.text(
            xy[0] + width / 2,
            xy[1] + height / 2,
            text,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight=weight,
            linespacing=1.15,
        )

    def arrow(
        start: tuple[float, float],
        end: tuple[float, float],
        text: str | None = None,
        text_xy: tuple[float, float] | None = None,
    ) -> None:
        axis.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops=dict(arrowstyle="-|>", linewidth=1.2, color="#1A1A1A"),
        )
        if text is not None and text_xy is not None:
            axis.text(
                text_xy[0],
                text_xy[1],
                text,
                ha="center",
                va="center",
                fontsize=9.0,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.0),
            )

    box((3.0, 5.35), 4.0, 1.05, "", "#E8F1FA", fontsize=12.0, weight="bold")
    axis.text(5.0, 6.16, "Near-RT RIC", ha="center", va="center", fontsize=12.0, fontweight="bold")
    box(
        (3.35, 5.62),
        3.3,
        0.45,
        "xApp phân mảnh RAN dựa trên DQN",
        "#D6E6F5",
        fontsize=8.8,
    )
    box((1.25, 2.25), 7.5, 2.0, "", "#FFF2CC", fontsize=11.0, weight="bold")
    axis.text(5.0, 4.03, "Môi trường gNB/RAN", ha="center", va="center", fontsize=11.0, fontweight="bold")
    box((4.35, 3.35), 1.3, 0.42, "Kho PRB", "#FFE599", fontsize=9.0, weight="bold")
    box((2.0, 2.58), 2.35, 0.55, "Emergency Slice", "#F6A391", fontsize=10.0, weight="bold")
    box((5.65, 2.58), 2.35, 0.55, "Ordinary Traffic Slice", "#D9E1D4", fontsize=10.0, weight="bold")
    box((2.0, 0.55), 2.35, 0.8, "Xe cứu thương\nlưu lượng khẩn cấp", "#FAD7CD", fontsize=8.8)
    box((5.65, 0.55), 2.35, 0.8, "Xe thông thường\nlưu lượng nền", "#E2E8DD", fontsize=8.8)
    box(
        (0.25, 4.05),
        2.15,
        0.86,
        "Phản hồi trạng thái/KPI:\ntải, hàng đợi, trễ,\nthông lượng, sử dụng PRB",
        "#E8F1FA",
        fontsize=7.8,
    )

    arrow((5.0, 5.35), (5.0, 4.25), "Quyết định\nphân bổ PRB", (4.25, 4.78))
    arrow((5.65, 4.25), (5.65, 5.35), "Trạng thái và\nphần thưởng", (6.75, 4.78))
    arrow((2.40, 4.48), (3.0, 5.35))
    arrow((5.0, 3.35), (3.25, 3.13), "PRB cho\nEmergency Slice", (3.0, 3.55))
    arrow((5.0, 3.35), (6.8, 3.13), "PRB cho\nOrdinary Slice", (7.0, 3.55))
    arrow((3.15, 1.35), (3.15, 2.58), "Lưu lượng\nkhẩn cấp", (2.35, 1.95))
    arrow((6.85, 1.35), (6.85, 2.58), "Lưu lượng\nthông thường", (7.7, 1.95))

    fig.tight_layout(pad=0.1)
    return save_figure(fig, output_dir, "system_model_overview_vi")


def load_and_validate_sources() -> dict[str, object]:
    training_seed_path = find_unique_file("training_seed_results.csv")
    multiseed_dir = training_seed_path.parent
    learning_path = multiseed_dir / "learning_curve_three_training_seeds.csv"
    action_path = multiseed_dir / "action_distribution_by_training_seed.csv"
    for path in (learning_path, action_path):
        if not path.is_file():
            raise FileNotFoundError(f"Không tìm thấy file bắt buộc: {path}")

    seed_results = pd.read_csv(training_seed_path)
    seed_columns = {
        "training_seed",
        "run_name",
        "evaluation_episodes",
        "ambulance_latency_ms_mean",
        "ambulance_p95_ms_mean",
        "ambulance_sla",
        "ordinary_demand_satisfaction",
        "ordinary_throughput_emergency_mbps",
        "ordinary_queue_p95_mbit",
        "ordinary_overflow_mbit",
        "ordinary_offered_mbit",
        "ordinary_overflow_fraction",
    }
    require_columns(seed_results, seed_columns, training_seed_path)
    seed_results["training_seed"] = pd.to_numeric(
        seed_results["training_seed"], errors="raise"
    ).astype(int)
    if set(seed_results["training_seed"]) != set(SEEDS) or len(seed_results) != 3:
        raise ValueError(
            f"training_seed_results.csv phải chứa đúng seed {SEEDS}, nhận được "
            f"{seed_results['training_seed'].tolist()}"
        )
    seed_results = seed_results.set_index("training_seed").loc[list(SEEDS)].reset_index()
    numeric_seed_columns = sorted(seed_columns.difference({"run_name"}))
    require_finite(seed_results, numeric_seed_columns, training_seed_path)
    if not (seed_results["evaluation_episodes"].astype(int) == 30).all():
        raise ValueError("Mỗi seed phải có đúng 30 episode đánh giá.")
    reconstructed_overflow = (
        seed_results["ordinary_overflow_mbit"]
        / seed_results["ordinary_offered_mbit"]
    )
    if not np.allclose(
        reconstructed_overflow,
        seed_results["ordinary_overflow_fraction"],
        rtol=1e-10,
        atol=1e-12,
    ):
        raise ValueError("ordinary_overflow_fraction không khớp overflow/offered.")

    learning = pd.read_csv(learning_path)
    learning_columns = {
        "episode",
        "timesteps",
        *{f"reward_seed_{seed}" for seed in SEEDS},
        *{f"reward_smooth_seed_{seed}" for seed in SEEDS},
        "reward_smooth_mean",
        "reward_smooth_sd",
    }
    require_columns(learning, learning_columns, learning_path)
    require_finite(learning, sorted(learning_columns), learning_path)
    if not learning["timesteps"].is_monotonic_increasing:
        raise ValueError("timesteps không tăng đơn điệu.")
    if int(learning["timesteps"].max()) != 100_000:
        raise ValueError("Learning curve không kết thúc tại 100000 timesteps.")
    smooth_columns = []
    for seed in SEEDS:
        expected = learning[f"reward_seed_{seed}"].rolling(20, min_periods=1).mean()
        actual_column = f"reward_smooth_seed_{seed}"
        if not np.allclose(expected, learning[actual_column], rtol=1e-9, atol=1e-8):
            raise ValueError(f"{actual_column} không phải trung bình trượt 20 episode.")
        smooth_columns.append(actual_column)
    expected_mean = learning[smooth_columns].mean(axis=1)
    expected_sd = learning[smooth_columns].std(axis=1, ddof=1)
    if not np.allclose(
        expected_mean, learning["reward_smooth_mean"], rtol=1e-9, atol=1e-8
    ):
        raise ValueError("reward_smooth_mean không khớp trung bình ba seed.")
    if not np.allclose(
        expected_sd, learning["reward_smooth_sd"], rtol=1e-9, atol=1e-8
    ):
        raise ValueError("reward_smooth_sd không khớp SD mẫu giữa ba seed.")

    actions = pd.read_csv(action_path)
    action_columns = {
        "training_seed",
        "emergency",
        "action",
        "ambulance_prb_percent",
        "rate",
        "count",
    }
    require_columns(actions, action_columns, action_path)
    require_finite(
        actions,
        ["training_seed", "action", "ambulance_prb_percent", "rate", "count"],
        action_path,
    )
    actions["training_seed"] = actions["training_seed"].astype(int)
    if actions["emergency"].dtype != bool:
        actions["emergency"] = actions["emergency"].map(
            {"True": True, "False": False, True: True, False: False}
        )
    if actions["emergency"].isna().any():
        raise ValueError("Cột emergency chứa giá trị không hợp lệ.")
    expected_action_rows = len(SEEDS) * 2 * 8
    if len(actions) != expected_action_rows:
        raise ValueError(
            f"Phải có {expected_action_rows} hàng phân bố hành động, nhận {len(actions)}."
        )
    action_sums: dict[str, float] = {}
    for seed in SEEDS:
        for emergency in (False, True):
            subset = actions[
                (actions["training_seed"] == seed)
                & (actions["emergency"] == emergency)
            ].sort_values("action")
            if subset["action"].astype(int).tolist() != list(range(8)):
                raise ValueError(f"Thiếu action cho seed={seed}, emergency={emergency}.")
            if subset["ambulance_prb_percent"].astype(int).tolist() != list(
                range(10, 81, 10)
            ):
                raise ValueError(f"Sai ánh xạ PRB cho seed={seed}, emergency={emergency}.")
            rate_sum = float(subset["rate"].sum())
            state_count = float(subset["count"].sum())
            reconstructed_rate = subset["count"] / state_count
            if not np.isclose(rate_sum, 1.0, atol=1e-9):
                raise ValueError(
                    f"Tổng tỷ lệ action không bằng 1: seed={seed}, "
                    f"emergency={emergency}, sum={rate_sum}"
                )
            if not np.allclose(
                reconstructed_rate, subset["rate"], rtol=1e-10, atol=1e-12
            ):
                raise ValueError(
                    f"rate không khớp count: seed={seed}, emergency={emergency}"
                )
            state_name = "emergency" if emergency else "non_emergency"
            action_sums[f"seed_{seed}_{state_name}_percent"] = rate_sum * 100.0

    comparison_paths: dict[int, Path] = {}
    comparisons: dict[int, pd.DataFrame] = {}
    comparison_required = {
        "policy",
        "n_episodes",
        "avg_ambulance_latency_ms_mean",
        "avg_ambulance_latency_ms_unit",
        "p95_ambulance_latency_ms_mean",
        "p95_ambulance_latency_ms_unit",
        "ambulance_sla_violation_rate_mean",
        "ambulance_sla_violation_rate_unit",
        "avg_ordinary_demand_satisfaction_mean",
        "avg_ordinary_demand_satisfaction_unit",
        "avg_ordinary_throughput_emergency_mbps_mean",
        "avg_ordinary_throughput_emergency_mbps_unit",
        "p95_ordinary_queue_mbit_mean",
        "p95_ordinary_queue_mbit_unit",
        "ordinary_overflow_fraction_mean",
        "ordinary_overflow_fraction_unit",
    }
    expected_units = {
        "avg_ambulance_latency_ms_unit": "ms",
        "p95_ambulance_latency_ms_unit": "ms",
        "ambulance_sla_violation_rate_unit": "fraction",
        "avg_ordinary_demand_satisfaction_unit": "fraction",
        "avg_ordinary_throughput_emergency_mbps_unit": "Mbit/s",
        "p95_ordinary_queue_mbit_unit": "Mbit",
        "ordinary_overflow_fraction_unit": "fraction",
    }
    crosscheck_pairs = {
        "ambulance_latency_ms_mean": "avg_ambulance_latency_ms_mean",
        "ambulance_p95_ms_mean": "p95_ambulance_latency_ms_mean",
        "ambulance_sla": "ambulance_sla_violation_rate_mean",
        "ordinary_demand_satisfaction": "avg_ordinary_demand_satisfaction_mean",
        "ordinary_throughput_emergency_mbps": (
            "avg_ordinary_throughput_emergency_mbps_mean"
        ),
        "ordinary_queue_p95_mbit": "p95_ordinary_queue_mbit_mean",
    }
    baseline_values: dict[str, dict[str, float]] = {}
    baseline_metrics = {
        "ambulance_latency_ms": "avg_ambulance_latency_ms_mean",
        "ambulance_p95_ms": "p95_ambulance_latency_ms_mean",
        "ambulance_sla_fraction": "ambulance_sla_violation_rate_mean",
    }
    for seed in SEEDS:
        run_name = str(
            seed_results.loc[seed_results["training_seed"] == seed, "run_name"].iloc[0]
        )
        comparison_path = RESULTS_ROOT / run_name / "final_model_evaluation" / "evaluation_comparison.csv"
        if not comparison_path.is_file():
            raise FileNotFoundError(f"Thiếu evaluation_comparison.csv: {comparison_path}")
        comparison = pd.read_csv(comparison_path)
        require_columns(comparison, comparison_required, comparison_path)
        expected_policies = {"DQN", "Static", "Guaranteed Load-based"}
        if set(comparison["policy"]) != expected_policies:
            raise ValueError(
                f"Sai tập policy trong {comparison_path}: {comparison['policy'].tolist()}"
            )
        for unit_column, expected_unit in expected_units.items():
            observed = set(comparison[unit_column].astype(str))
            if observed != {expected_unit}:
                raise ValueError(
                    f"Sai đơn vị {unit_column} trong {comparison_path}: {observed}"
                )
        dqn_row = comparison.set_index("policy").loc["DQN"]
        seed_row = seed_results.set_index("training_seed").loc[seed]
        for seed_column, comparison_column in crosscheck_pairs.items():
            if not np.isclose(
                float(seed_row[seed_column]),
                float(dqn_row[comparison_column]),
                rtol=1e-10,
                atol=1e-12,
            ):
                raise ValueError(
                    f"KPI DQN không khớp seed={seed}: {seed_column} vs "
                    f"{comparison_column}"
                )
        comparison_paths[seed] = comparison_path
        comparisons[seed] = comparison
        indexed = comparison.set_index("policy")
        for policy in ("Static", "Guaranteed Load-based"):
            values = {
                name: float(indexed.loc[policy, column])
                for name, column in baseline_metrics.items()
            }
            if policy not in baseline_values:
                baseline_values[policy] = values
            elif not all(
                np.isclose(values[key], baseline_values[policy][key], atol=1e-12)
                for key in values
            ):
                raise ValueError(f"Baseline {policy} không nhất quán giữa ba file đánh giá.")

    seed43_run_name = str(
        seed_results.loc[seed_results["training_seed"] == 43, "run_name"].iloc[0]
    )
    overflow_steps_path = (
        RESULTS_ROOT
        / seed43_run_name
        / "final_model_evaluation"
        / "evaluation_steps.csv"
    )
    if not overflow_steps_path.is_file():
        raise FileNotFoundError(f"Thiếu dữ liệu quỹ đạo seed 43: {overflow_steps_path}")
    overflow_steps = pd.read_csv(overflow_steps_path)
    overflow_columns = {
        "policy",
        "episode",
        "seed",
        "step",
        "alpha_A",
        "ambulance_emergency",
        "ordinary_arrival_mbit",
        "ordinary_capacity_mbps",
        "ordinary_queue_mbit",
        "ordinary_overflow_mbit",
    }
    require_columns(overflow_steps, overflow_columns, overflow_steps_path)
    overflow_case = overflow_steps[
        (overflow_steps["policy"] == "DQN")
        & (pd.to_numeric(overflow_steps["seed"], errors="coerce") == 10026)
    ].copy()
    require_finite(
        overflow_case,
        [
            "episode",
            "seed",
            "step",
            "alpha_A",
            "ordinary_arrival_mbit",
            "ordinary_capacity_mbps",
            "ordinary_queue_mbit",
            "ordinary_overflow_mbit",
        ],
        overflow_steps_path,
    )
    if len(overflow_case) != 200 or set(overflow_case["episode"].astype(int)) != {26}:
        raise ValueError(
            "Trường hợp seed môi trường 10026 phải có đúng 200 bước ở episode 26."
        )
    if overflow_case["ambulance_emergency"].dtype != bool:
        overflow_case["ambulance_emergency"] = overflow_case[
            "ambulance_emergency"
        ].map({"True": True, "False": False, True: True, False: False})
    if overflow_case["ambulance_emergency"].isna().any():
        raise ValueError("Cột ambulance_emergency chứa giá trị không hợp lệ.")
    overflow_case = overflow_case.sort_values("step")
    if int((overflow_case["ordinary_overflow_mbit"] > 0).sum()) == 0:
        raise ValueError("Seed môi trường 10026 không chứa bước tràn dữ liệu.")

    return {
        "training_seed_path": training_seed_path,
        "learning_path": learning_path,
        "action_path": action_path,
        "comparison_paths": comparison_paths,
        "overflow_steps_path": overflow_steps_path,
        "seed_results": seed_results,
        "learning": learning,
        "actions": actions,
        "baseline_values": baseline_values,
        "action_sums": action_sums,
        "overflow_case": overflow_case,
    }


def plot_learning_curve(learning: pd.DataFrame, output_dir: Path) -> list[Path]:
    fig, axis = plt.subplots(figsize=(6.8, 3.55))
    x = learning["timesteps"].to_numpy(dtype=float)
    mean = learning["reward_smooth_mean"].to_numpy(dtype=float)
    sd = learning["reward_smooth_sd"].to_numpy(dtype=float)
    axis.fill_between(
        x,
        mean - sd,
        mean + sd,
        color=COLOR_DQN,
        alpha=0.18,
        linewidth=0,
        label="Trung bình ± 1 độ lệch chuẩn",
    )
    axis.plot(x, mean, color=COLOR_DQN, linewidth=2.0, label="Trung bình 3 seed")
    axis.set_xlim(0, 100_000)
    axis.set_xticks(np.arange(0, 100_001, 20_000))
    axis.set_xlabel("Số bước huấn luyện")
    axis.set_ylabel("Phần thưởng episode\n(trung bình trượt 20 episode)")
    style_axis(axis)
    axis.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    return save_figure(fig, output_dir, "fig_learning_curve_three_seeds")


def plot_ambulance_kpi(
    seed_results: pd.DataFrame,
    baseline_values: dict[str, dict[str, float]],
    output_dir: Path,
) -> list[Path]:
    seeds = seed_results["training_seed"].to_numpy(dtype=int)
    metrics = (
        (
            "ambulance_latency_ms_mean",
            "ambulance_latency_ms",
            "Trễ trung bình Emergency (ms)",
            1.0,
        ),
        (
            "ambulance_p95_ms_mean",
            "ambulance_p95_ms",
            "Trễ P95 Emergency (ms)",
            1.0,
        ),
        (
            "ambulance_sla",
            "ambulance_sla_fraction",
            "Tỷ lệ vi phạm SLA (%)",
            100.0,
        ),
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.25))
    line_handles = None
    for panel_index, (axis, (seed_column, baseline_key, ylabel, scale)) in enumerate(
        zip(axes, metrics)
    ):
        values = seed_results[seed_column].to_numpy(dtype=float) * scale
        dqn_handle = axis.plot(
            seeds,
            values,
            color=COLOR_DQN,
            marker="o",
            markersize=5.0,
            linewidth=2.0,
            label="DQN",
            zorder=4,
        )[0]
        static_value = baseline_values["Static"][baseline_key] * scale
        glb_value = baseline_values["Guaranteed Load-based"][baseline_key] * scale
        static_handle = axis.axhline(
            static_value,
            color=COLOR_STATIC,
            linestyle="--",
            linewidth=1.6,
            label="Static",
            zorder=2,
        )
        glb_handle = axis.axhline(
            glb_value,
            color=COLOR_GLB,
            linestyle="-.",
            linewidth=1.6,
            label="Guaranteed Load-based",
            zorder=2,
        )
        handles = [dqn_handle, static_handle, glb_handle]
        if panel_index == 1:
            sla_handle = axis.axhline(
                100.0,
                color=COLOR_SLA,
                linestyle="--",
                linewidth=1.5,
                label="Ngưỡng SLA 100 ms",
                zorder=1,
            )
            handles.append(sla_handle)
        axis.set_xticks(seeds)
        axis.set_xlabel("Seed huấn luyện DQN")
        axis.set_ylabel(ylabel)
        axis.set_ylim(bottom=0)
        axis.text(
            0.03,
            0.96,
            f"({chr(ord('a') + panel_index)})",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
        )
        style_axis(axis)
        if panel_index == 1:
            line_handles = handles
    if line_handles is None:
        raise AssertionError("Không tạo được legend cho Ambulance KPI.")
    fig.legend(
        handles=line_handles,
        labels=[handle.get_label() for handle in line_handles],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        frameon=False,
        ncol=2,
        handlelength=2.4,
    )
    fig.subplots_adjust(left=0.08, right=0.99, top=0.98, bottom=0.34, wspace=0.48)
    return save_figure(fig, output_dir, "fig_ambulance_kpi_by_seed")


def add_bar_labels(axis: plt.Axes, bars, values: np.ndarray, format_spec: str) -> None:
    labels = [format(value, format_spec) for value in values]
    axis.bar_label(bars, labels=labels, padding=3, fontsize=8.0, color="#202020")


def plot_ordinary_kpi(seed_results: pd.DataFrame, output_dir: Path) -> list[Path]:
    seeds = seed_results["training_seed"].to_numpy(dtype=int)
    metrics = (
        (
            "ordinary_demand_satisfaction",
            "Đáp ứng nhu cầu Ordinary (%)",
            100.0,
            ".2f",
            (0.0, 110.0),
        ),
        (
            "ordinary_throughput_emergency_mbps",
            "Thông lượng khi khẩn cấp (Mbit/s)",
            1.0,
            ".2f",
            None,
        ),
        (
            "ordinary_queue_p95_mbit",
            "Hàng đợi P95 của Ordinary (Mbit)",
            1.0,
            ".2f",
            None,
        ),
        (
            "ordinary_overflow_fraction",
            "Dữ liệu tràn của Ordinary (%)",
            100.0,
            ".3f",
            None,
        ),
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.55), constrained_layout=True)
    for panel_index, (axis, (column, ylabel, scale, fmt, fixed_ylim)) in enumerate(
        zip(axes.flat, metrics)
    ):
        values = seed_results[column].to_numpy(dtype=float) * scale
        bars = axis.bar(
            seeds.astype(str),
            values,
            width=0.58,
            color=COLOR_DQN,
            edgecolor="#1A1A1A",
            linewidth=0.65,
        )
        add_bar_labels(axis, bars, values, fmt)
        axis.set_xlabel("Seed huấn luyện DQN")
        axis.set_ylabel(ylabel)
        if fixed_ylim is not None:
            axis.set_ylim(*fixed_ylim)
        else:
            maximum = float(np.max(values))
            axis.set_ylim(0.0, maximum * 1.30 if maximum > 0 else 1.0)
        axis.text(
            0.03,
            0.96,
            f"({chr(ord('a') + panel_index)})",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
        )
        style_axis(axis)
    return save_figure(fig, output_dir, "fig_ordinary_kpi_by_seed")


def plot_action_distribution(actions: pd.DataFrame, output_dir: Path) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25), sharey=True)
    seed_styles = {
        42: ("-", "o"),
        43: ("--", "s"),
        44: (":", "^")
    }
    for panel_index, (axis, emergency) in enumerate(zip(axes, (False, True))):
        for seed in SEEDS:
            subset = actions[
                (actions["training_seed"] == seed)
                & (actions["emergency"] == emergency)
            ].sort_values("ambulance_prb_percent")
            line_style, marker = seed_styles[seed]
            axis.plot(
                subset["ambulance_prb_percent"],
                subset["rate"] * 100.0,
                color=SEED_COLORS[seed],
                linestyle=line_style,
                marker=marker,
                markersize=4.5,
                linewidth=1.8,
                label=f"Seed {seed}",
            )
        axis.set_xticks(np.arange(10, 81, 10), [f"{value}%" for value in range(10, 81, 10)])
        axis.set_xlabel("PRB cấp cho Emergency Slice")
        axis.set_ylim(0.0, 100.0)
        state_label = "Không khẩn cấp" if not emergency else "Khẩn cấp"
        axis.text(
            0.00,
            1.05,
            f"({chr(ord('a') + panel_index)}) {state_label}",
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontweight="bold",
            clip_on=False,
        )
        style_axis(axis)
    axes[0].set_ylabel("Tỷ lệ chọn hành động (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=3,
        frameon=False,
    )
    fig.subplots_adjust(left=0.09, right=0.99, top=0.87, bottom=0.25, wspace=0.12)
    return save_figure(fig, output_dir, "fig_dqn_action_distribution_three_seeds")


def plot_seed43_overflow_case_legacy(
    overflow_case: pd.DataFrame, output_dir: Path
) -> list[Path]:
    window = overflow_case[
        (overflow_case["step"] >= 35) & (overflow_case["step"] <= 90)
    ].copy()
    x = window["step"].to_numpy(dtype=float)
    emergency = window["ambulance_emergency"].to_numpy(dtype=bool)

    fig, axes = plt.subplots(3, 1, figsize=(7.0, 5.5), sharex=True)
    axes[0].step(
        x,
        window["alpha_A"].to_numpy(dtype=float) * 100.0,
        where="post",
        color=COLOR_DQN,
        linewidth=1.8,
        label="PRB cấp cho Emergency",
    )
    axes[0].fill_between(
        x,
        0,
        85,
        where=emergency,
        step="post",
        color="#E69F00",
        alpha=0.16,
        label="Trạng thái khẩn cấp",
    )
    axes[0].set_ylim(0, 85)
    axes[0].set_ylabel("PRB Emergency (%)")
    axes[0].legend(frameon=False, loc="lower left", ncol=2)

    axes[1].plot(
        x,
        window["ordinary_arrival_mbit"],
        color=COLOR_STATIC,
        linestyle="--",
        linewidth=1.5,
        label="Dữ liệu đến",
    )
    axes[1].plot(
        x,
        window["ordinary_capacity_mbps"],
        color=COLOR_GLB,
        linewidth=1.7,
        label="Dung lượng phục vụ",
    )
    axes[1].set_ylabel("Ordinary (Mbit/s)")
    axes[1].set_ylim(bottom=0)
    axes[1].legend(frameon=False, loc="upper left", bbox_to_anchor=(0.02, 1.12), ncol=2)

    axes[2].plot(
        x,
        window["ordinary_queue_mbit"],
        color=COLOR_DQN,
        linewidth=1.8,
        label="Hàng đợi Ordinary",
    )
    axes[2].bar(
        x,
        window["ordinary_overflow_mbit"],
        width=0.8,
        color=COLOR_SLA,
        alpha=0.55,
        label="Dữ liệu tràn",
    )
    axes[2].axhline(
        100.0,
        color=COLOR_STATIC,
        linestyle="--",
        linewidth=1.3,
        label="Giới hạn hàng đợi 100 Mbit",
    )
    axes[2].set_ylabel("Dữ liệu (Mbit)")
    axes[2].set_xlabel("Chu kỳ quyết định")
    axes[2].set_ylim(0, 110)
    axes[2].legend(frameon=False, loc="lower right", ncol=3)

    for panel_index, axis in enumerate(axes):
        axis.text(
            0.01,
            0.94,
            f"({chr(ord('a') + panel_index)})",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
        )
        style_axis(axis)
    axes[2].set_xlim(35, 90)
    axes[2].set_xticks(np.arange(35, 91, 5))
    fig.subplots_adjust(left=0.11, right=0.99, top=0.99, bottom=0.10, hspace=0.13)
    return save_figure(fig, output_dir, "fig_seed43_overflow_case")


def build_audit_payload(data: dict[str, object], output_files: list[Path]) -> dict[str, object]:
    seed_results = data["seed_results"]
    if not isinstance(seed_results, pd.DataFrame):
        raise TypeError("seed_results không phải DataFrame.")
    comparison_paths = data["comparison_paths"]
    if not isinstance(comparison_paths, dict):
        raise TypeError("comparison_paths không phải dict.")
    source_paths = [
        data["training_seed_path"],
        data["learning_path"],
        data["action_path"],
        data["overflow_steps_path"],
        *comparison_paths.values(),
    ]
    typed_sources = [path for path in source_paths if isinstance(path, Path)]
    return {
        "sources": [
            {"path": str(path), "sha256": sha256(path)} for path in typed_sources
        ],
        "columns_used": {
            "learning_curve_three_training_seeds.csv": [
                "timesteps",
                "reward_seed_42",
                "reward_seed_43",
                "reward_seed_44",
                "reward_smooth_seed_42",
                "reward_smooth_seed_43",
                "reward_smooth_seed_44",
                "reward_smooth_mean",
                "reward_smooth_sd",
            ],
            "training_seed_results.csv": [
                "training_seed",
                "run_name",
                "ambulance_latency_ms_mean",
                "ambulance_p95_ms_mean",
                "ambulance_sla",
                "ordinary_demand_satisfaction",
                "ordinary_throughput_emergency_mbps",
                "ordinary_queue_p95_mbit",
                "ordinary_overflow_fraction",
                "ordinary_overflow_mbit",
                "ordinary_offered_mbit",
            ],
            "action_distribution_by_training_seed.csv": [
                "training_seed",
                "emergency",
                "action",
                "ambulance_prb_percent",
                "rate",
                "count",
            ],
            "evaluation_comparison.csv": [
                "policy",
                "avg_ambulance_latency_ms_mean",
                "p95_ambulance_latency_ms_mean",
                "ambulance_sla_violation_rate_mean",
                "avg_ordinary_demand_satisfaction_mean",
                "avg_ordinary_throughput_emergency_mbps_mean",
                "p95_ordinary_queue_mbit_mean",
                "*_unit tương ứng",
            ],
            "evaluation_steps.csv (DQN seed 43, environment seed 10026)": [
                "episode",
                "seed",
                "step",
                "alpha_A",
                "ambulance_emergency",
                "ordinary_arrival_mbit",
                "ordinary_capacity_mbps",
                "ordinary_queue_mbit",
                "ordinary_overflow_mbit",
            ],
        },
        "unit_validation": {
            "ambulance_latency": "ms",
            "ambulance_p95": "ms",
            "ambulance_sla": "fraction, plotted as percent",
            "ordinary_demand_satisfaction": "fraction, plotted as percent",
            "ordinary_emergency_throughput": "Mbit/s",
            "ordinary_p95_queue": "Mbit",
            "ordinary_overflow_fraction": "fraction, plotted as percent",
        },
        "action_rate_sums": data["action_sums"],
        "baseline_values": data["baseline_values"],
        "dqn_kpi_points": seed_results.to_dict(orient="records"),
        "output_files": [str(path) for path in output_files],
    }


def plot_system_model_overview(thesis_dir: Path) -> list[Path]:
    """Draw Figure 3.1 as a closed-loop RAN slicing control diagram."""
    output_dir = thesis_dir / "Images"
    fig, axis = plt.subplots(figsize=(7.4, 4.5))
    axis.set_xlim(0, 12)
    axis.set_ylim(0, 7.2)
    axis.axis("off")

    def box(
        x: float,
        y: float,
        width: float,
        height: float,
        text: str,
        facecolor: str,
        edgecolor: str = "#222222",
        fontsize: float = 9.0,
        weight: str = "normal",
    ) -> matplotlib.patches.FancyBboxPatch:
        patch = matplotlib.patches.FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.035,rounding_size=0.08",
            linewidth=1.0,
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
            linespacing=1.15,
        )
        return patch

    def arrow(
        start: tuple[float, float],
        end: tuple[float, float],
        text: str | None = None,
        text_xy: tuple[float, float] | None = None,
        color: str = "#222222",
        connectionstyle: str = "arc3,rad=0.0",
    ) -> None:
        axis.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops=dict(
                arrowstyle="-|>",
                linewidth=1.15,
                color=color,
                shrinkA=2,
                shrinkB=2,
                connectionstyle=connectionstyle,
            ),
        )
        if text is not None and text_xy is not None:
            axis.text(
                text_xy[0],
                text_xy[1],
                text,
                ha="center",
                va="center",
                fontsize=8.1,
                color=color,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.8),
                linespacing=1.1,
            )

    box(0.45, 0.75, 2.1, 0.82, "Xe cứu thương\nlưu lượng khẩn cấp", "#FAD7CD", fontsize=8.5)
    box(0.45, 2.00, 2.1, 0.82, "Xe thông thường\nlưu lượng nền", "#E2E8DD", fontsize=8.5)
    box(3.35, 1.15, 2.15, 1.15, "gNB/RAN\nmột ô phủ", "#FFF2CC", fontsize=9.5, weight="bold")
    box(6.20, 1.15, 1.35, 1.15, "Kho\nPRB", "#FFE599", fontsize=9.2, weight="bold")
    box(8.15, 2.15, 2.65, 0.82, "Emergency Slice\nưu tiên độ trễ", "#F6A391", fontsize=9.0, weight="bold")
    box(8.15, 0.55, 2.65, 0.82, "Ordinary Traffic Slice\nthông lượng, hàng đợi", "#D9E1D4", fontsize=8.6, weight="bold")

    box(3.15, 5.05, 5.55, 1.08, "", "#E8F1FA", fontsize=11.0, weight="bold")
    axis.text(5.92, 5.91, "Near-RT RIC", ha="center", va="center", fontsize=10.7, fontweight="bold")
    box(3.65, 5.25, 4.55, 0.42, "xApp phân mảnh RAN dựa trên DQN", "#D6E6F5", fontsize=8.4)
    box(0.65, 4.95, 2.00, 1.25, "Trạng thái/KPI\n$t$: tải, hàng đợi,\ntrễ, thông lượng", "#EDF5FC", fontsize=7.8)
    box(9.25, 4.95, 2.00, 1.25, "Phần thưởng\n$r_t$: SLA, đáp ứng\nnhu cầu, tràn dữ liệu", "#EDF5FC", fontsize=7.6)

    arrow((2.55, 1.16), (3.35, 1.55))
    arrow((2.55, 2.41), (3.35, 1.95))
    arrow((5.50, 1.73), (6.20, 1.73))
    arrow((7.55, 2.05), (8.15, 2.56), "PRB", (7.86, 2.55))
    arrow((7.55, 1.45), (8.15, 0.96), "PRB", (7.86, 1.05))

    arrow((4.45, 2.30), (4.45, 5.05), "quan sát\n$s_t$", (3.72, 3.65), color=COLOR_DQN)
    arrow((7.35, 5.05), (7.05, 2.30), "hành động\n$a_t$: tỷ lệ PRB", (8.23, 3.65), color=COLOR_DQN)
    arrow((9.25, 5.57), (8.70, 5.57), color=COLOR_GLB)
    arrow((2.65, 5.57), (3.15, 5.57), color=COLOR_GLB)
    arrow(
        (10.20, 4.95),
        (10.15, 2.95),
        "đánh giá\nKPI",
        (10.88, 4.05),
        color=COLOR_GLB,
        connectionstyle="arc3,rad=0.08",
    )

    axis.text(1.50, 3.35, "Nguồn lưu lượng", ha="center", va="center", fontsize=8.6, fontweight="bold")
    axis.text(6.15, 0.35, "Mặt phẳng dữ liệu RAN", ha="center", va="center", fontsize=8.6, color="#555555")
    axis.text(6.00, 6.62, "Mặt phẳng điều khiển gần thời gian thực", ha="center", va="center", fontsize=8.6, color="#555555")

    fig.tight_layout(pad=0.1)
    return save_figure(fig, output_dir, "system_model_overview_vi")


def plot_seed43_overflow_case(
    overflow_case: pd.DataFrame, output_dir: Path
) -> list[Path]:
    """Draw Figure 4.5 with legends in a separate right-side column."""
    window = overflow_case[
        (overflow_case["step"] >= 35) & (overflow_case["step"] <= 90)
    ].copy()
    x = window["step"].to_numpy(dtype=float)
    emergency = window["ambulance_emergency"].to_numpy(dtype=bool)

    fig = plt.figure(figsize=(7.6, 5.55))
    grid = fig.add_gridspec(
        3,
        2,
        width_ratios=(4.9, 1.45),
        height_ratios=(1, 1, 1),
        hspace=0.24,
        wspace=0.08,
    )
    axes = [fig.add_subplot(grid[row, 0]) for row in range(3)]
    legend_axes = [fig.add_subplot(grid[row, 1]) for row in range(3)]
    for legend_axis in legend_axes:
        legend_axis.axis("off")

    axes[0].step(
        x,
        window["alpha_A"].to_numpy(dtype=float) * 100.0,
        where="post",
        color=COLOR_DQN,
        linewidth=1.8,
        label="PRB cấp cho Emergency",
    )
    axes[0].fill_between(
        x,
        0,
        85,
        where=emergency,
        step="post",
        color="#E69F00",
        alpha=0.16,
        label="Trạng thái khẩn cấp",
    )
    axes[0].set_ylim(0, 85)
    axes[0].set_ylabel("PRB Emergency (%)")

    axes[1].plot(
        x,
        window["ordinary_arrival_mbit"],
        color=COLOR_STATIC,
        linestyle="--",
        linewidth=1.5,
        label="Dữ liệu đến",
    )
    axes[1].plot(
        x,
        window["ordinary_capacity_mbps"],
        color=COLOR_GLB,
        linewidth=1.7,
        label="Dung lượng phục vụ",
    )
    axes[1].set_ylabel("Ordinary (Mbit/s)")
    axes[1].set_ylim(bottom=0)

    axes[2].plot(
        x,
        window["ordinary_queue_mbit"],
        color=COLOR_DQN,
        linewidth=1.8,
        label="Hàng đợi Ordinary",
    )
    axes[2].bar(
        x,
        window["ordinary_overflow_mbit"],
        width=0.8,
        color=COLOR_SLA,
        alpha=0.55,
        label="Dữ liệu tràn",
    )
    axes[2].axhline(
        100.0,
        color=COLOR_STATIC,
        linestyle="--",
        linewidth=1.3,
        label="Giới hạn hàng đợi\n100 Mbit",
    )
    axes[2].set_ylabel("Dữ liệu (Mbit)")
    axes[2].set_xlabel("Chu kỳ quyết định")
    axes[2].set_ylim(0, 110)

    for panel_index, axis in enumerate(axes):
        axis.text(
            0.01,
            0.93,
            f"({chr(ord('a') + panel_index)})",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
        )
        axis.set_xlim(35, 90)
        axis.set_xticks(np.arange(35, 91, 5))
        if panel_index < 2:
            axis.tick_params(labelbottom=False)
        style_axis(axis)

        handles, labels = axis.get_legend_handles_labels()
        legend_axes[panel_index].legend(
            handles,
            labels,
            frameon=False,
            loc="center left",
            handlelength=1.8,
            handletextpad=0.6,
            borderaxespad=0.0,
            labelspacing=0.9,
        )

    fig.subplots_adjust(left=0.09, right=0.985, top=0.985, bottom=0.10)
    return save_figure(fig, output_dir, "fig_seed43_overflow_case")


def main() -> None:
    configure_style()
    data = load_and_validate_sources()
    thesis_dir = find_thesis_dir()
    output_dir = thesis_dir / "Images" / "results_chapter4"

    seed_results = data["seed_results"]
    learning = data["learning"]
    actions = data["actions"]
    baseline_values = data["baseline_values"]
    overflow_case = data["overflow_case"]
    if not isinstance(seed_results, pd.DataFrame):
        raise TypeError("seed_results không phải DataFrame.")
    if not isinstance(learning, pd.DataFrame):
        raise TypeError("learning không phải DataFrame.")
    if not isinstance(actions, pd.DataFrame):
        raise TypeError("actions không phải DataFrame.")
    if not isinstance(baseline_values, dict):
        raise TypeError("baseline_values không phải dict.")
    if not isinstance(overflow_case, pd.DataFrame):
        raise TypeError("overflow_case không phải DataFrame.")

    output_files: list[Path] = []
    output_files.extend(plot_system_model_overview(thesis_dir))
    output_files.extend(plot_learning_curve(learning, output_dir))
    output_files.extend(
        plot_ambulance_kpi(seed_results, baseline_values, output_dir)
    )
    output_files.extend(plot_ordinary_kpi(seed_results, output_dir))
    output_files.extend(plot_action_distribution(actions, output_dir))
    output_files.extend(plot_seed43_overflow_case(overflow_case, output_dir))

    audit_payload = build_audit_payload(data, output_files)
    audit_path = output_dir / "figure_source_audit.json"
    audit_path.write_text(
        json.dumps(audit_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Validated data and saved {len(output_files)} figure files to {output_dir}")
    print(json.dumps(data["action_sums"], ensure_ascii=False, indent=2))
    print(f"Audit: {audit_path}")


if __name__ == "__main__":
    main()
