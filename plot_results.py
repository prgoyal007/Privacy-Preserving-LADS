import json, os, glob, re, math
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

ds_names = ["RobustSL", "ThresholdZipZipTree", "BiasedZipZipTree", "CTreap", "LTreap", "AVL"]
n_values = [100, 500, 1000, 2000]


# Publication-like rc styling 
mpl.rcParams.update({
    "font.family": "arial",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "axes.linewidth": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.grid": False,
})


# Color palette (similar to RobustSL paper look)
# color_map = {
#     "RobustSL": "#2A73C7",            # blue
#     "ThresholdZipZipTree": "#F2C57C", # warm tan
#     "BiasedZipZipTree": "#9ACD9A",    # light green
#     "CTreap": "#E6A0C4",              # light pink
#     "LTreap": "#C0C0F0",              # pale purple
#     "AVL": "#E8B4A2",                 # salmon
# }

# fallback palette if dataset missing
# default_colors = ["#2A73C7", "#9ACD9A", "#E6A0C4", "#C0C0F0", "#E8B4A2", "#F2C57C"]

# Okabe-Ito color-blind friendly palette
color_map = {
    "RobustSL": "#0072B2",            # blue
    "ThresholdZipZipTree": "#E69F00", # orange
    "BiasedZipZipTree": "#009E73",    # green
    "CTreap": "#CC79A7",              # pink
    "LTreap": "#56B4E9",              # light blue
    "AVL": "#D55E00",                 # red
}
default_colors = ["#0072B2", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#E69F00"]

def _alpha_to_fname_part(alpha: float) -> str:
    if float(alpha).is_integer():
        return str(int(round(alpha)))
    
    # Use repr-like conversion but strip trailing zeros
    s = ('%f' % alpha).rstrip('0').rstrip('.')
    return s

def detect_error_values(path_dir: str, ds: str, n: int, alpha: float) -> List[float]:
    alpha_part = _alpha_to_fname_part(alpha)
    results = set()

    # pattern looking for either with a `_e..._a{alpha}` or just `_a{alpha}`
    patterns = [
        os.path.join(path_dir, f"{ds}_n{n}_e*_a{alpha_part}.json"),
        os.path.join(path_dir, f"{ds}_n{n}_a{alpha_part}.json"),
    ]

    matches = []
    for pat in patterns:
        matches.extend(glob.glob(pat))
    
    # parse _e(\d+) occurences
    for fpath in matches:
        fname = os.path.basename(fpath)
        m = re.search(r"_e(\d+)_a", fname)

        if m:
            try:
                ev = int(m.group(1)) / 100.0
                results.add(ev)
            except ValueError:
                continue
        else:
            # no explicit _e token -> treat as 0.0 error
            results.add(0.0)

    # if nothing found, return [0.0] as safe default (fallback)
    if not results:
        return [0.0]
    return sorted(results)


def load_avg_costs(path_dir: str, 
                   ds_names: List[str], 
                   n_values: List[int], 
                   alpha_values: List[float], 
                   error_values: Optional[List[float]] = None) -> Dict[float, Dict[float, Dict[str, Dict[int, float]]]]:
    

    if alpha_values is None:
        raise ValueError("alpha_values must be provided (list of floats)")

    avg_costs = {}

    for alpha in alpha_values:
        alpha_part = _alpha_to_fname_part(alpha)
        avg_costs[alpha] = {}

        for ds in ds_names:
            for n in n_values:
                ev_list = error_values if error_values is not None else detect_error_values(path_dir, ds, n, alpha)
                
                for ev in ev_list:
                    avg_costs[alpha].setdefault(ev, {})
                    avg_costs[alpha][ev].setdefault(ds, {})

                    # try candidate filename forms (prefer _e{...} if ev>0, but try both)
                    candidates = []
                    if ev == 0.0:
                        # try both with and without _e0
                        candidates.append(os.path.join(path_dir, f"{ds}_n{n}_a{alpha_part}.json"))
                        candidates.append(os.path.join(path_dir, f"{ds}_n{n}_e0_a{alpha_part}.json"))
                    else:
                        candidates.append(os.path.join(path_dir, f"{ds}_n{n}_e{int(round(ev*100))}_a{alpha_part}.json"))

                    value = np.nan
                    for filename in candidates:
                        if os.path.exists(filename):
                            try:
                                with open(filename, 'r') as fh:
                                    data = json.load(fh)
                                costs = data.get("costs", [])
                                # If costs list is empty or missing, set NaN
                                if costs:
                                    value = float(np.mean(costs))
                                else:
                                    value = np.nan
                            except Exception as e:
                                # Any parse error -> record NaN but continue
                                value = np.nan
                            break

                    avg_costs[alpha][ev][ds][n] = value

    return avg_costs


def plot_grouped_bar(avg_costs_per_n: Dict[str, Dict[int, float]],
                     n_values: List[int],
                     title: str,
                     ylabel: str,
                     ax=None,
                     ds_order: Optional[List[str]] = None,
                     annotate_threshold: Optional[float] = None,
                     ymax_cap: Optional[float] = None):

    if ax is None:
        ax = plt.gca()

    if ds_order is None:
        ds_order_local = list(avg_costs_per_n.keys())
    else:
        ds_order_local = ds_order

    df = pd.DataFrame(
        {ds: [avg_costs_per_n.get(ds, {}).get(n, np.nan) for n in n_values]
         for ds in ds_order_local},
        index=n_values
    )

    num_groups = len(n_values)
    num_ds = len(ds_order_local)
    group_gap = 1.5
    group_width = 0.6
    bar_width = group_width / max(1, num_ds)
    x_centers = np.arange(num_groups) * group_gap
    hatch_patterns = ["/", "o", "|", "x", "+", "*", "\\", "O", ".", "-"]
    hatches = [hatch_patterns[i % len(hatch_patterns)] for i in range(num_ds)]

    offsets = (np.arange(num_ds) - (num_ds - 1) / 2.0) * bar_width

    all_values = df.to_numpy().flatten()
    finite_vals = all_values[np.isfinite(all_values)]
    if ymax_cap is None:
        ymax_cap = np.nanpercentile(finite_vals, 95) * 1.2 if finite_vals.size > 0 else 1.0

    overflow_tracker = {xc: 0 for xc in x_centers}

    for i, ds in enumerate(ds_order_local):
        heights = df[ds].values.astype(float)
        heights = np.where(np.isnan(heights), 0.0, heights)
        color = color_map.get(ds, default_colors[i % len(default_colors)])
        xpos = x_centers + offsets[i]

        bars = ax.bar(xpos, np.minimum(heights, ymax_cap),
                      width=bar_width, color=color, edgecolor='black',
                      linewidth=0.6, hatch=hatches[i], label=ds, zorder=3)

        for bar, h, group_center in zip(bars, heights, x_centers):
            if h > ymax_cap:
                ypos = min(h, ymax_cap)
                if overflow_tracker[group_center] == 0:
                    ax.text(bar.get_x() - 0.05, ypos - 0.5, f"{h:.1f}",
                            ha='right', va='center', fontsize=10, color='red', clip_on=False)
                else:
                    ax.text(bar.get_x() + bar.get_width() + 0.05, ypos - 0.5, f"{h:.1f}",
                            ha='left', va='center', fontsize=10, color='red', clip_on=False)
                overflow_tracker[group_center] += 1

    ax.set_ylim(0, ymax_cap * 1.1)
    ax.set_xticks(x_centers)
    ax.set_xticklabels([str(n) for n in n_values])
    ax.set_xlabel("Number of keys (n)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=12, weight="bold")
    ax.grid(axis="y", linestyle=":", linewidth=0.6, zorder=0)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)




def plot_zipf_parameter_sweep(avg_costs_per_alpha: Dict[float, Dict[str, float]],
                              alpha_values: List[float],
                              ds_names: List[str],
                              title: str,
                              ylabel: str,
                              ax=None,
                              annotate_threshold: Optional[float] = None,
                              save_path: Optional[str] = None,
                              ymax_cap: Optional[float] = None):
    if ax is None:
        ax = plt.gca() 

    ds_order_local = ds_names
    df = pd.DataFrame(
        {ds: [avg_costs_per_alpha.get(alpha, {}).get(ds, np.nan) for alpha in alpha_values]
         for ds in ds_order_local},
        index=[_alpha_to_fname_part(a) for a in alpha_values]
    )

    num_groups = len(alpha_values)
    num_ds = len(ds_order_local)
    group_gap = 1.5
    group_width = 0.6
    bar_width = group_width / max(1, num_ds)
    x_centers = np.arange(num_groups) * group_gap
    hatch_patterns = ["/", "o", "|", "x", "+", "*", "\\", "O", ".", "-"]
    hatches = [hatch_patterns[i % len(hatch_patterns)] for i in range(num_ds)]

    ax = plt.gca()
    offsets = (np.arange(num_ds) - (num_ds - 1) / 2.0) * bar_width

    all_values = df.to_numpy().flatten()
    finite_vals = all_values[np.isfinite(all_values)]
    if ymax_cap is None:
        ymax_cap = np.nanmax(finite_vals) * 1.2 if finite_vals.size > 0 else 1.0

    # Track overflows per x-center
    overflow_tracker = {xc: 0 for xc in x_centers}

    for i, ds in enumerate(ds_order_local):
        raw = df[ds].values.astype(float)
        nan_mask = np.isnan(raw)
        heights = np.where(nan_mask, 0.0, raw)
        color = color_map.get(ds, default_colors[i % len(default_colors)])
        xpos = x_centers + offsets[i]
        bars = ax.bar(xpos, np.minimum(heights, ymax_cap),
                      width=bar_width, color=color, edgecolor="black",
                      hatch=hatches[i], linewidth=0.6, label=ds, zorder=3)

        for bar, h, group_center in zip(bars, heights, x_centers):
            if h > ymax_cap:
                ypos = min(h, ymax_cap)
                if overflow_tracker[group_center] == 0:
                    # First overflow → place LEFT
                    ax.text(bar.get_x() - 0.1, ypos - 0.5,
                            f"{h:.1f}", ha='right', va='center',
                            fontsize=10, color='red', clip_on=False)
                else:
                    # Second overflow → place RIGHT
                    ax.text(bar.get_x() + bar.get_width() + 0.1, ypos - 0.5,
                            f"{h:.1f}", ha='left', va='center',
                            fontsize=10, color='red', clip_on=False)
                overflow_tracker[group_center] += 1

    ax.set_ylim(0, ymax_cap * 1.1)
    ax.set_xticks(x_centers)
    ax.set_xticklabels(df.index)
    ax.set_xlabel("Zipf parameter α")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=12, weight="bold")
    ax.grid(axis="y", linestyle=":", linewidth=0.6, zorder=0)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    # patches = [mpl.patches.Patch(facecolor=color_map.get(ds, default_colors[i % len(default_colors)]),
    #                              edgecolor="black", hatch=hatches[i], label=ds)
    #            for i, ds in enumerate(ds_order_local)]
    # # ax.legend(handles=patches, ncol=min(6, num_ds), frameon=False,
    # #           loc="upper center", bbox_to_anchor=(0.5, 1.25))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

def load_avg_sizes(path_dir: str, n_values: List[int]) -> Dict[int, float]:
    ds = "RobustSL"
    alpha_part = "2"
    avg_sizes = {}

    for n in n_values:
        fname = os.path.join(path_dir, f"{ds}_n{n}_e0_a{alpha_part}.json")
        value = np.nan
        if os.path.exists(fname):
            try:
                with open(fname, "r") as fh:
                    data = json.load(fh)
                sizes = data.get("size", [])
                value = float(np.mean(sizes)) if sizes else np.nan
            except Exception:
                value = np.nan
        else:
            print(f"File not found: {fname}")

        avg_sizes[n] = value

    return avg_sizes



def plot_sizes(avg_sizes_per_n: Dict[int, float], 
                       n_values: List[int], 
                       title: str, 
                       ylabel: str, 
                       ax=None, 
                       ymax_cap: Optional[float] = None):
    if ax is None:
        ax = plt.gca()
    
    heights = [avg_sizes_per_n.get(n, np.nan) for n in n_values]
    heights = np.array(heights, dtype=float)
    finite_vals = heights[np.isfinite(heights)]

    # Robust ymax_cap calculation
    if ymax_cap is None:
        if finite_vals.size > 0:
            ymax_cap = np.nanpercentile(finite_vals, 95) * 1.2
        else:
            ymax_cap = 1.0  # fallback

    x = np.arange(len(n_values))

    # Split into baseline (n) and overhead (extra beyond n)
    baseline = np.array(n_values, dtype=float)
    overhead = np.where(heights > baseline, heights - baseline, 0.0)

    # First bar (baseline, blue)
    bars_base = ax.bar(x, np.minimum(baseline, ymax_cap),
                       color=color_map.get("RobustSL", "#0072B2"),
                       edgecolor="black", width=0.6, label="Baseline size (n) nodes")

    # Second bar (overhead, stacked on baseline, red)
    bars_over = ax.bar(x, np.minimum(overhead, ymax_cap - baseline),
                       bottom=np.minimum(baseline, ymax_cap),
                       color=color_map.get("CTreap", "#CC79A7"), edgecolor="black", width=0.6, label="Additional overhead")

    # Annotate inside bars
    for i, (b_base, b_over, h) in enumerate(zip(bars_base, bars_over, heights)):
        if np.isfinite(h):
            if i == 0:
                # Special case: for the first bar, put annotations OUTSIDE
                # Baseline label just above the bar
                ax.text(b_base.get_x() + b_base.get_width()/2,
                        b_base.get_height() + 0.2,
                        f"{int(n_values[i])}",
                        ha="center", va="bottom", fontsize=9, color="black", rotation=0)

                # Overhead label just above total bar height
                if overhead[i] > 0:
                    ax.text(b_over.get_x() + b_over.get_width()/2,
                            b_base.get_height() + b_over.get_height() + 500,
                            f"{int(h)}",
                            ha="center", va="bottom", fontsize=9, color="red", rotation=0)
            elif i == 1:
                # Same thing for second bar, but baseline label INSIDE
                ax.text(b_base.get_x() + b_base.get_width()/2, 
                        b_base.get_height()/2, 
                        f"{int(n_values[i])}", 
                        ha="center", va="center", fontsize=9, color="white", rotation=0)
                
                if overhead[i] > 0:
                    ax.text(b_over.get_x() + b_over.get_width()/2,
                            b_base.get_height() + b_over.get_height() + 25,
                            f"{int(h)}",
                            ha="center", va="bottom", fontsize=9, color="red", rotation=0)
            else:
                # All other bars keep annotations INSIDE
                ax.text(b_base.get_x() + b_base.get_width()/2, 
                        b_base.get_height()/2, 
                        f"{int(n_values[i])}", 
                        ha="center", va="center", fontsize=9, color="white", rotation=0)

                if overhead[i] > 0:
                    ax.text(b_over.get_x() + b_over.get_width()/2, 
                            b_base.get_height() + b_over.get_height()/2, 
                            f"{int(h)}", 
                            ha="center", va="center", fontsize=9, color="white", rotation=0)


    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in n_values])
    ax.set_xlabel("Number of keys (n)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=12, weight="bold")
    ax.grid(axis="y", linestyle=":", linewidth=0.6)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.set_ylim(0, ymax_cap * 1.1)

    ax.legend(frameon=False, loc="upper left")



if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")

    fig1, axes1 = plt.subplots(1, 3, figsize=(6 * 3, 5), sharey=False)

    # Test 1 (α=2, δ=0) - ROTZipfianTest
    path_rot = os.path.join(results_dir, "ROTZipfianTest")
    avg_rot = load_avg_costs(path_rot, ds_names, n_values, alpha_values=[2.0])
    plot_grouped_bar({ds: {n: avg_rot[2.0][0.0].get(ds, {}).get(n, np.nan) for n in n_values} for ds in ds_names},
                     n_values,
                     "Ideal Zipfian Test (α=2, δ=0)",
                     "Avg. # of Comparisons per Query",
                     ax=axes1[0],
                     ds_order=ds_names,
                     annotate_threshold=25,
                     ymax_cap=25)

    # Test 2 (α=2, δ=0.9) - ROFZipfianTest
    path_rof = os.path.join(results_dir, "ROFZipfianTest")
    avg_rof = load_avg_costs(path_rof, ds_names, n_values, alpha_values=[2.0], error_values=[0.9])
    plot_grouped_bar({ds: {n: avg_rof[2.0][0.9].get(ds, {}).get(n, np.nan) for n in n_values} for ds in ds_names},
                     n_values,
                     "Noisy Zipfian Test (α=2, δ=0.9)",
                     "Avg. # of Comparisons per Query",
                     ax=axes1[1],
                     ds_order=ds_names,
                     annotate_threshold=25,
                     ymax_cap=25)

    # Test 3 (α varies, n=2000, δ=0)
    alpha_sweep = [1, 1.25, 1.5, 2, 3]
    avg_alpha = load_avg_costs(path_rof, ds_names, n_values=[2000], alpha_values=alpha_sweep, error_values=[0.0])
    avg_alpha_flat = {alpha: {ds: avg_alpha[alpha][0.0].get(ds, {}).get(2000, np.nan) for ds in ds_names} for alpha in alpha_sweep}
    plot_zipf_parameter_sweep(avg_alpha_flat, alpha_sweep, ds_names,
                              "Impact of Zipf Parameter α\n on DS Performance (n=2000, δ=0)",
                              "Avg. # of Comparisons per Query",
                              ax=axes1[2],
                              annotate_threshold=25,
                              ymax_cap=25)
    
    # Global legend for Figure 1
    handles, labels = axes1[0].get_legend_handles_labels()
    plt.subplots_adjust(wspace=0.35)  # wider gap between subplots
    fig1.legend(handles, labels, ncol=6, frameon=False,
                loc="upper center", bbox_to_anchor=(0.5, 0.95))
    plt.show()

    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

    # Test 4 (α=1.01, δ=0,0.9) - InversePowerTest
    path_ip = os.path.join(results_dir, "InversePowerTest")
    avg_ip = load_avg_costs(path_ip, ds_names, n_values, alpha_values=[1.01], error_values=[0.0, 0.9])
    
    plot_grouped_bar({ds: {n: avg_ip[1.01][0.0].get(ds, {}).get(n, np.nan) for n in n_values} for ds in ds_names},
                     n_values,
                     "Inverse Power Distribution Test (α=1.01, δ=0.0)",
                     "Avg. # of Comparisons per Query",
                     ax=axes2[0],
                     ds_order=ds_names,
                     annotate_threshold=25,
                     ymax_cap=25)

    plot_grouped_bar({ds: {n: avg_ip[1.01][0.9].get(ds, {}).get(n, np.nan) for n in n_values} for ds in ds_names},
                     n_values,
                     "Inverse Power Distribution Test (α=1.01, δ=0.9)",
                     "Avg. # of Comparisons per Query",
                     ax=axes2[1],
                     ds_order=ds_names,
                     annotate_threshold=25,
                     ymax_cap=25)

    # Global legend for Figure 2
    handles, labels = axes2[0].get_legend_handles_labels()
    fig2.subplots_adjust(top=0.75)
    fig2.legend(handles, labels, ncol=6, frameon=False,
                loc="upper center", bbox_to_anchor=(0.5, 0.89))
    # plt.tight_layout()
    plt.show()

    # Plot sizes for RobustSL only
    sizes_dir = os.path.join(results_dir, "SizeTest")
    n_values = [1000, 2000, 5000, 10000]
    avg_sizes = load_avg_sizes(sizes_dir, n_values)
    print("Loaded avg_sizes:", avg_sizes)

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_sizes(avg_sizes_per_n=avg_sizes,
            n_values=n_values,
            title="Average Size of RobustSL (α=2)",
            ylabel="Avg. Number of Nodes",
            ax=ax)
    plt.show()