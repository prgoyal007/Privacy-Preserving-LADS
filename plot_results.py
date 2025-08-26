import json, os, glob, re, math
from typing import List, Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt

ds_names = ["RobustSL", "ThresholdZipZipTree", "BiasedZipZipTree", "CTreap", "LTreap", "AVL"]
n_values = [100, 500, 1000, 2000]


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

                    found = False
                    value = np.nan
                    for filename in candidates:
                        if os.path.exists(filename):
                            found = True
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

                    # If none found, keep np.nan
                    avg_costs[alpha][ev][ds][n] = value

    return avg_costs


def plot_grouped_bar(avg_costs_per_n: Dict[str, Dict[int, float]],
                     n_values: List[int],
                     title: str,
                     ylabel: str,
                     ds_order: Optional[List[str]] = None,
                     annotate_threshold: Optional[float] = None):

    if ds_order is None:
        ds_names = list(avg_costs_per_n.keys())
    else:
        ds_names = ds_order

    x = np.arange(len(ds_names))
    # adapt width to number of groups to avoid overlap
    width = 0.8 / max(1, len(n_values))

    plt.figure(figsize=(10, 6))

    # For visual consistency, convert missing values (np.nan) to 0 for plotting,
    # but annotate or warn about missing values.
    for i, n_val in enumerate(n_values):
        heights = []
        missing_any = False
        for ds in ds_names:
            h = avg_costs_per_n.get(ds, {}).get(n_val, np.nan)
            if h is None or (isinstance(h, float) and math.isnan(h)):
                missing_any = True
                heights.append(0.0)
            else:
                heights.append(h)

        bars = plt.bar(x + i * width, heights, width=width, label=f"n={n_val}")

        # Annotate bars exceeding threshold or show values
        for bar, h in zip(bars, heights):
            if annotate_threshold is not None:
                if h > annotate_threshold:
                    plt.text(bar.get_x() + bar.get_width() / 2,
                             h + 0.5,
                             f"{h:.1f}", ha='center', va='bottom', fontsize=8, rotation=90, color='red')
            else:
                # label each bar with its value (small font)
                plt.text(bar.get_x() + bar.get_width() / 2,
                         h + 0.2,
                         f"{h:.1f}", ha='center', va='bottom', fontsize=7, rotation=90)

        if missing_any:
            print(f"Warning: some values for n={n_val} are missing or NaN (plotted as 0).")

    plt.xticks(x + width * (len(n_values) - 1) / 2, ds_names)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()



def plot_zipf_parameter_sweep(avg_costs_per_alpha: Dict[float, Dict[str, float]],
                              alpha_values: List[float],
                              ds_names: List[str],
                              title: str,
                              ylabel: str,
                              annotate_threshold: Optional[float] = None):
    
    x = np.arange(len(alpha_values))
    width = 0.8 / max(1, len(ds_names))

    plt.figure(figsize=(10, 6))

    for i, ds in enumerate(ds_names):
        heights = []
        for alpha in alpha_values:
            v = avg_costs_per_alpha.get(alpha, {}).get(ds, np.nan)
            heights.append(0.0 if (isinstance(v, float) and math.isnan(v)) else v)

        bars = plt.bar(x + i * width, heights, width=width, label=ds)

        for bar, h in zip(bars, heights):
            if annotate_threshold is not None and h > annotate_threshold:
                plt.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                         f"{h:.1f}", ha='center', va='bottom', fontsize=8, rotation=90, color='red')

    # x-axis labels should be the formatted alpha parts (1, 1.25, 1.5, 2, ...)
    alpha_labels = [_alpha_to_fname_part(a) for a in alpha_values]
    plt.xticks(x + width * (len(ds_names) - 1) / 2, alpha_labels)
    plt.xlabel("Zipf parameter α")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")

    # Test 1 (α=2, δ=0) - ROTZipfianTest
    path_rot = os.path.join(results_dir, "ROTZipfianTest")
    avg_rot = load_avg_costs(path_rot, ds_names, n_values, alpha_values=[2.0])
    # avg_rot[2.0][0.0] is mapping ds -> { n -> value }
    plot_grouped_bar({ds: {n: avg_rot[2.0][0.0].get(ds, {}).get(n, np.nan) for n in n_values} for ds in ds_names},
                     n_values,
                     "Standard Zipfian Test (α=2, δ=0)",
                     "Avg. # of Comparisons per Query",
                     ds_order=ds_names,
                     annotate_threshold=25)


    # Test 2 (α=2, δ=0.9) - ROFZipfianTest
    path_rof = os.path.join(results_dir, "ROFZipfianTest")
    avg_rof = load_avg_costs(path_rof, ds_names, n_values, alpha_values=[2.0], error_values=[0.9])
    plot_grouped_bar({ds: {n: avg_rof[2.0][0.9].get(ds, {}).get(n, np.nan) for n in n_values} for ds in ds_names},
                     n_values,
                     "Non-randomized Zipfian Test (α=2, δ=0.9)",
                     "Avg. # of Comparisons per Query",
                     ds_order=ds_names,
                     annotate_threshold=25)

    # Test 3 (α varies, n=2000, δ=0)
    alpha_sweep = [1, 1.25, 1.5, 2, 3]
    avg_alpha = load_avg_costs(path_rof, ds_names, n_values=[2000], alpha_values=alpha_sweep, error_values=[0.0])
    avg_alpha_flat = {alpha: {ds: avg_alpha[alpha][0.0].get(ds, {}).get(2000, np.nan) for ds in ds_names} for alpha in alpha_sweep}
    plot_zipf_parameter_sweep(avg_alpha_flat, alpha_sweep, ds_names,
                              "Impact of Zipf Parameter α on DS Performance (n=2000, δ=0)",
                              "Avg. # of Comparisons per Query",
                              annotate_threshold=25)


    # Test 4 (α=1.01, δ=0,0.9) - InversePowerTest
    path_ip = os.path.join(results_dir, "InversePowerTest")
    avg_ip = load_avg_costs(path_ip, ds_names, n_values, alpha_values=[1.01], error_values=[0.0, 0.9])
    
    # plot both δ=0 and δ=0.9
    plot_grouped_bar({ds: {n: avg_ip[1.01][0.0].get(ds, {}).get(n, np.nan) for n in n_values} for ds in ds_names},
                     n_values,
                     "Inverse Power Distribution Test (α=1.01, δ=0.0)",
                     "Avg. # of Comparisons per Query",
                     ds_order=ds_names,
                     annotate_threshold=25)
    plot_grouped_bar({ds: {n: avg_ip[1.01][0.9].get(ds, {}).get(n, np.nan) for n in n_values} for ds in ds_names},
                     n_values,
                     "Inverse Power Distribution Test (α=1.01, δ=0.9)",
                     "Avg. # of Comparisons per Query",
                     ds_order=ds_names,
                     annotate_threshold=25)