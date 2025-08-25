import json, os, glob, re
import numpy as np
import matplotlib.pyplot as plt

ds_names = ["StaticRSL", "BiasedZipZipTree", "ThresholdZipZipTree", "Treap", "AVL"]
n_values = [1000, 2000, 5000, 10000]


"""
Detect all available adversarial error values for a given dataset configuration.

Parameters:
- path_dir : str
    Directory where JSON result files are stored.
- ds : str
    Data structure name (e.g., "StaticRSL", "Treap").
- n : int
    Problem size (number of elements).
- alpha : float
    Zipfian parameter.

Returns:
- list[float] : sorted list of error values (as fractions, e.g., 0.0, 0.45, 0.9)that were found in the filenames.
"""
def detect_error_values(path_dir, ds, n, alpha):
    pattern = f"{ds}_n{n}_e*_a{alpha}.json"
    files = glob.glob(os.path.join(path_dir, pattern))
    errors = []
    
    for f in files:
        m = re.search(r"_e(\d+)_a", os.path.basename(f))
        if m:
            errors.append(int(m.group(1)) / 100)
    
    return sorted(errors)



"""
Load average search costs for multiple datasets across DS, n, and error values.

The function scans result JSON files, loads recorded search costs, and computes
per-query averages. Results are organized in a nested dictionary.

Parameters:
- path_dir : str
    Directory where JSON result files are stored.
- ds_names : list[str]
    Names of data structures to include.
- n_values : list[int]
    List of dataset sizes (n values) to load.
- alpha_values : list[float] (default=[1])
    Zipf parameters to evaluate.
- error_values : list[float] or None (default=None)
    Error (δ) values to evaluate. If None, values are automatically detected
    from filenames.

Returns:
- dict : nested dictionary of average costs
    Structure: avg_costs[alpha][error][ds] = list of averages for n_values
"""
def load_avg_costs(path_dir, ds_names, n_values, alpha_values=[1], error_values=None):
    avg_costs = {}

    for alpha in alpha_values:
        avg_costs[alpha] = {}
        for ds in ds_names:
            for n in n_values:
                # Detect error_Values if None

                evs = error_values
                if evs is None:
                    evs = detect_error_values(path_dir, ds, n, alpha)
                    if not evs:
                        evs = [None]
                
                for error in evs:
                    if alpha not in avg_costs:
                        avg_costs[alpha] = {}
                    if error not in avg_costs[alpha]:
                        avg_costs[alpha][error] = {ds: [] for ds in ds_names}
                    
                    if error is None:
                        filename = f"{path_dir}/{ds}_n{n}_a{alpha}.json"
                    else:
                        filename = f"{path_dir}/{ds}_n{n}_e{int(error*100)}_a{alpha}.json"
                    
                    if os.path.exists(filename):
                        # Adjusted to handle both costs and size in results
                        # results = {"costs": [...], "size": ...}
                        with open(filename) as f:
                            data = json.load(f)
                        costs = data.get("costs", [])
                        avg_costs[alpha][error][ds].append(np.mean(costs))
                    else:
                        avg_costs[alpha][error][ds].append(np.nan)

    return avg_costs

"""
Load all JSON files and collect sizes per DS per n.
Returns nested dictionary: sizes[alpha][error][ds] = list of sizes for n_values
"""
def load_sizes(path_dir, ds_names, n_values, alpha_values=[1], error_values=None):
    sizes = {}

    for alpha in alpha_values:
        sizes[alpha] = {}
        for ds in ds_names:
            for n in n_values:
                evs = error_values
                if evs is None:
                    evs = detect_error_values(path_dir, ds, n, alpha)
                    if not evs:
                        evs = [None]

                for error in evs:
                    if alpha not in sizes:
                        sizes[alpha] = {}
                    if error not in sizes[alpha]:
                        sizes[alpha][error] = {ds: [] for ds in ds_names}

                    if error is None:
                        filename = f"{path_dir}/{ds}_n{n}_a{alpha}.json"
                    else:
                        filename = f"{path_dir}/{ds}_n{n}_e{int(error*100)}_a{alpha}.json"

                    if os.path.exists(filename):
                        with open(filename) as f:
                            data = json.load(f)
                        sizes[alpha][error][ds].append(data.get("size", np.nan))
                    else:
                        sizes[alpha][error][ds].append(np.nan)
    return sizes

def plot_grouped_size(sizes_per_n, n_values, title, ylabel):
    ds_names = list(sizes_per_n.keys())
    x = np.arange(len(ds_names))
    width = 0.2

    plt.figure(figsize=(10,6))
    bars_list = []

    for i, n_val in enumerate(n_values):
        heights = [sizes_per_n[ds][i] for ds in ds_names]
        bars = plt.bar(x + i*width, heights, width=width, label=f"n={n_val}")
        bars_list.append(bars)

        # Annotate bars with actual size values
        for bar, h in zip(bars, heights):
            plt.text(bar.get_x() + bar.get_width()/2, h + 15,
                     f"{int(h)}", ha='center', va='bottom', fontsize=10, rotation=0, color='black')

    plt.xticks(x + width*(len(n_values)-1)/2, ds_names)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(title="Input size n")
    plt.tight_layout()
    plt.show()




"""
Plot a grouped bar chart comparing average search costs across data structures.

Parameters:
- avg_costs_per_n : dict
    Dictionary mapping DS name → list of average costs (aligned with n_values).
- n_values : list[int]
    Dataset sizes (n values) represented in the grouped bars.
- title : str
    Title of the plot.
- ylabel : str
    Y-axis label (e.g., "Avg. # of Comparisons").
- annotate_threshold : float or None (default=None)
    If set, annotate bars that exceed this threshold with their numeric value.

Side effects:
- Displays a matplotlib grouped bar chart.
"""
def plot_grouped_bar(avg_costs_per_n, n_values, title, ylabel, annotate_threshold=None):
    ds_names = list(avg_costs_per_n.keys())
    x = np.arange(len(ds_names))
    width = 0.2

    plt.figure(figsize=(10,6))
    for i, n_val in enumerate(n_values):
        heights = [avg_costs_per_n[ds][i] for ds in ds_names]
        bars = plt.bar(x + i*width, heights, width=width, label=f"n={n_val}")

        # Annotate bars exceeding threshold
        if annotate_threshold is not None:
            for bar, h in zip(bars, heights):
                if h > annotate_threshold:
                    plt.text(bar.get_x() + bar.get_width()/2, annotate_threshold + 0.5, 
                             f"{h:.1f}", ha='center', va='bottom', fontsize=9, rotation=90, color='black')

    plt.xticks(x + width*(len(n_values)-1)/2, ds_names)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()



"""
Plot a bar chart showing the effect of varying the Zipf parameter (α).

Parameters:
- avg_costs_per_alpha : dict
    Nested dictionary mapping alpha → { ds : avg_cost } for each DS.
- alpha_values : list[float]
    Zipf parameters (x-axis values).
- ds_names : list[str]
    Names of data structures to include as bar groups.
- title : str
    Title of the plot.
- ylabel : str
    Y-axis label.
- annotate_threshold : float or None (default=None)
    If set, annotate bars that exceed this threshold with their numeric value.

Side effects:
- Displays a matplotlib grouped bar chart.
"""
def plot_zipf_parameter_sweep(avg_costs_per_alpha, alpha_values, ds_names, title, ylabel, annotate_threshold=None):
    x = np.arange(len(alpha_values))
    width = 0.15

    plt.figure(figsize=(10,6))
    for i, ds in enumerate(ds_names):
        heights = [avg_costs_per_alpha[alpha][ds] for alpha in alpha_values]  # pick n=2000 or first n
        bars = plt.bar(x + i*width, heights, width=width, label=ds)

        # Annotate bars exceeding threshold
        if annotate_threshold is not None:
            for bar, h in zip(bars, heights):
                if h > annotate_threshold:
                    plt.text(bar.get_x() + bar.get_width()/2, annotate_threshold + 0.5, 
                             f"{h:.1f}", ha='center', va='bottom', fontsize=8, rotation=90, color='red')

    plt.xticks(x + width*(len(ds_names)-1)/2, alpha_values)
    plt.xlabel("Zipf parameter α")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    
    # Standard Zipfian Test (δ=0)
    path_std = "results/StandardZipfianTest"
    avg_std = load_avg_costs(path_std, ds_names, n_values)
    plot_grouped_bar(avg_std[1][None], n_values, "Standard Zipfian Test (α=1)", "Avg. # of Comparisons per Query", annotate_threshold=25)

    # Adversarial Zipfian Test (pick δ=0.9 for plotting)
    path_adv = "results/AdvZipfianTest"
    avg_adv = load_avg_costs(path_adv, ds_names, n_values, alpha_values=[1], error_values=[0.9])
    plot_grouped_bar(avg_adv[1][0.9], n_values, "Adversarial Zipfian Test (δ=0.9, α=1)", "Avg. # of Comparisons per Query", annotate_threshold=25)

    # Random Zipfian Test (pick δ=0.9 for plotting)
    path_adv = "results/RandomZipfianTest"
    avg_adv = load_avg_costs(path_adv, ds_names, n_values, alpha_values=[1], error_values=[0.9])
    plot_grouped_bar(avg_adv[1][0.9], n_values, "Random Zipfian Test (δ=0.9, α=1)", "Avg. # of Comparisons per Query", annotate_threshold=25)

    # Zipf parameter sweep (α varies, n=2000, δ=0.9)
    alpha_sweep = [1, 1.25, 1.5, 2, 3]
    avg_alpha = load_avg_costs(path_adv, ds_names, n_values=[2000], alpha_values=alpha_sweep, error_values=[0.9])
    avg_alpha_flat = {alpha: {ds: avg_alpha[alpha][0.9][ds][0] for ds in ds_names} for alpha in alpha_sweep}
    plot_zipf_parameter_sweep(avg_alpha_flat, alpha_sweep, ds_names, "Impact of Zipf Parameter α on DS Performance", "Avg. # of Comparisons per Query", annotate_threshold=25)

    # -----------------------------
    # Plot sizes for Standard Zipfian Test

    # Standard Zipfian Test - SIZE
    sizes_std = load_sizes(path_std, ds_names, n_values)
    plot_grouped_size(sizes_std[1][None], n_values, "Standard Zipfian Test Sizes (α=1)", "Number of Nodes")

    # Adversarial Zipfian Test - SIZE
    sizes_adv = load_sizes(path_adv, ds_names, n_values, alpha_values=[1], error_values=[0.9])
    plot_grouped_size(sizes_adv[1][0.9], n_values, "Adversarial Zipfian Test Sizes (δ=0.9, α=1)", "Number of Nodes")

    # Random Zipfian Test - SIZE
    sizes_rand = load_sizes("results/RandomZipfianTest", ds_names, n_values, alpha_values=[1], error_values=[0.9])
    plot_grouped_size(sizes_rand[1][0.9], n_values, "Random Zipfian Test Sizes (δ=0.9, α=1)", "Number of Nodes")
