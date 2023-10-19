import json
import numpy as np
import os
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler
from compile_results import MODELS, MODES, TASKS

plt.style.use("csail-like")  # github.com/theoxo/csail-like-colormap
#plt.style.use("seaborn-v0_8-paper")

#cycle = cycler(color=["tab:red", "tab:blue", "tab:orange", "tab:green"])
#plt.rc("axes", prop_cycle=cycle)

mpl.rc("font", family="serif", size=28)


name_map = {
    "baseline": "Naive",
    "cot": "Chain-of-Thought",
    "neurosymbolic": "LINC (ours)",
    "scratchpad": "Scratchpad",
    "starcoderplus": "StarCoder+",
    "gpt-3.5-turbo-16k-0613": "GPT-3.5",
    "gpt-4-0613": "GPT-4",
}

def make_figure_kmaj_curve(table):
    #colors=["tab:orange", "tab:blue", "tab:green"]
    table = table.loc[table["nshot"] == 8]  # only plot 8-shot
    for task in ["folio"]:
        _, ax = plt.subplots(figsize=(12, 8))
        color_cycle = ax._get_lines.prop_cycler
        colors = []
        for _ in range(5):
            colors.append(next(color_cycle)["color"])
        print(colors)
        #colors = colors[1:]  # skip first color
        #ax.set_prop_cycle(color=next(color_cycler))
        samples = table.loc[table["task"] == task]
        for i, model in enumerate(MODELS):
            model_samples = samples.loc[samples["model"] == model]
            for mode in ["cot", "neurosymbolic"]:
                mode_samples = model_samples.loc[samples["mode"] == mode]
                ax.plot(
                    mode_samples["kmaj"],
                    mode_samples["accuracy_kmaj"],
                    #yerr=mode_samples["accuracy_kmaj_std"],
                    label=f"{name_map[model]}-{name_map[mode]}",
                    marker="o" if mode == "neurosymbolic" else "x",
                    linestyle="-" if mode == "neurosymbolic" else "--",
                    color=colors[i],
                    zorder=3,
                    #**kwargs,
                )
                ax.fill_between(
                    mode_samples["kmaj"],
                    mode_samples["accuracy_kmaj"] - mode_samples["accuracy_kmaj_std"],
                    mode_samples["accuracy_kmaj"] + mode_samples["accuracy_kmaj_std"],
                    alpha=0.2,
                    zorder=1,
                    color="#898D8D" if mode == "cot" else colors[i],
                )
        #[spine.set_visible(False) for spine in plt.gca().spines.values()]
        xticks = sorted(table.kmaj.unique())
        yticks = [0.167, 0.333, 0.5, 0.667]
        chance = 65/(65+63+54) if task == "folio" else 1/3
        plt.axhline(chance, linestyle="--", color="black", label="Chance", zorder=2)
        plt.xlabel("K-way majority vote")
        plt.xticks(xticks, [str(n) for n in xticks])
        plt.xlim(min(xticks) - 0.5, max(xticks) + 0.5)
        plt.ylabel("Accuracy on held-out test set")
        #plt.yticks(yticks, [f"{f:.3f}" for f in yticks])
        #plt.ylim(0.15, 0.70)
        plt.ylim(0.0, 1.0)
        plt.legend(loc="best", fontsize=16, ncols=2, bbox_to_anchor=(1., 1)) #"upper center", bbox_to_anchor=(0.5, 1.5))
        #plt.gcf().set_size_inches(12, 8)
        plt.savefig(f"../figures/{task}_kmaj.pdf", bbox_inches="tight")#, dpi=600)
        print(f"Saved {os.path.abspath(f'../figures/{task}_kmaj.pdf')}")
        #plt.close()

def make_main_bar_chart(table, output_file_name, bar_width=0.2, shift_text=False, chance=None):

    models = MODELS

    # Set the x positions for the bars
    xs = np.arange(len(models))

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(12, 8))
    #ax.grid(zorder=0)

    # Iterate over the models and create the bars for each condition
    for i, model in enumerate(models):
        # Calculate the x positions for the bars of each model
        x_model  = xs[i] 
        model_samples = table.loc[table["model"] == model]
        ax.set_prop_cycle(None)  # reset the color cycle
        
        for j, mode, hatch in zip(range(len(MODES)), MODES, ["/", "x", "\\", ""]):
            # Calculate the x positions for the bars of each mode within the category
            x_mode = x_model + (j * bar_width) - (bar_width * (len(MODES) - 1) / 2)

            mode_samples = model_samples.loc[model_samples["mode"] == mode]
            mode_samples = mode_samples.loc[mode_samples["nshot"] == 8]
            assert len(mode_samples) == 1, f"Expected 1 sample, got {len(mode_samples)}"
            accuracy = mode_samples["accuracy_raw"].values[0]
            std = mode_samples["accuracy_raw_std"].values[0]

            ax.bar(x_mode, accuracy, yerr=std, width=bar_width, zorder=3, hatch=hatch, edgecolor="black", linewidth=1, capsize=5,)
            # add exact value at the top of each bar
            if not shift_text or mode != "neurosymbolic":
                ax.text(x_mode, accuracy + 0.05, f"{accuracy*100:.1f}%", ha="center", va="bottom", fontsize=12)
            elif model == "gpt-4-0613":
                # put text to left of bar instead of on top
                ax.text(x_mode - 2*bar_width/3, accuracy, f"{accuracy*100:.1f}%", ha="right", va="top", fontsize=12)
            else:
                print(f'Putting text to the right for {model=}')
                # put text to right of bar instead of on top
                ax.text(x_mode + 2*bar_width/3, accuracy, f"{accuracy*100:.1f}%", ha="left", va="top", fontsize=12)

            
    # add legend labels for each mode
    leg = ax.legend([name_map[m] for m in MODES], loc="upper left", fontsize=16, ncols=2)
    leg.set_zorder(1)
    

    # add chance line (after legend to not mess it up)
    if chance is not None:
        ax.axhline(chance, linestyle="--", color="black", zorder=2)

    # Set the x-axis labels and tick positions
    ax.set_xticks(xs)
    ax.set_xticklabels([name_map[m] for m in models])
    ax.set_ylim(0., 1.)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"])

    # Add a legend
    #ax.legend()
    plt.savefig(output_file_name, bbox_inches="tight", dpi=600)
    print(f"Saved {os.path.abspath(output_file_name)}")
    plt.close()

def make_qdep_line_plots():
    # get the "chance" performance (majority guess) for each qdep
    # (labels are balanced across all samples, but not necessarily within a given qdep)

    mpl.rc("font", family="serif", size=32)

    for model in MODELS:
        chances_per_qdep = {qdep: {} for qdep in range(0, 6)}
        with open(f'../../outputs/{model}_proofwriter-baseline-8shot_references.json', 'r') as f:
            refs = json.load(f)
        with open('../../outputs/pw-qdeps.json', 'r') as f:
            qdeps = json.load(f)
        for qdep in range(0, 6):
            refs_qdep = [r for r, q in zip(refs, qdeps) if q == qdep]
            # get the most common ref
            for label in ["True", "False", "Uncertain"]:
                chances_per_qdep[qdep][label] = refs_qdep.count(label) / len(refs_qdep)
        assert all([all([np.isclose(chances_per_qdep[qdep][label], 1/3) for label in ["True", "False", "Uncertain"]]) for qdep in range(0, 6)]), f'{chances_per_qdep}'
        data = {mode: [] for mode in MODES} 
        template = "../tables/%s_pw_qdep_%d_results.csv"
        for qdep in range(0, 6):
            _d = pd.read_csv(template % (model, qdep))
            for mode in MODES:
                data[mode].append((_d.loc[_d["mode"] == mode]["accuracy_raw"].values[0], _d.loc[_d["mode"] == mode]["accuracy_raw_std"].values[0]))
        
        _, ax = plt.subplots(figsize=(9, 6))
        # plot chance line
        ax.plot(range(0, 6), [1/3 for _ in range(0, 6)], label="Chance", linestyle="--", color='black', zorder=2)
        for mode, marker in zip(MODES, ["o", "x", "+", "^"]):
            ax.plot(range(0, 6), [d[0] for d in data[mode]], label=name_map[mode], marker=marker, zorder=3)
            ax.fill_between(range(0, 6), [d[0] - d[1] for d in data[mode]], [d[0] + d[1] for d in data[mode]], alpha=0.2, zorder=3)
        leg = ax.legend(loc='lower left', ncols=3, fontsize=15)
        leg.set_zorder(1)
        ax.set_xticks(range(0, 6))
        ax.set_xticklabels(range(0, 6))
        ax.set_xlabel("Proof Depth")
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Accuracy")
        plt.savefig(f"../figures/{model}_qdep_line_plot.pdf", bbox_inches="tight")#, dpi=600)
        print(f"Saved {os.path.abspath(f'../figures/{model}_qdep_line_plot.pdf')}")
        plt.close()

def main():
    data = pd.read_csv(f"../tables/kmaj_results.csv")
    make_figure_kmaj_curve(data)
    data = pd.read_csv("../tables/core_results.csv")
    for task in TASKS:
        task_data = data.loc[data["task"] == task]
        chance = 65/(65+63+54) if task == "folio" else 1/3
        make_main_bar_chart(task_data, f"../figures/{task}_bar_chart.pdf", shift_text=task=='proofwriter', chance=chance)
    make_qdep_line_plots()


if __name__ == "__main__":
    main()
