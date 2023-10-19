import os
import itertools
import json
import pathlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from eval.tasks.folio import FOLIOBase

MODELS = ["starcoderplus", "gpt-3.5-turbo-16k-0613", "gpt-4-0613"]
TASKS = ["folio", "proofwriter"]
MODES = ["baseline", "scratchpad", "cot", "neurosymbolic"]
N = [1, 2, 4, 8]
ERROR = "Error"
METRIC = FOLIOBase.metric

def populate_scores_table(scores, model, task, mode, n, qdep=None):
    def calc_mean_std(gens, refs, n=1_000):
        def sample_acc(gens, refs):
            indices = np.random.choice(len(gens), size=len(gens), replace=True)
            gs = [gens[i] for i in indices]
            rs = [refs[i] for i in indices]
            return METRIC(gs, rs, ERROR)["accuracy (pass@1 majority)"]

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(sample_acc, gens, refs) for _ in range(n)]
            accs = [f.result() for f in futures]
        return np.mean(accs), np.std(accs)

    template = "../../outputs/%s_%s-%s-%dshot_results.json"
    score_json = template % (model, task, mode, n)
    if not pathlib.Path(score_json).exists():
        return 
    gens_json = score_json.replace("results", "generations_prc")
    refs_json = score_json.replace("results", "references")
    with open(score_json) as f:
        score = json.load(f)
    with open(gens_json) as f:
        gens = json.load(f)
    with open(refs_json) as f:
        refs = json.load(f)
    if qdep is not None:
        assert task == "proofwriter"
        with open('../../outputs/pw-qdeps.json', 'r') as f:
            qdeps = json.load(f)
        gens = [g for g, q in zip(gens, qdeps) if q == qdep]
        refs = [r for r, q in zip(refs, qdeps) if q == qdep]
        assert refs.count("True") == refs.count("False") == refs.count("Uncertain"), f"{refs.count('True')} {refs.count('False')} {refs.count('Uncertain')} at qdep {qdep}"
        raw_acc, raw_acc_std = calc_mean_std(gens, refs)
    else:
        raw_acc = score[score_json.split("_")[1]]["accuracy (pass@1 majority)"]
        m, s = calc_mean_std(gens, refs)
        assert np.isclose(raw_acc, m, atol=1e-2), f'{raw_acc} != {m}'  # @ben - is this sanity check correct?
        raw_acc_std = s

    parse_errs = sum(all(g_i == "Error" for g_i in g) for g in gens)
    scores["model"].append(model)
    scores["task"].append(task)
    scores["mode"].append(mode)
    scores["nshot"].append(n)
    scores["accuracy_raw"].append(raw_acc)
    scores["accuracy_raw_std"].append(raw_acc_std)
    scores["accuracy_raw_ci"].append(3.291 * raw_acc_std / np.sqrt(len(gens)))
    scores["parse_errors"].append(parse_errs)

def make_core_table():

    scores = defaultdict(list)
    for model, task, mode, n in itertools.product(MODELS, TASKS, MODES, N):
        populate_scores_table(scores, model, task, mode, n)
    pd.DataFrame(scores).to_csv(f"../tables/core_results.csv", index=False)
    print(f"Saved {os.path.abspath('../tables/core_results.csv')}")


def make_qdep_tables():
    task = "proofwriter"
    n = 8
    for model in MODELS:
        for qdep in range(0, 6):
            scores = defaultdict(list)
            for mode in MODES:
                populate_scores_table(scores, model, task, mode, n, qdep)
            pd.DataFrame(scores).to_csv(f"../tables/{model}_pw_qdep_{qdep}_results.csv", index=False)
            print(f"Saved {os.path.abspath(f'../tables/{model}_pw_qdep_{qdep}_results.csv')}")


def make_kmaj_table():
    def calc_kmaj_acc(gens, refs, k, n=1_000):
        def sample_kmaj_acc(gens, refs, k):
            indices = np.random.choice(len(gens[0]), size=k, replace=True)
            gs = [[g[i] for i in indices] for g in gens]
            return METRIC(gs, refs, ERROR)["accuracy (pass@1 majority)"]

        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(sample_kmaj_acc, gens, refs, k) for _ in range(n)
            ]
            accs = [f.result() for f in futures]
        return np.mean(accs), np.std(accs)

    scores = defaultdict(list)
    template = "../../outputs/%s_%s-%s-%dshot_generations_prc.json"
    for model, task, mode, n in itertools.product(MODELS, TASKS, MODES, N):
        gens_json = template % (model, task, mode, n)
        if not pathlib.Path(gens_json).exists():
            continue
        refs_json = gens_json.replace("generations_prc", "references")
        with open(gens_json) as f:
            gens = json.load(f)
        with open(refs_json) as f:
            refs = json.load(f)
        for k in range(1, len(gens[0]) + 1):
            acc, std = calc_kmaj_acc(gens, refs, k)
            scores["model"].append(model)
            scores["task"].append(task)
            scores["mode"].append(mode)
            scores["nshot"].append(n)
            scores["kmaj"].append(k)
            scores["accuracy_kmaj"].append(acc)
            scores["accuracy_kmaj_std"].append(std)
            scores["accuracy_kmaj_ci"].append(3.291 * std / np.sqrt(len(gens[0])))
    pd.DataFrame(scores).to_csv(f"../tables/kmaj_results.csv", index=False)
    print(f"Saved {os.path.abspath('../tables/kmaj_results.csv')}")


def main():
    make_core_table()
    make_kmaj_table()
    make_qdep_tables()


if __name__ == "__main__":
    main()
