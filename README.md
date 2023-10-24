# LINC: Logical Inference via Neurosymbolic Computation

Repository for the paper [LINC: A neuro-symbolic approach for logical reasoning by combining language models with first-order logic provers](https://arxiv.org/abs/2310.15164) by `Theo X. Olausson*, Alex Gu*, Ben Lipkin*, Cedegao E. Zhang*, Armando Solar-Lezama, Joshua B. Tenenbaum, & Roger P. Levy`, to be presented at EMNLP 2023.

Code is provided to reproduce all experiments and figures.

## Setup

Requirements: [Anaconda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html), [Make](https://www.gnu.org/software/make/manual/make.html), [Prover9](https://formulae.brew.sh/formula/prover9)

```bash
make setup
```

## Usage

To rerun our exact experiments:
```bash
nano SUBMIT.sh # cfg for own cluster and submit contents of $JOB env variable
make run
```

To run custom experiments within our framework:
```bash
accelerate launch runner.py **kwargs
# see `eval/args.py` for options.
```

To replicate figures and tables from provided outputs:
```bash
touch outputs/run.done # don't rerun analyses
make analyze # core tables and figures
# see `analysis/notebooks` for supplemental tables and figures
```

## Acknowledgements
This evaluation framework structure is adapted from the [BigCode](https://github.com/bigcode-project/bigcode-evaluation-harness) project and [EleutherAI](https://github.com/EleutherAI/lm-evaluation-harness) whom we thank for their contributions to open source.

## Citation

```bibtex
@inproceedings{OGLZ_LINC_2023,
	author={Theo X. Olausson* and Alex Gu* and Ben Lipkin* and Cedegao E. Zhang* and Armando Solar-Lezama and Joshua B. Tenenbaum and Roger P. Levy},
	title={LINC: A neuro-symbolic approach for logical reasoning by combining language models with first-order logic provers},
	year={2023},
	journal={Proceedings of the Conference on Empirical Methods in Natural Language Processing},
}
```
