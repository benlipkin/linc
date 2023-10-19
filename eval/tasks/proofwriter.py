from eval.base import OWAFOLTask

_CITATION = """
@inproceedings{Tafjord2020ProofWriterGI,
  title={ProofWriter: Generating Implications, Proofs, and Abductive Statements over Natural Language},
  author={Oyvind Tafjord and Bhavana Dalvi and Peter Clark},
  booktitle={Findings},
  year={2020}
}
"""


def create_all_tasks():
    def create_task(mode, n):
        class ProofWriter(ProofWriterBase):
            def __init__(self):
                super().__init__(mode, n)

        return ProofWriter

    return {
        f"proofwriter-{mode}-{n}shot": create_task(mode, n)
        for mode in ["baseline", "scratchpad", "neurosymbolic", "cot"]
        for n in [1, 2, 4, 8, 16]
    }


class ProofWriterBase(OWAFOLTask):
    DATASET_PATH = "theoxo/proofwriter-deduction-balanced"
    DATASET_NAME = None

    def __init__(self, mode, n, seed=7):
        super().__init__(mode, n)
        self._test = self.reformat(self.dataset["test"]).shuffle(seed)


    def reformat(self, dataset):

        def punctuate(s):
            if s[-1] not in [".", "?", "!"]:
                s += "."
            return s

        def reformat_sample(sample):
            sample["premises"] = [punctuate(p) for p in sample.pop("theory").split(". ")]
            sample["conclusion"] = punctuate(sample.pop("question"))
            sample["label"] = sample.pop("answer")
            return sample

        return dataset.map(reformat_sample)