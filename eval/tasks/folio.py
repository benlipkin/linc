"""
FOLIO: Natural Language Reasoning with First-Order Logic
https://arxiv.org/pdf/2209.00840.pdf
"""
from eval.base import OWAFOLTask
from eval.tasks.utils import evaluate, convert_to_nltk_rep

_CITATION = """
@article{han2022folio,
  title={Folio: Natural language reasoning with first-order logic},
  author={Han, Simeng and Schoelkopf, Hailey and Zhao, Yilun and Qi, Zhenting and Riddell, Martin and Benson, Luke and Sun, Lucy and Zubova, Ekaterina and Qiao, Yujie and Burtell, Matthew and others},
  journal={arXiv preprint arXiv:2209.00840},
  year={2022}
}
"""


def create_all_tasks():
    def create_task(mode, n):
        class FOLIO(FOLIOBase):
            def __init__(self):
                super().__init__(mode, n)

        return FOLIO

    return {
        f"folio-{mode}-{n}shot": create_task(mode, n)
        for mode in ["baseline", "scratchpad", "neurosymbolic", "cot"]
        for n in [1, 2, 4, 8, 16]
    }


class FOLIOBase(OWAFOLTask):
    DATASET_PATH = "benlipkin/folio"
    DATASET_NAME = None

    def __init__(self, mode, n, seed=7):
        super().__init__(mode, n)
        # process validation dataset
        self._dataset = self.reformat_fol_samples(self.dataset["validation"]).shuffle(seed)
        self._test = self._dataset.select(range(0, len(self._dataset)))

    def reformat_fol_samples(self, dataset):
        def reformat_fol_sample(sample):
            sample["premises-FOL"] = [
                convert_to_nltk_rep(premise) for premise in sample["premises-FOL"]
            ]
            sample["conclusion-FOL"] = convert_to_nltk_rep(sample["conclusion-FOL"])
            try:
                assert len(sample["premises"]) == len(sample["premises-FOL"])
                label = evaluate(sample["premises-FOL"], sample["conclusion-FOL"])
                assert sample["label"] == label
            except Exception as e:
                # print(f"Error in parsing FOL: {e}")
                # print(sample)
                sample["label"] = self.ERROR_TOKEN
            return sample

        return dataset.map(reformat_fol_sample).filter(
            lambda x: x["label"] != self.ERROR_TOKEN
        )