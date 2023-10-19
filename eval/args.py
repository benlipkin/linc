from dataclasses import dataclass, field
from typing import Optional, List

from eval.tasks import ALL_TASKS


@dataclass
class RunnerArguments:
    """
    Arguments for running the evaluator.
    """

    model: str = field(
        default="Salesforce/codegen-350M-mono",
        metadata={"help": "Model to evaluate, all HuggingFace models supported"},
    )
    tasks: Optional[str] = field(
        default=None,
        metadata={"help": f"Evaluation tasks from {ALL_TASKS}"},
    )
    n_samples: Optional[int] = field(
        default=1,
        metadata={"help": "Number of completions to generate for each sample."},
    )
    batch_size: int = field(
        default=1,
        metadata={"help": "Batch size for evaluation on each worker"},
    )
    limit: Optional[int] = field(
        default=None,
        metadata={"help": "Number of samples to solve and evaluate from the benchmark"},
    )
    postprocess: bool = field(
        default=True,
        metadata={"help": "Postprocess model outputs before execution"},
    )
    allow_code_execution: bool = field(
        default=False,
        metadata={"help": "Allow generated code to be executed on your machine"},
    )
    generation_only: bool = field(
        default=False,
        metadata={"help": "Do code generation but no evaluation"},
    )
    generations_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to generated solutions; skip to evaluation"},
    )
    output_dir: str = field(
        default="outputs",
        metadata={"help": "Path to save the results"},
    )
    save_generations_raw: bool = field(
        default=False,
        metadata={"help": "Whether to save raw intermediate generations"},
    )
    save_generations_prc: bool = field(
        default=False,
        metadata={"help": "Whether to save final postprocessed generations"},
    )
    save_references: bool = field(
        default=False,
        metadata={"help": "Whether to save reference solutions/tests"},
    )
    save_results: bool = field(
        default=False,
        metadata={"help": "Whether to save final metrics"},
    )
    save_generations_raw_path: Optional[str] = field(
        default="generations_raw.json",
        metadata={"help": "Path to save raw intermediate generations"},
    )
    save_generations_prc_path: Optional[str] = field(
        default="generations_prc.json",
        metadata={"help": "Path to save final postprocessed generations"},
    )
    save_references_path: Optional[str] = field(
        default="references.json",
        metadata={"help": "Path to save reference solutions/tests"},
    )
    save_results_path: Optional[str] = field(
        default="results.json",
        metadata={"help": "Path to save final metrics"},
    )
    cache_dir: Optional[str] = field(
        default=".cache",
        metadata={"help": "Path to cache directory"},
    )


@dataclass
class HFArguments:
    """
    Arguments specific to Hugging Face models.
    """

    precision: Optional[str] = field(
        default="fp32",
        metadata={"help": "Precision to use (fp32, fp16, bf16)"},
    )
    revision: Optional[str] = field(
        default=None,
        metadata={"help": "Model revision to use"},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Use the token generated when running `huggingface-cli login` (necessary for private model)."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": "Use a model with custom code, this requires executing code by the author of the model."
        },
    )


@dataclass
class OAIArguments:
    """
    Arguments for OpenAI models.
    """

    openai_api_env_keys: List[str] = field(
        default=None,
        metadata={"help": "The environment variable(s) pointing to OpenAI API key(s)"},
    )
    chat_system_instruction: Optional[str] = field(
        default="You are a helpful assistant that carefully follows instructions. "
        + "You should complete the user text, continuing from the example format, "
        + "rather than providing a conversational response.",
        metadata={"help": "System instruction to use for chat models"},
    )


@dataclass
class GenerationArguments:
    """
    Arguments for generations.
    """

    prefix: Optional[str] = field(
        default="",
        metadata={
            "help": "Prefix to add to the prompt. For example InCoder needs prefix='<| file ext=.py |>\n'"
        },
    )
    max_length_generation: int = field(
        default=1024,
        metadata={"help": "Maximum length of generated sequence (prompt+generation)"},
    )
    do_sample: Optional[bool] = field(
        default=True,
        metadata={"help": "Sample from the language model's output distribution."},
    )
    temperature: Optional[float] = field(
        default=0.2, metadata={"help": "Sampling temperature used for generation."}
    )
    top_k: Optional[int] = field(
        default=0, metadata={"help": "Top-k parameter used for generation."}
    )
    top_p: Optional[float] = field(
        default=0.95, metadata={"help": "Top-p parameter used for nucleus sampling."}
    )
    eos: Optional[str] = field(
        default="<|endoftext|>", metadata={"help": "end of sentence token."}
    )
    seed: Optional[int] = field(
        default=0, metadata={"help": "Random seed used for evaluation."}
    )
