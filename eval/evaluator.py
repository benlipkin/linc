from concurrent.futures import ThreadPoolExecutor
import hashlib
import time
import random
import json
import os
import warnings
from abc import abstractmethod, ABC
from warnings import warn

import openai
from diskcache import Cache

from eval import tasks
from eval.generation import parallel_generations

_WARNING = """
################################################################################
                                  !!!WARNING!!!
################################################################################
The task you are about to use executes untrusted model-generated code.
Although it is highly unlikely that model-generated code will do something
overtly malicious in response to this test suite, model-generated code may act
destructively due to a lack of model capability or alignment.
Users are strongly encouraged to sandbox this evaluation suite so that it
does not perform destructive actions on their host or network. For more
information on how OpenAI sandboxes its code, see the paper "Evaluating Large
Language Models Trained on Code" (https://arxiv.org/abs/2107.03374).
Once you have read this disclaimer and taken appropriate precautions, set the argument 
"allow_code_execution" to True.
################################################################################\
"""


class Evaluator(ABC):
    def __init__(self, args):
        self.args = args
        self.allow_code_execution = args.allow_code_execution

    @abstractmethod
    def generate_text(self, task_name):
        pass

    def evaluate(self, task_name):
        task = tasks.get_task(task_name)
        if task.requires_execution and not self.allow_code_execution:
            raise ValueError(_WARNING)

        generations_prc, generations_raw, references = self.generate_text(task_name)
        if len(generations_prc[0]) != self.args.n_samples:
            generations_prc = [l[: self.args.n_samples] for l in generations_prc]
            warnings.warn(
                "Number of tasks wasn't proportional to number of devices, we removed extra predictions"
            )

        if not hasattr(self, "accelerator") or self.accelerator.is_main_process:
            if not self.args.generations_path:
                if self.args.save_generations_raw:
                    with open(self.args.save_generations_raw_path, "w") as fp:
                        json.dump(generations_raw, fp)
                        print("raw generations were saved")
                if self.args.save_generations_prc:
                    with open(self.args.save_generations_prc_path, "w") as fp:
                        json.dump(generations_prc, fp)
                        print("processed generations were saved")
                if self.args.save_references:
                    with open(self.args.save_references_path, "w") as fp:
                        json.dump(references, fp)
                        print("references were saved")

            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            if self.allow_code_execution and task.requires_execution:
                os.environ["HF_ALLOW_CODE_EVAL"] = "1"
            results = task.process_results(generations_prc, references)
            return results


class HFEvaluator(Evaluator):
    def __init__(self, accelerator, model, tokenizer, args):
        super().__init__(args)
        self.accelerator = accelerator
        self.model = model
        self.tokenizer = tokenizer

    def generate_text(self, task_name):
        task = tasks.get_task(task_name)
        dataset = task.get_dataset()
        n_tasks = self.args.limit if self.args.limit else len(dataset)
        generations_prc, generations_raw = parallel_generations(
            task,
            dataset,
            self.accelerator,
            self.model,
            self.tokenizer,
            n_tasks=n_tasks,
            args=self.args,
        )
        references = [task.get_reference(dataset[i]) for i in range(n_tasks)]
        return generations_prc, generations_raw, references


class OAIEvaluator(Evaluator):
    def __init__(self, args, chat=False):
        super().__init__(args)
        self.chat = chat
        self.model = args.model
        self.api_keys = [os.environ[key] for key in args.openai_api_env_keys]
        self.cache = Cache(args.cache_dir)
        assert (
            len(self.api_keys) >= 1
        ), "You must provide at least one OpenAI API key to use OAIEvaluator"

    def generate_text(self, task_name):
        task = tasks.get_task(task_name)
        dataset = task.get_dataset()
        n_tasks = self.args.limit if self.args.limit else len(dataset)
        prompts = [task.get_prompt(dataset[i]) for i in range(n_tasks)]
        stops = [task.stop_words for _ in range(n_tasks)]

        with ThreadPoolExecutor() as executor:
            res = executor.map(self.get_completion, prompts, stops)
        generations_raw = list(res)
        if self.args.postprocess:
            generations_prc = [
                [
                    task.postprocess_generation(
                        generations_raw[i][j], i, completion_only=True
                    )
                    for j in range(self.args.n_samples)
                ]
                for i in range(n_tasks)
            ]
        else:
            generations_prc = generations_raw
        references = [task.get_reference(dataset[i]) for i in range(n_tasks)]
        return generations_prc, generations_raw, references

    def make_request(self, prompt, stop):
        if self.chat:
            response = openai.ChatCompletion.create(
                model=self.model,
                n=self.args.n_samples,
                messages=[
                    {"role": "system", "content": self.args.chat_system_instruction},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.args.temperature,
                max_tokens=self.args.max_length_generation,
                top_p=self.args.top_p,
                stop=stop,
                stream=False,
            )
        else:
            response = openai.Completion.create(
                engine=self.model,
                n=self.args.n_samples,
                prompt=prompt,
                temperature=self.args.temperature,
                max_tokens=self.args.max_length_generation,
                top_p=self.args.top_p,
                stop=stop,
                stream=False,
            )
        return response

    def get_completion(self, prompt, stop, api_key=None, exhausted={}, retry_after=60):
        if self.args.temperature == 0:
            request_id = "_".join(
                str(x)
                for x in [
                    self.model,
                    self.args.n_samples,
                    prompt,
                    self.args.max_length_generation,
                    stop,
                ]
            )
            request_key = hashlib.sha256(request_id.encode("utf-8")).hexdigest()
            if request_key in self.cache:
                print(
                    "Identical OpenAI API request previously executed. Loading response from cache."
                )
                return self.cache[request_key]
        if api_key is None:
            api_key = random.choice(self.api_keys)
        openai.api_key = api_key
        try:
            response = self.make_request(prompt, stop)
        except openai.error.RateLimitError:
            if len(self.api_keys) == 1:
                warn(
                    f"Only one API key was provided, and it has been rate limited. sleeping for {retry_after}s. Please provide more API keys to avoid sleeping."
                )
                time.sleep(retry_after)
                return self.get_completion(prompt, stop, api_key)
            else:
                print(f"Rate limit error; trying again with a different API key.")
                exhausted[api_key] = time.time()
                exhausted = {
                    k: v
                    for k, v in exhausted.items()
                    if (time.time() - v) < retry_after
                }
                if len(exhausted) == len(self.api_keys):
                    print(
                        f"All API keys have been exhausted. sleeping for {retry_after}s then trying again with all keys."
                    )
                    time.sleep(retry_after)
                    exhausted = {}
                try_next = random.choice(
                    [k for k in self.api_keys if k != api_key and k not in exhausted]
                )
                return self.get_completion(prompt, stop, try_next, exhausted)
        except (openai.error.Timeout, openai.error.APIError, openai.error.ServiceUnavailableError) as e:
            print(f"API Error; sleeping for {retry_after}s then trying again.")
            time.sleep(retry_after)
            return self.get_completion(prompt, stop, api_key, exhausted)
        if self.chat:
            response = [c["message"]["content"] for c in response["choices"]]
        else:
            response = [c["text"] for c in response["choices"]]
        if self.args.temperature == 0:
            print("Temperature is 0, caching OpenAI API response for future use.")
            self.cache[request_key] = response
        return response
