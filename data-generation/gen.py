import os
import json
import numpy as np
import io
import torch
import ray
import ray.data
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from vllm import LLM, SamplingParams
import datasets
from datasets import load_dataset
from typing import Any, Dict, List
#from mathruler.grader import grade_answer
from datasets import Dataset


TEMPLATE = """## Task Description

You will play the role of a strict fact-checker, responsible for evaluating the factual accuracy and logical consistency of AI assistant answers. Your evaluation must be thorough, objective, and uncompromising in identifying errors.

## Evaluation Criteria

1. Carefully analyze both the user's question and the AI's answer
2. Check if the answer contains any of the following issues:
   - Factual errors (incorrect information, outdated data, misattributions)
   - Logical contradictions (internal inconsistencies, flawed reasoning)
   - Technical inaccuracies (incorrect code, improper terminology)
3. If even one error is found, the entire answer must be marked as incorrect
4. Provide your judgment in a structured JSON format with clear reasoning

## Output Format

```json
{
  "Is correct": true/false,
  "Reason for error": "Detailed explanation of the error(s) found" [Only include if "Is correct" is false]
}
```

## Examples

### Example 1: Simple Factual Error

**User question:**
```What is the boiling point of water at standard atmospheric pressure? ```

**AI answer:**
```At standard atmospheric pressure (1 atm), water boils at 100 degrees Fahrenheit. ```

**Your judgment:**
```json
{
  "Is correct": false,
  "Reason for error": "The answer contains a unit error. At standard atmospheric pressure, water boils at 100 degrees Celsius (212 degrees Fahrenheit), not 100 degrees Fahrenheit."
}
```

### Example 2: Correct Technical Information

**User question:**
```How do you reverse a string in Python?```

**AI answer:**
```In Python, you can reverse a string using slice notation with a negative step: `my_string[::-1]`. For example, if `my_string = 'hello'`, then `my_string[::-1]` will return `'olleh'`. Alternatively, you can use the reversed() function and join the result: `''.join(reversed(my_string))`.```

**Your judgment:**
```json
{
  "Is correct": true
}
```

### Example 3: Logical Contradiction

**User question:**
```What are the benefits of regular exercise? ```

**AI answer:**
```Regular exercise has numerous health benefits. It reduces the risk of heart disease while simultaneously increasing your chances of cardiovascular problems. It also helps maintain a healthy weight and improves mental health by releasing endorphins. ```

**Your judgment:**
```json
{
  "Is correct": false,
  "Reason for error": "The answer contains a logical contradiction. It states that exercise reduces the risk of heart disease but also increases chances of cardiovascular problems, which directly contradict each other."
}
```

### Example 4: Incomplete But Accurate Information

**User question:**
```What is the capital of Australia? ```

**AI answer:**
```The capital of Australia is Canberra. ```

**Your judgment:**
```json
{
  "Is correct": true
}
```

### Example 5: Technical Error in Code

**User question:**
```How do I create a list comprehension in Python that gives me all even numbers from 1 to 10? ```

**AI answer:**
```You can use a list comprehension like this: `[x for x in range(1, 11) if x % 2 = 0]` ```

**Your judgment:**
```json
{
  "Is correct": false,
  "Reason for error": "The code contains a syntax error. The comparison operator should be '==' not '='. The correct code would be: `[x for x in range(1, 11) if x % 2 == 0]`"
}
```

## Important Guidelines

- Always evaluate with a rigorous and objective attitude
- Be thorough in your analysis, checking every claim made in the AI's response
- Focus on factual accuracy and logical consistency, not writing style or completeness
- Provide specific, detailed reasoning when marking an answer as incorrect
- Remember that even a single error invalidates the entire answer
- When evaluating code, check both syntax and logical correctness

Please apply these standards consistently to maintain the highest level of factual integrity in AI responses.

Now, given a user question and an AI response, make your judgement.

**User question:**
```<|user-inst|> ```

**AI answer:**
```<|ai-res|> ```
"""



def dataset_to_ray(hf_dataset):
    """Convert HuggingFace dataset to Ray dataset"""
    data = []
    for sample in hf_dataset:
        data.append({
            'question': sample['prompt'],
            'response': sample['generated_text']
        })
    return ray.data.from_items(data)

def prepare_prompts(batch):
    """Prepare prompts for the model"""
    prompt_template = (
            "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n{problem}"
            "\n\nLet's think step by step and output the final answer in JSON format.<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n"
        )

    formatted_prompt = []
    for question, answer in zip(batch['question'], batch['response']):
        prompt = TEMPLATE[:].replace("<|user-inst|>", question).replace("<|ai-res|>", answer)
        formatted_prompt.append({
            'formatted_prompt': prompt_template.format(problem=prompt)
        })

    return {
        **batch,  # 保留所有原始键
        'prompt': formatted_prompt  # 新增 prompt 字段
    }

class LLMPredictor:
    def __init__(
        self,
        model_path,
        tokenizer_path,
        prompt_key="prompt",
        generation_key="QwQresponse",
        max_tokens=10000,
        max_model_len=10000,
        num_generations=1,
        temperature=1.0,
        top_p=1.0,
        stop_tokens=None,
        tensor_parallel_size=1,
        swap_space=4,
    ):
        self.llm = LLM(
            model=model_path,
            tokenizer=tokenizer_path,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            swap_space=swap_space,
        )
        self.prompt_key = prompt_key
        self.generation_key = generation_key
        self.sampling_params = SamplingParams(
            n=num_generations,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop_tokens
        )

    def __call__(self, batch):
        prompts = []
        for item in batch[self.prompt_key]:
            # Convert bytes back to PIL Image
            prompts.append({
                "prompt": item['formatted_prompt'],
            })

        outputs = self.llm.generate(prompts, self.sampling_params)
        # generated_texts = [o.outputs[0].text for o in outputs]
        generated_texts = [[oo.text for oo in o.outputs] for o in outputs]
        return {**batch, self.generation_key: generated_texts}


def run_vllm_inference_distributed(
    ds,
    **kwargs,
):
    tensor_parallel_size = kwargs.get("tensor_parallel_size", 1)

    # Guarentee the compute resources is available
    if torch.cuda.device_count() < tensor_parallel_size:
        raise MemoryError(
            "Insufficient GPUs: tensor_parallel_size ({}) < available gpus ({})".format(
                tensor_parallel_size, torch.cuda.device_count()
            )
        )

    # Set number of instances. Each instance will use tensor_parallel_size GPUs.
    num_instances = torch.cuda.device_count() // tensor_parallel_size
    print("Launch {} instances for vllm inference.".format(num_instances))

    # For tensor_parallel_size > 1, we need to create placement groups for vLLM
    # to use. Every actor has to have its own placement group.
    def scheduling_strategy_fn():
        # One bundle per tensor parallel worker
        pg = ray.util.placement_group(
            [{"GPU": 1, "CPU": 1}] * tensor_parallel_size, strategy="STRICT_PACK"
        )
        return dict(
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                pg, placement_group_capture_child_tasks=True
            )
        )

    resources_kwarg: Dict[str, Any] = {}
    if tensor_parallel_size == 1:
        # For tensor_parallel_size == 1, we simply set num_gpus=1.
        resources_kwarg["num_gpus"] = 1
    else:
        # Otherwise, we have to set num_gpus=0 and provide
        # a function that will create a placement group for
        # each instance.
        resources_kwarg["num_gpus"] = 0
        resources_kwarg["ray_remote_args_fn"] = scheduling_strategy_fn

    batch_size = ds.count() // num_instances + 1
    print("batch_size:", batch_size)
    print("ds.count()", ds.count())
    # Apply batch inference for all input data.
    ds = ds.map_batches(
        LLMPredictor,
        # Set the concurrency to the number of LLM instances.
        concurrency=num_instances,
        # Specify the batch size for inference.
        batch_size=batch_size,
        fn_constructor_kwargs=kwargs,
        **resources_kwarg,
    )

    return ds

    
def process_single_json(json_file_path, output_dir, model_path):
    # 加载单个json文件的数据
    data = json.load(open(json_file_path, 'r'))
    
    # 转换为datasets格式
    hf_dataset = Dataset.from_list(data)
    dataset = dataset_to_ray(hf_dataset)

    # 数据准备
    dataset = dataset.map_batches(prepare_prompts, batch_size=16)

    # 分布式推理
    dataset = run_vllm_inference_distributed(
        ds=dataset,
        model_path=model_path,
        tokenizer_path=model_path,
        prompt_key="prompt",
        generation_key="QwQresponse",
        max_tokens=10000,
        max_model_len=10000,
        num_generations=1,
        temperature=0.6,
        top_p=0.95,
        stop_tokens=["<|im_start|>", "<|im_end|>", "<|endoftext|>"],
        tensor_parallel_size=4,
        swap_space=32,
    )

    # 根据输入的json文件名生成输出JSON文件路径
    base_name = os.path.basename(json_file_path)
    file_name_without_ext = os.path.splitext(base_name)[0]
    output_file_path = os.path.join(output_dir, f"{file_name_without_ext}.judge.json")

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    # 类型转换辅助函数
    def convert_numpy_types(obj):
        if isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(v) for v in obj]
        else:
            return obj

    # 收集并转换数据
    # output_data = list(dataset.collect())
    # converted_data = [convert_numpy_types(item) for item in output_data]
    output_data = dataset.take_all()  # 使用take_all()替代collect()
    converted_data = [convert_numpy_types(item) for item in output_data]

    # 写入JSON文件
    with open(output_file_path, 'w') as f:
        json.dump(converted_data, f, indent=4, ensure_ascii=False)



def main(json_files, output_dir, model_path):
    # 初始化ray
    if not ray.is_initialized():
        ray.init()

    # 遍历每个json文件，并处理
    for json_file in json_files:
        process_single_json(json_file, output_dir, model_path)


if __name__ == "__main__":
    json_files = [
        "/lustre/projects/polyullm/yggu/data-generation/data/response/prompt-po-60k.QwenCoder.s0.json"
    ]
    output_dir = "v0"
    model_path = "/lustre/projects/polyullm/models/QwQ-32B"
    main(json_files, output_dir, model_path)
