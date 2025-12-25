import os
os.environ['HF_ENDPOINT'] = ""
os.environ['HF_HOME'] = ''
os.environ['HF_HUB_CACHE'] = ""
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
import asyncio
from pathlib import Path
import sys
import argparse
from typing import List, Dict, Any, Tuple, Optional, DefaultDict, Union
import uuid
import os
import numpy as np
import json
import re
import torch
import aiohttp
from transformers import AutoTokenizer
from tqdm import tqdm
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential)
from verl_hamster.Hamster_Generation_Manager import Hamster_LLM_Generation_Manager
from verl_hamster.Hamster_Generation_Manager import Hamster_Generation_Config, Hamster_Graph_Config
from verl_hamster.Hamster_Generation_Manager import Wrapped_VLLM_Client, Wrapped_VLLM_Config
from verl_hamster.verl.utils.hamster_utils.ece import compute
from verl_hamster.verl import DataProto
from tensordict import TensorDict
from hamster_tool.schema import Message
from hamster_tool.llm import LLM, LLMSettings
from vllm import LLM as VLLM_LLM
from vllm import SamplingParams
from eval.step_judge_prompt import STEP_JUDGE_PROMPT, DEDUCE_CONFIDENCE_PROMPT
from eval.compute import stepwise_judge_ece, overconfidence_rate

def _infer_tp_size(default_tp: int = 1) -> int:
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd:
        try:
            return max(1, len([x for x in cvd.split(",") if x.strip() != ""]))
        except Exception:
            return default_tp
    try:
        import subprocess
        out = subprocess.check_output(["nvidia-smi", "-L"]).decode("utf-8").strip()
        gpus = [l for l in out.splitlines() if l.strip().startswith("GPU ")]
        return max(1, len(gpus))
    except Exception:
        return default_tp
    
def in_debug_mode() -> bool:
    return sys.gettrace() is not None

class Here_LLM():
    def __init__(self, model_name : str, end_point : str, api_key : str):
        self.model_name = model_name
        self.end_point = end_point
        self.api_key = api_key
    
    async def ask(self, message: Union[str, List[str]]) -> Union[str, List[str]]:
        is_batch = isinstance(message, list)
        messages = message if is_batch else [message]
        
        answers = []
        async with aiohttp.ClientSession() as session:
            tasks = [self._call_ark_api(session, msg) for msg in messages]
            answers = await asyncio.gather(*tasks)
            
        if is_batch:
            return answers
        else:
            return answers[0]
    
    async def _call_ark_api(self, session, prompt_text: Optional[str] | Message) -> str:
        api_key = self.api_key if self.api_key else os.environ.get("ARK_API_KEY")
        if not api_key:
            raise RuntimeError(" ARK_API_KEY not set in environment variables")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        
        if isinstance(prompt_text, Message):
            prompt_text = prompt_text.content
        if prompt_text is None:
            raise RuntimeError("Prompt text is None")

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt_text
                }
            ],
        }

        max_retries = 5
        timeout_seconds = 100

        for attempt in range(max_retries):
            try:
                async with session.post(
                    self.end_point,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=timeout_seconds)
                ) as resp:

                    if 400 <= resp.status < 500:
                        msg = await resp.text()
                        # raise RuntimeError(f"Ark API returned {resp.status}: {msg}")
                        print(f"[ERROR] Ark API returned {resp.status}: {msg}, 返回空")
                        return ""

                    resp.raise_for_status()
                    data = await resp.json()

                if "choices" in data and len(data["choices"]) > 0:
                    return data["choices"][0]["message"]["content"].strip()

                assistant_text_parts = []
                for item in data.get("output", []):
                    if item.get("type") == "message":
                        for c in item.get("content", []):
                            if c.get("type") == "output_text":
                                assistant_text_parts.append(c.get("text", ""))

                return "\n".join(assistant_text_parts).strip()

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt < max_retries - 1:
                    wait = 0.2 * (2 ** attempt)  
                    print(f"[WARN] Ark API 调用失败，第 {attempt+1}/{max_retries} 次重试中... 错误: {e}")
    
    async def _call_ark_api_with_image(self, session, image_bs64: str, prompt_text: str) -> str:
        api_key = os.environ.get("ARK_API_KEY")
        if not api_key:
            raise RuntimeError("环境变量 ARK_API_KEY 未设置")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model_name,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_image",
                            "image_url": image_bs64,
                        },
                        {
                            "type": "input_text",
                            "text": prompt_text,
                        },
                    ],
                }
            ],
        }

        max_retries = 5
        timeout_seconds = 100

        for attempt in range(max_retries):
            try:
                async with session.post(
                    self.end_point,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=timeout_seconds)
                ) as resp:

                    if 400 <= resp.status < 500:
                        msg = await resp.text()
                        # raise RuntimeError(f"Ark API returned {resp.status}: {msg}")
                        print(f"[ERROR] Ark API returned {resp.status}: {msg}, 返回空")
                        return ""

                    resp.raise_for_status()
                    data = await resp.json()

                assistant_text_parts = []
                for item in data.get("output", []):
                    if item.get("type") == "message":
                        for c in item.get("content", []):
                            if c.get("type") == "output_text":
                                assistant_text_parts.append(c.get("text", ""))

                return "\n".join(assistant_text_parts).strip()

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt < max_retries - 1:
                    wait = 0.2 * (2 ** attempt)  
                    print(f"[WARN] Ark API 调用失败，第 {attempt+1}/{max_retries} 次重试中... 错误: {e}")
                    await asyncio.sleep(wait)
                    continue
                else:
                    print(f"[WARN] Ark API 调用失败，第 {attempt+1}/{max_retries} 次重试中... 错误: {e}, 返回空")
                    return ""

def prepare_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--inference_dir", type=str, required=True, help="Path to the store inference directory.")
    argparser.add_argument("--output_dir", type=str, required=True, help="Path to the output JSONL file.")
    argparser.add_argument("--llm_api_key", type=str, default=None, help="API key for LLM if needed.")
    argparser.add_argument("--end_point", type=str, default="https://api.openai.com/v1/chat/completions", help="LLM API endpoint.")
    argparser.add_argument("--model_name", type=str, default="gpt-4", help="LLM model name to use.")
    argparser.add_argument("--max_new_tokens", type=int, default=512)
    argparser.add_argument("--temperature", type=float, default=0.8)
    argparser.add_argument("--top_p", type=float, default=0.9)
    argparser.add_argument("--gpu_memory_utilization", type=float, default=0.7, help="GPU memory utilization for vLLM")
    argparser.add_argument("--batch_size", type=int, default=1, help="Batch size for processing. If > 1, uses batch processing.")

    args = argparser.parse_args()
    
    inference_file_name = Path(args.inference_dir).name
    print(inference_file_name)
    
    inference_dir_to_model_map = {
    }
    
    local_model_path = inference_dir_to_model_map.get(inference_file_name, "")
    
    llm = Here_LLM(
        model_name = args.model_name,
        end_point = args.end_point,
        api_key = args.llm_api_key
    )

    # load tokenizer   
    tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # initialize vLLM client
    tp = _infer_tp_size(default_tp=4) if not in_debug_mode() else 1
    print(f"Using tensor parallel size: {tp}")
    vllm_client = VLLM_LLM(
        model=local_model_path,
        tensor_parallel_size=tp,
        pipeline_parallel_size=1,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=8192,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=False if not in_debug_mode() else True,
    )
    
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=-1,
        n=1,
        max_tokens=args.max_new_tokens
    )
    
    vllm_config = Wrapped_VLLM_Config(sampling_params = sampling_params)
    
    
    args_dict = vars(args)
    args_dict['llm'] = llm
    args_dict['tokenizer'] = tokenizer
    args_dict['vllm_client'] = vllm_client
    
    return args_dict

def parse_score_from_response(response_text: str, default: float = 0.5) -> float:
    if not isinstance(response_text, str):
        return default

    match = re.search(r"<score>\s*([0-9]*\.?[0-9]+)\s*</score>", response_text)
    if not match:
        return float(default)

    try:
        value = float(match.group(1))
    except ValueError:
        return float(default)

    value = max(0.0, min(1.0, value))
    return value

def build_step_judge_prompt(
    question: str,
    golden_answer: str,
    up_context: str,
    current_state: str,
    current_think: str,
    current_action: str,
    current_observation: str,
) -> str:
    return STEP_JUDGE_PROMPT.format(
        question=question,
        golden_answer=golden_answer,
        up_context=up_context if up_context is not None else "",
        current_state=current_state if current_state is not None else "",
        current_think=current_think if current_think is not None else "",
        current_action=current_action if current_action is not None else "",
        current_observation=current_observation if current_observation is not None else "",
    )

def _safe_float(x: Any, default: float = 0.5) -> float:
    try:
        v = float(x)
        if not (v == v and np.isfinite(v)):  
            return default
        return float(min(max(v, 0.0), 1.0))
    except Exception:
        return default

def _find_json_substr(s: str) -> Optional[Dict[str, Any]]:
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(s[start:end + 1])
    except Exception:
        return None
    
@retry(
        wait=wait_random_exponential(min=1, max=2),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type(
            (Exception, ValueError)
        ), 
)
def get_action_verbal_confidence(vllm_client : Wrapped_VLLM_Client, traj: Dict[str, Any], step_index : int) -> float:
    steps = traj.get("steps", [])
    if step_index >= len(steps):
        return 0.5
    
    current_step = steps[step_index]
    think = current_step.get("think", "")
    action = current_step.get("action", "")
    
    # Construct history
    history_content = ""
    for i in range(step_index):
        s = steps[i]
        action_type = s.get("action_type", "")
        if action_type == "search" :
            history_content += f"<think> {s.get('think', '')} </think>\n <search>{s.get('action', '')} </search>\n <information>{s.get('observation', '')}</information>\n\n"
        else:
            history_content += f"<think> {s.get('think', '')} </think>\n <answer>{s.get('action', '')} </answer>\n"
 
        
    prompt = DEDUCE_CONFIDENCE_PROMPT.format(
        history_content=history_content,
        think_content=think,
        action_content=action
    )
    
    max_retries = 3
    for attempt in range(max_retries):
        sampling_params = SamplingParams(temperature=0.8, max_tokens=256)
        
        outputs = vllm_client.generate([prompt], sampling_params, use_tqdm=False)
        if not outputs:
            continue
            
        generated_text = outputs[0].outputs[0].text
        
        try:
            match = re.search(r"<confidence>\s*([0-9]*\.?[0-9]+)\s*</confidence>", generated_text)
            if match:
                print(f"Extracted confidence: {match.group(1)}")
                return _safe_float(match.group(1), 0.5)
            else:
                print(f"Attempt {attempt+1}/{max_retries}: Confidence value not found in generated text")
        except Exception as e:
            print(f"Attempt {attempt+1}/{max_retries}: Error parsing confidence: {e}")
            pass

    return 0.5

def construct_confidence_prompt(traj: Dict[str, Any], step_index: int) -> str:
    steps = traj.get("steps", [])
    if step_index >= len(steps):
        return ""
    
    current_step = steps[step_index]
    think = current_step.get("think", "")
    action = current_step.get("action", "")
    
    history_content = ""
    for i in range(step_index):
        s = steps[i]
        action_type = s.get("action_type", "")
        if action_type == "search" :
            history_content += f"<think> {s.get('think', '')} </think>\n <search>{s.get('action', '')} </search>\n <information>{s.get('observation', '')}</information>\n\n"
        else:
            history_content += f"<think> {s.get('think', '')} </think>\n <answer>{s.get('action', '')} </answer>\n"
 
    prompt = DEDUCE_CONFIDENCE_PROMPT.format(
        history_content=history_content,
        think_content=think,
        action_content=action
    )
    return prompt

def parse_confidence(generated_text: str) -> Optional[float]:
    try:
        match = re.search(r"<confidence>\s*([0-9]*\.?[0-9]+)\s*</confidence>", generated_text)
        if match:
            return _safe_float(match.group(1), 0.5)
    except Exception:
        pass
    return None

async def process_batch(batch_data: List[Dict[str, Any]], llm: Here_LLM, vllm_client):
    all_confidence_prompts = []
    all_judge_prompts = []
    
    confidence_map = [] 
    judge_map = []
    
    for b_idx, traj in enumerate(batch_data):
        question = traj['question']
        final_answer = traj.get('final_answer', "")
        golden_answer = traj.get('golden_answer', "")
        if isinstance(golden_answer, list):
            golden_answer = "\n".join(golden_answer)
            
        up_context = ""
        steps = traj.get('steps', [])
        
        for step_idx, step in enumerate(steps):
            conf_prompt = construct_confidence_prompt(traj, step_idx)
            all_confidence_prompts.append(conf_prompt)
            confidence_map.append((b_idx, step_idx))
            
            current_state = step.get("state", "")
            current_think = step.get("think", "")
            current_action = step.get("action", "")
            current_observation = step.get("observation", "")
            
            judge_prompt = build_step_judge_prompt(
                question=question,
                golden_answer=golden_answer,
                up_context=up_context,
                current_state=current_state,
                current_think=current_think,
                current_action=current_action,
                current_observation=current_observation,
            )
            all_judge_prompts.append(judge_prompt)
            judge_map.append((b_idx, step_idx))
            
            up_context += f"\n[STEP {step_idx}] STATE: {current_state}\nTHINK: {current_think}\nACTION: {current_action}\nOBS: {current_observation}\n"

    llm_task = None
    if all_judge_prompts:
        llm_task = asyncio.create_task(llm.ask(all_judge_prompts))

    confidence_results = [None] * len(all_confidence_prompts)
    
    pending_indices = list(range(len(all_confidence_prompts)))
    
    max_retries = 3
    sampling_params = SamplingParams(temperature=0.8, max_tokens=256)
    
    for attempt in range(max_retries):
        if not pending_indices:
            break
            
        prompts_to_run = [all_confidence_prompts[i] for i in pending_indices]
        
        if prompts_to_run:
            outputs = vllm_client.generate(prompts_to_run, sampling_params, use_tqdm=False)
            
            new_pending_indices = []
            
            for i, output in enumerate(outputs):
                original_idx = pending_indices[i]
                generated_text = output.outputs[0].text
                val = parse_confidence(generated_text)
                
                if val is not None:
                    confidence_results[original_idx] = val
                else:
                    new_pending_indices.append(original_idx)
            
            pending_indices = new_pending_indices
        else:
            break
    
    for i in pending_indices:
        confidence_results[i] = 0.5
        # print(f"Warning: Failed to get confidence for item {i}, using default 0.5")

    judge_scores = []
    if llm_task:
        judge_responses = await llm_task
        if isinstance(judge_responses, str): # Should be list if input is list, but safety check
            judge_responses = [judge_responses]
            
        judge_scores = [parse_score_from_response(r, default=0.5) for r in judge_responses]
    

    batch_confidences = {i: [] for i in range(len(batch_data))}
    batch_judges = {i: [] for i in range(len(batch_data))}
    
    for i, (b_idx, step_idx) in enumerate(confidence_map):
        batch_confidences[b_idx].append(confidence_results[i])
        
    for i, (b_idx, step_idx) in enumerate(judge_map):
        batch_judges[b_idx].append(judge_scores[i])
        
    for b_idx, traj in enumerate(batch_data):
        verbal_score = batch_confidences[b_idx]
        judge_score = batch_judges[b_idx]
        
        traj["judge_step_scores"] = judge_score
        traj["verbal_step_confidences"] = verbal_score
        
        stepwise_ece = stepwise_judge_ece(verbal_score, judge_score, n_bins=10)
        
        overconf_rate_tau06 = overconfidence_rate(verbal_score, judge_score, tau=0.6, judge_threshold=0.5)
        overconf_rate_tau07 = overconfidence_rate(verbal_score, judge_score, tau=0.7, judge_threshold=0.5)
        overconf_rate_tau08 = overconfidence_rate(verbal_score, judge_score, tau=0.8, judge_threshold=0.5)
        overconf_rate_tau09 = overconfidence_rate(verbal_score, judge_score, tau=0.9, judge_threshold=0.5)

        traj["stepwise_ece"] = stepwise_ece
        traj["overconfidence_rate_tau06"] = overconf_rate_tau06
        traj["overconfidence_rate_tau07"] = overconf_rate_tau07
        traj["overconfidence_rate_tau08"] = overconf_rate_tau08
        traj["overconfidence_rate_tau09"] = overconf_rate_tau09
        
    return batch_data

async def judge_single_jsonl_line(input_one_line_jsonl: Dict[str, Any], llm: LLM, vllm_client : Wrapped_VLLM_Client) -> Dict[str, Any]:
    verbal_score = []
    judge_scores = []
    

    traj = input_one_line_jsonl
    question = input_one_line_jsonl['question']
    final_answer = input_one_line_jsonl.get('final_answer', "")
    golden_answer = input_one_line_jsonl.get('golden_answer', "")
    if isinstance(golden_answer, list):
        golden_answer = "\n".join(golden_answer)

    up_context = ""

    for step_idx, step in enumerate(traj.get('steps', [])):
        verbal_score.append(get_action_verbal_confidence(vllm_client, traj, step_idx))

        current_state = step.get("state", "")
        current_think = step.get("think", "")
        current_action = step.get("action", "")
        current_observation = step.get("observation", "")

        prompt = build_step_judge_prompt(
            question=question,
            golden_answer=golden_answer,
            up_context=up_context,
            current_state=current_state,
            current_think=current_think,
            current_action=current_action,
            current_observation=current_observation,
        )

        response_text = await llm.ask(
            message=prompt
            )  

        score = parse_score_from_response(response_text, default=0.5)
        judge_scores.append(score)
        print(f"[Step {step_idx}] Judge score: {score}, Verbal score: {verbal_score[-1]}")

        up_context += f"\n[STEP {step_idx}] STATE: {current_state}\nTHINK: {current_think}\nACTION: {current_action}\nOBS: {current_observation}\n"

    stepwise_ece = stepwise_judge_ece(verbal_score, judge_scores, n_bins=10)
    
    overconf_rate_tau06 = overconfidence_rate(verbal_score, judge_scores, tau=0.6, judge_threshold=0.5)
    overconf_rate_tau07 = overconfidence_rate(verbal_score, judge_scores, tau=0.7, judge_threshold=0.5)
    overconf_rate_tau08 = overconfidence_rate(verbal_score, judge_scores, tau=0.8, judge_threshold=0.5)
    overconf_rate_tau09 = overconfidence_rate(verbal_score, judge_scores, tau=0.9, judge_threshold=0.5)

    input_one_line_jsonl["judge_step_scores"] = judge_scores
    input_one_line_jsonl["verbal_step_confidences"] = verbal_score
    input_one_line_jsonl["stepwise_ece"] = stepwise_ece
    input_one_line_jsonl["overconfidence_rate_tau06"] = overconf_rate_tau06
    input_one_line_jsonl["overconfidence_rate_tau07"] = overconf_rate_tau07
    input_one_line_jsonl["overconfidence_rate_tau08"] = overconf_rate_tau08
    input_one_line_jsonl["overconfidence_rate_tau09"] = overconf_rate_tau09

    return input_one_line_jsonl

async def run_main(args):
    inference_dir = Path(args['inference_dir'])
    output_dir = Path(args['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    llm = args['llm']
    vllm_client = args['vllm_client']
    
    target_files = [
        "step_90__ece_sampled_nq.jsonl",
        "step_90__ece_sampled_hotpotqa.jsonl",
        "step_90__ece_sampled_2wikimultihopqa.jsonl",
        "step_90__ece_sampled_bamboogle.jsonl",
        "step_90__ece_sampled_popqa.jsonl",
        "step_90__ece_sampled_triviaqa.jsonl",
    ]
    
    inference_prefix = inference_dir.name
    do_replace = False
    if "PeterJinGo_" in inference_prefix:
        inference_prefix = inference_prefix.replace("PeterJinGo_", "")
        do_replace = True
        
    if "rl_train_" in inference_prefix:
        inference_prefix = inference_prefix.replace("rl_train_", "")
        do_replace = True

    for filename in target_files:
        if do_replace:
            filename = filename.replace("step_90","")
            filename = f"hamster_train_1113_formatrl_3b{filename}"
        input_file = inference_dir / filename
        if not input_file.exists():
            print(f"Skipping {filename}: File not found in {inference_dir}")
            continue
            
        output_file = output_dir / f"evaluated_{filename}"
        summary_file = output_dir / f"summary_{filename.replace('.jsonl', '.txt')}"
        
        print(f"Processing {input_file} -> {output_file}")
        
        all_stepwise_ece = []
        all_overconf_06 = []
        all_overconf_07 = []
        all_overconf_08 = []
        all_overconf_09 = []
        
        # Load existing processed questions to skip
        processed_questions = set()
        if output_file.exists():
            print(f"Output file {output_file} exists. Loading processed questions...")
            with open(output_file, 'r', encoding='utf-8') as f_existing:
                for line in f_existing:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        existing_data = json.loads(line)
                        if "question" in existing_data:
                            processed_questions.add(existing_data["question"])
                            
                            # Also collect metrics from existing data to keep summary correct
                            if "stepwise_ece" in existing_data:
                                all_stepwise_ece.append(existing_data["stepwise_ece"])
                            if "overconfidence_rate_tau06" in existing_data:
                                all_overconf_06.append(existing_data["overconfidence_rate_tau06"])
                            if "overconfidence_rate_tau07" in existing_data:
                                all_overconf_07.append(existing_data["overconfidence_rate_tau07"])
                            if "overconfidence_rate_tau08" in existing_data:
                                all_overconf_08.append(existing_data["overconfidence_rate_tau08"])
                            if "overconfidence_rate_tau09" in existing_data:
                                all_overconf_09.append(existing_data["overconfidence_rate_tau09"])
                    except Exception:
                        pass
            print(f"Found {len(processed_questions)} processed questions.")

        # Open in append mode 'a' instead of write mode 'w'
        with open(input_file, 'r', encoding='utf-8') as f_in, \
             open(output_file, 'a', encoding='utf-8') as f_out:
            
            batch_data = []
            BATCH_SIZE = args.get('batch_size', 1)
            
            for line in tqdm(f_in, desc=f"Processing {filename}"):
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    data = json.loads(line)
                    
                    # Skip if already processed
                    if data.get("question") in processed_questions:
                        continue
                    
                    if BATCH_SIZE > 1:
                        batch_data.append(data)
                        if len(batch_data) >= BATCH_SIZE:
                            processed_batch = await process_batch(batch_data, llm, vllm_client)
                            for p_data in processed_batch:
                                # Collect metrics
                                if "stepwise_ece" in p_data:
                                    all_stepwise_ece.append(p_data["stepwise_ece"])
                                if "overconfidence_rate_tau06" in p_data:
                                    all_overconf_06.append(p_data["overconfidence_rate_tau06"])
                                if "overconfidence_rate_tau07" in p_data:
                                    all_overconf_07.append(p_data["overconfidence_rate_tau07"])
                                if "overconfidence_rate_tau08" in p_data:
                                    all_overconf_08.append(p_data["overconfidence_rate_tau08"])
                                if "overconfidence_rate_tau09" in p_data:
                                    all_overconf_09.append(p_data["overconfidence_rate_tau09"])
                                
                                f_out.write(json.dumps(p_data, ensure_ascii=False) + "\n")
                            f_out.flush()
                            batch_data = []
                    else:
                        processed_data = await judge_single_jsonl_line(data, llm, vllm_client)
                        
                        # Collect metrics
                        if "stepwise_ece" in processed_data:
                            all_stepwise_ece.append(processed_data["stepwise_ece"])
                        if "overconfidence_rate_tau06" in processed_data:
                            all_overconf_06.append(processed_data["overconfidence_rate_tau06"])
                        if "overconfidence_rate_tau07" in processed_data:
                            all_overconf_07.append(processed_data["overconfidence_rate_tau07"])
                        if "overconfidence_rate_tau08" in processed_data:
                            all_overconf_08.append(processed_data["overconfidence_rate_tau08"])
                        if "overconfidence_rate_tau09" in processed_data:
                            all_overconf_09.append(processed_data["overconfidence_rate_tau09"])
                        
                        f_out.write(json.dumps(processed_data, ensure_ascii=False) + "\n")
                        f_out.flush()
                    
                except Exception as e:
                    print(f"Error processing line in {filename}: {e}")
                    continue
            
            # Process remaining batch
            if batch_data and BATCH_SIZE > 1:
                try:
                    processed_batch = await process_batch(batch_data, llm, vllm_client)
                    for p_data in processed_batch:
                        # Collect metrics
                        if "stepwise_ece" in p_data:
                            all_stepwise_ece.append(p_data["stepwise_ece"])
                        if "overconfidence_rate_tau06" in p_data:
                            all_overconf_06.append(p_data["overconfidence_rate_tau06"])
                        if "overconfidence_rate_tau07" in p_data:
                            all_overconf_07.append(p_data["overconfidence_rate_tau07"])
                        if "overconfidence_rate_tau08" in p_data:
                            all_overconf_08.append(p_data["overconfidence_rate_tau08"])
                        if "overconfidence_rate_tau09" in p_data:
                            all_overconf_09.append(p_data["overconfidence_rate_tau09"])
                        
                        f_out.write(json.dumps(p_data, ensure_ascii=False) + "\n")
                    f_out.flush()
                except Exception as e:
                    print(f"Error processing remaining batch in {filename}: {e}")
                    
        # Calculate averages and write summary
        avg_ece = np.mean(all_stepwise_ece) if all_stepwise_ece else 0.0
        avg_overconf_06 = np.mean(all_overconf_06) if all_overconf_06 else 0.0
        avg_overconf_07 = np.mean(all_overconf_07) if all_overconf_07 else 0.0
        avg_overconf_08 = np.mean(all_overconf_08) if all_overconf_08 else 0.0
        avg_overconf_09 = np.mean(all_overconf_09) if all_overconf_09 else 0.0
        
        summary_content = (
            f"Input File: {input_file}\n"
            f"Total Samples: {len(all_stepwise_ece)}\n"
            f"Average Stepwise ECE: {avg_ece:.4f}\n"
            f"Average Overconfidence Rate (tau=0.6): {avg_overconf_06:.4f}\n"
            f"Average Overconfidence Rate (tau=0.7): {avg_overconf_07:.4f}\n"
            f"Average Overconfidence Rate (tau=0.8): {avg_overconf_08:.4f}\n"
            f"Average Overconfidence Rate (tau=0.9): {avg_overconf_09:.4f}\n"
        )
        
        with open(summary_file, 'w', encoding='utf-8') as f_sum:
            f_sum.write(summary_content)
            
        print(f"\nEvaluation Summary for {filename}:")
        print(summary_content)

async def run():
    args = prepare_args()
    
    # loop = asyncio.new_event_loop()
    # asyncio.set_event_loop(loop)
    # args['loop'] = loop
    
    await run_main(args)
    
    # loop.close()
    
if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("Process interrupted by user. Exiting...")
        exit(0)