from pathlib import Path
import sys
from typing import List, Dict, Any, Tuple, Optional, Union
import os
import json
import asyncio
import numpy as np
from tqdm import tqdm

from transformers import AutoTokenizer
from verl_hamster.Hamster_Generation_Manager import Hamster_Graph_Config
from CPT_FactorGraph_Run.build_trajectory_to_graph import GraphBuilder
from CPT_FactorGraph_Run.prompt import prompt_no_confidence, deduce_confidence_prompt, prompt_no_confidence_with_history
from hamster_tool.llm import LLMSettings as Hamster_LLMSettings
from hamster_tool.llm import LLM as Hamster_LLM
from hamster_tool.schema import Message
from CPT_FactorGraph_Run.utils import (
    compare_answers,
    extract_triples_robust,
    fix_tags_tokenizer_verbal,
    fix_tags_tokenizer_non_verbal,
    make_identity_obs,
    results_to_string,
)

def _safe_float(x: Any, default: float = 0.5) -> float:
    try:
        v = float(x)
        if not (v == v and np.isfinite(v)):  
            return default
        return float(min(max(v, 0.0), 1.0))
    except Exception:
        return default

def _is_json_obj(s: str) -> bool:
    try:
        j = json.loads(s)
        return isinstance(j, dict)
    except Exception:
        return False

def _find_json_substr(s: str) -> Optional[Dict[str, Any]]:
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(s[start:end + 1])
    except Exception:
        return None

class Hamster_Generation_Manager_Close_Source:
    def __init__(
        self,
        graph_config: Hamster_Graph_Config,
        llm_setting: Hamster_LLMSettings,
        max_iterations: int = 10,
        search_url: Optional[str] = None,
        is_verbal: bool = False,
        evidence_strength: float = 0.999,
        lbp_iters: int = 100,
        tokenizer: Optional[AutoTokenizer] = None,
    ):
        self.graph_config = graph_config
        self.graph_builder: Optional[GraphBuilder] = None
        if self.graph_config.cpt_model_path:
            self.graph_builder = GraphBuilder(
                self.graph_config.cpt_model_path,
                cpt_do_calibration=self.graph_config.cpt_do_calibration,
                sharpen_strength=self.graph_config.graph_sharpen_strength,
                min_assoc_strength=self.graph_config.graph_min_assoc_strength,
            )

        self.llm_setting = llm_setting
        self.llm = Hamster_LLM(llm_config=llm_setting)

        self.max_iterations = int(max_iterations)
        self.search_url = search_url
        self.is_verbal = is_verbal
        self.lbp_iters = lbp_iters
        self.evidence_strength = float(evidence_strength)
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen2.5-7B-Instruct",
                use_fast=True,
                trust_remote_code=True
            )
        self.tokenizer = tokenizer

    async def get_action_verbal_confidence(self, history_content: str, think: str, action: str) -> Tuple[float, float, Dict[str, Any]]:
        deduce_prompt_filled = deduce_confidence_prompt.format(
            history_content=history_content,
            think_content=think,
            action_content=action
        )
        raw_confidence = await self.llm.ask(
            messages = [Message.user_message(deduce_prompt_filled)])

        conf_json = None
        if _is_json_obj(raw_confidence):
            conf_json = json.loads(raw_confidence)
        else:
            conf_json = _find_json_substr(raw_confidence) or {}

        cS = _safe_float(conf_json.get("think_confidence"), 0.5)
        cA = _safe_float(conf_json.get("action_confidence"), 0.5)
        return cS, cA, conf_json

    async def run_llm_loop(self, jsonl_file_or_single_dict: Union[Path, Dict[str, Any]], output_path: Optional[Path]):
        print(f"Starting run_llm_loop with input: {jsonl_file_or_single_dict} and output: {output_path}")
        if isinstance(jsonl_file_or_single_dict, Path):
            return await self._run_jsonl_file(jsonl_file_or_single_dict, output_path)
        else:
            return await self._run_single_dict(jsonl_file_or_single_dict, output_path)

    async def _run_question(self, question: str) -> Dict[str, Any]:
        traj: Dict[str, Any] = {
            "question": question,
            "steps": [],
            "final_answer": None
        }

        for t in range(self.max_iterations):
            history_content = self._format_history_context(traj["steps"])
            if t == 0:
                gen_prompt = prompt_no_confidence.format(question=question)
            else:
                gen_prompt = prompt_no_confidence_with_history.format(history_content=history_content, question=question)
                
            #print(f"Generation Prompt at step {t}:\n{gen_prompt}\n")
                
            raw_generation = await self.llm.ask(
                messages = [Message.user_message(gen_prompt)])

            normalized = raw_generation.strip()
            normalized = (fix_tags_tokenizer_verbal if self.is_verbal else fix_tags_tokenizer_non_verbal)(self.tokenizer, normalized)

            triples = extract_triples_robust(normalized)
            think = triples[0].get("think") if triples else ""
            action = triples[0].get("action") if triples else ""
            action_type = triples[0].get("action_type") if triples else None

            cS, cA, conf_json = await self.get_action_verbal_confidence(history_content, think, action)

            observation, done = await self._execute_env(action_type, action)

            step_item = {
                "step": t,
                "think": think,
                "action": action,
                "action_type": action_type,
                "observation": observation,
                "think_confidence": cS,
                "action_confidence": cA,
                "raw_generation": raw_generation,
                "raw_confidence_json": conf_json
            }
            traj["steps"].append(step_item)

            if action_type == "answer":
                traj["final_answer"] = action
                break
            if done:
                break

        return traj

    async def _run_single_dict(
        self,
        data_dict: Dict[str, Any],
        output_path: Optional[Path],
        *,
        write_lock: Optional[asyncio.Lock] = None,
    ) -> Dict[str, Any]:
        if not (("question" in data_dict) and ("golden_answers" in data_dict)):
            raise ValueError("Input dict must contain 'question' and 'golden_answers'.")

        question: str = data_dict["question"]
        golden_answers: List[str] = data_dict["golden_answers"]
        #print(f"Processing question: {question}")
        #print(f"Golden answers: {golden_answers}")

        traj = await self._run_question(question)
        #print(f"Generated trajectory: {traj}")
        calib = self._calibrate_trajectory(traj, golden_answers)

        result = {
            "question": question,
            "trajectory": traj,
            "graph_calibrations": calib,
            "golden_answers": golden_answers
        }
        
        #print(f"output_file_path: {output_path}")

        if output_path:
            #print(f"writing to {output_path}")
            await self._write_to_file(result, output_path, write_lock=write_lock)
        return result

    async def _run_jsonl_file(self, jsonl_file: Path, output_path: Optional[Path]) -> Dict[str, Any]:
        if output_path is None:
            raise ValueError("output_path must be provided when processing JSONL files")

        processed_hashes: set[str] = set()
        if output_path.exists():
            with output_path.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        question = record.get("question")
                        if question is None:
                            continue
                        processed_hashes.add(self._hash_question(question))
                    except Exception:
                        continue

        outputs = []
        write_lock = asyncio.Lock()
        progress_lock = asyncio.Lock()

        progress_bar = None

        async def process_line(line: str):
            try:
                item = json.loads(line)
                question = item.get("question")
                if question is None:
                    return
                question_hash = self._hash_question(question)
                if question_hash in processed_hashes:
                    return
                # print(item)
                out = await self._run_single_dict(item, output_path, write_lock=write_lock)
                # print(out)
                # await self._write_to_file(out, output_path, write_lock=write_lock)
                processed_hashes.add(question_hash)
                outputs.append(out)
            except Exception as e:
                print(f"Error processing line: {e}")
                outputs.append({"error": str(e), "line": line})
            finally:
                async with progress_lock:
                    if progress_bar is not None:
                        progress_bar.update(1)

        with jsonl_file.open("r", encoding="utf-8") as f:
            lines = f.readlines()

        total = len(lines)
        with tqdm(total=total, desc=f"Processing {jsonl_file.name}") as bar:
            progress_bar = bar
            await asyncio.gather(*(process_line(line) for line in lines))

        return {"results": outputs}

    def _hash_question(self, question: str) -> str:
        import hashlib
        return hashlib.sha256(question.strip().encode("utf-8")).hexdigest()

    async def _write_to_file(
        self,
        result: Dict[str, Any],
        output_path: Path,
        *,
        write_lock: Optional[asyncio.Lock] = None,
    ) -> None:
        """
        Write the result to the file as soon as the processing is done.
        """
        # print(f"Attempting to write to {output_path}...")  # Log message
        # sys.stdout.flush()
        lock = write_lock or asyncio.Lock()
        async with lock:
            with output_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
            # print(f"Successfully wrote result to {output_path}")  # Log message
            # sys.stdout.flush()

    def _format_history_context(self, steps: List[Dict[str, Any]]) -> str:
        """
        将已有轨迹压缩为短上下文，给闭源 LLM 用于自评。
        """
        if not steps:
            return "No prior steps."
        lines = []
        for s in steps[-3:]: 
            t = s.get("step")
            think = (s.get("think") or "").strip()
            act_type = s.get("action_type") or "None"
            act = (s.get("action") or "").strip()
            obs = (s.get("observation") or "").strip()
            lines.append(
                f"Step {t}:\n<think>{think}</think>\n<action>{act}</action>\n<information>{obs}</information>\n"
            )
        return "\n".join(lines)

    async def _execute_env(self, action_type: Optional[str], action_content: str) -> Tuple[str, bool]:
        """
        search: 调用 batch 搜索；answer: 结束；无效：返回纠正提示。
        """
        if action_type == "answer":
            return "", True

        if action_type == "search":
            result = await self._search_async(action_content)
            obs = f"\n\n<information>{result.strip()}</information>\n\n"
            return obs, False

        # invalid
        tip = (
            "\nMy previous action is invalid. If I want to search, I should put the query "
            "between <search> and </search>. If I want to give the final answer, I should "
            "put the answer between <answer> and </answer>. Let me try again.\n"
        )
        return tip, False

    async def _search_async(self, query: str) -> str:
        if not self.search_url or not query:
            return ""
        import aiohttp
        payload = {"queries": [query], "topk": 5, "return_scores": True}
        async with aiohttp.ClientSession() as session:
            async with session.post(self.search_url, json=payload, timeout=6000) as resp:
                js = await resp.json()
        result = js.get("result", [""])[0]
        return results_to_string(result)

    def _calibrate_trajectory(self, trajectory: Dict[str, Any], golden_answers: List[str]) -> Optional[Dict[str, Any]]:
        if self.graph_builder is None:
            return None

        steps = trajectory.get("steps", [])
        if not steps:
            return None

        gb: GraphBuilder = self.graph_builder
        obs_eps = self.graph_config.graph_obs_eps
        lbp_iters = self.lbp_iters

        def build_graph_inplace(prior_mode: str):
            gb.reset()
            prev_o = None
            for s in steps:
                t = int(s.get("step", 0))
                S, A, O = f"S_{t}", f"A_{t}", f"O_{t}"

                gb.add_rv(S, 2, s.get("think", ""))
                gb.add_rv(A, 2, s.get("action", ""))
                gb.add_rv(O, 2, s.get("observation", ""))

                if t == 0:
                    cS = _safe_float(s.get("think_confidence"), 0.5)
                    prior_S = np.array([1.0 - cS, cS], np.float32) if prior_mode == "real" else np.array([0.5, 0.5], np.float32)
                    gb.add_factor([S], f"phi_prior_{S}", potential=prior_S)

                cA = _safe_float(s.get("action_confidence"), 0.5)
                prior_A = np.array([1.0 - cA, cA], np.float32) if prior_mode == "real" else np.array([0.5, 0.5], np.float32)
                gb.add_factor([A], f"phi_prior_{A}", potential=prior_A)

                gb.add_factor([S, A], f"phi_sa_{t}", potential=None)  # CPT
                gb.add_factor([A, O], f"phi_ao_{t}", potential=make_identity_obs(eps=obs_eps))
                if prev_o is not None:
                    gb.add_factor([prev_o, S], f"phi_os_{t-1}", potential=None)
                prev_o = O

        # forward 
        build_graph_inplace("real")
        gb.do_lbp(init=True, normalize=True, max_iters=lbp_iters)
        forward_marginals = {k: v.tolist() for k, v in gb.get_marginals(normalize=True).items()}

        # forward_no_evidence
        build_graph_inplace("uniform")
        gb.do_lbp(init=True, normalize=True, max_iters=lbp_iters)
        forward_no_evidence_marginals = {k: v.tolist() for k, v in gb.get_marginals(normalize=True).items()}

        # posterior
        build_graph_inplace("real")
        agent_answer = trajectory.get("final_answer")
        ok = None
        for g in golden_answers or []:
            ok = compare_answers(expected_answer=g, agent_answer=agent_answer)
            if ok is True:
                break
        if ok is not None:
            self._inject_exponential_back_evidence(
                gb, num_steps=len(steps), answer_true_or_false=bool(ok),
                s_max=self.evidence_strength, gamma=0.75, s_min=0.55,
                inject_A=False, inject_S=False
            )
        gb.do_lbp(init=True, normalize=True, max_iters=lbp_iters)
        posterior_marginals = {k: v.tolist() for k, v in gb.get_marginals(normalize=True).items()}

        return {
            "forward_marginals": forward_marginals,
            "forward_no_evidence_marginals": forward_no_evidence_marginals,
            "posterior_marginals": posterior_marginals,
            "answer_state": True if ok is True else False if ok is False else None,
        }

    def _inject_exponential_back_evidence(
        self,
        gb: GraphBuilder,
        num_steps: int,
        answer_true_or_false: bool,
        s_max: float = 0.95,
        gamma: float = 0.75,
        s_min: float = 0.55,
        inject_A: bool = False,
        inject_S: bool = False,
        alpha_A: float = 0.50,
        alpha_S: float = 0.25,
    ):
        """
        仅 O_t 注入强证据，A/S 可选弱证据（与你原实现一致的参数语义）。
        """
        def make_e(s: float) -> np.ndarray:
            s = max(s_min, min(s, 0.999))
            if answer_true_or_false:
                return np.array([1.0 - s, s], np.float32)
            else:
                return np.array([s, 1.0 - s], np.float32)

        k = num_steps - 1
        for t in range(k, -1, -1):
            base = s_max * (gamma ** (k - t))
            gb.add_factor([f"O_{t}"], f"evidence_O_{t}", potential=make_e(base))
            if inject_A:
                gb.add_factor([f"A_{t}"], f"evidence_A_{t}", potential=make_e(base * alpha_A))
            if inject_S:
                gb.add_factor([f"S_{t}"], f"evidence_S_{t}", potential=make_e(base * alpha_S))
