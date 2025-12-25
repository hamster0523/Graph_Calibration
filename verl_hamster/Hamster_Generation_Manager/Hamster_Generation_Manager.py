import sys
import torch
import re
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
import requests
import os
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential)

import torch
import torch.nn.functional as F
import numpy as np
from CPT_FactorGraph_Run.utils import (
                    compare_answers,
                    get_confidence_safe,
                    extract_triples_robust,
                    fix_tags_tokenizer_verbal,
                    fix_tags_tokenizer_non_verbal,
                    results_to_string,
                    token_confidence_from_dataproto,
                    make_identity_obs)
from CPT_FactorGraph_Run.build_trajectory_to_graph import GraphBuilder
from Wrapped_Vllm_Client import Wrapped_VLLM_Client
from Hamster_TensorHelper import TensorHelper, TensorConfig
from verl import DataProto
from Online_Search_Server import search_pipeline
from Online_Search_Server.config import PipelineConfig, WebReadAgentConfig
from Calibrate_RPC import CalibrateRPCClient

@dataclass
class Hamster_Graph_Config:
    cpt_model_path: Optional[str] = None
    cpt_rpc_url: Optional[str] = None
    cpt_do_calibration: bool = True
    graph_sharpen_strength: float = 3.0
    graph_min_assoc_strength: float = 0.15
    graph_obs_eps: float = 1e-6
    graph_lbp_iters: int = 100
    graph_evidence_strength: float = 0.9999
    graph_do_forward_lbp: bool = True
    
@dataclass
class Hamster_Generation_Config:
    max_turns: int
    max_start_length: int
    max_prompt_length: int 
    max_response_length: int
    max_obs_length: int
    num_gpus: int
    no_think_rl: bool = False
    search_url: Optional[str] = None
    topk: Optional[int] = None
    is_verbal : Optional[bool] = False
    graph_config : Hamster_Graph_Config = field(default_factory=Hamster_Graph_Config)
    
    online_search: bool = False
    serper_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    jina_api_key: Optional[str] = None
    llm_api_key: Optional[str] = None
    llm_model: Optional[str] = None
    llm_base_url: Optional[str] = None

class Hamster_LLM_Generation_Manager:
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config : Hamster_Generation_Config,
        is_validation : bool = False,
        test_vllm_engine : Optional[Wrapped_VLLM_Client] = None,
    ):
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.is_validation = is_validation

        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_obs_length=config.max_obs_length,
            max_start_length=config.max_start_length
        ))
        self.graph_builder = None
        if self.config.graph_config.cpt_model_path:
            self.graph_builder = GraphBuilder(
                self.config.graph_config.cpt_model_path,
                cpt_do_calibration=self.config.graph_config.cpt_do_calibration,
                sharpen_strength=self.config.graph_config.graph_sharpen_strength,
                min_assoc_strength=self.config.graph_config.graph_min_assoc_strength,
            )
        self.calibrate_rpc_client = None
        if self.config.graph_config.cpt_rpc_url:
            print(f"Initialize Calibrate RPC Client Successfully with url {self.config.graph_config.cpt_rpc_url}...")
            self.calibrate_rpc_client = CalibrateRPCClient(self.config.graph_config.cpt_rpc_url)
        
        self.graph_obs_eps = self.config.graph_config.graph_obs_eps
        self.graph_lbp_iters = self.config.graph_config.graph_lbp_iters
        self.graph_evidence_strength = self.config.graph_config.graph_evidence_strength
        self.test_vllm_engine = test_vllm_engine
        
        self.online_search_config = PipelineConfig()
        self.online_search_config.top_k = self.config.topk

        if os.getenv("SERPER_API_KEY") is None and self.config.serper_api_key is not None:
            os.environ["SERPER_API_KEY"] = self.config.serper_api_key
        if os.getenv("GEMINI_API_KEY") is None and self.config.gemini_api_key is not None:
            os.environ['GEMINI_API_KEY'] = self.config.gemini_api_key
        if os.getenv("JINA_API_KEY") is None and self.config.jina_api_key is not None:
            os.environ['JINA_API_KEY'] = self.config.jina_api_key

        read_config = WebReadAgentConfig(
            llm_model=self.config.llm_model if self.config.llm_model is not None else "gpt-5-mini",
            llm_base_url=self.config.llm_base_url if self.config.llm_base_url is not None else "https://api.agicto.cn/v1",
            llm_api_key=self.config.llm_api_key if self.config.llm_api_key is not None else ""
        )
        self.online_search_config.read = read_config

    def _extract_question_from_prompt(self, prompt_text: str) -> str:
        if not prompt_text:
            return ""
        match = re.search(r"Question:\s*(.*)", prompt_text, re.DOTALL)
        if match:
            return match.group(1).strip()
        lines = [line.strip() for line in prompt_text.splitlines() if line.strip()]
        return lines[-1] if lines else prompt_text.strip()

    def _initialize_trajectories(self, gen_batch: DataProto, initial_input_ids: torch.Tensor) -> List[Dict[str, Any]]:
        meta_info = getattr(gen_batch, "meta_info", {}) or {}
        batch_size = initial_input_ids.shape[0]
        decoded_inputs = self.tokenizer.batch_decode(initial_input_ids.detach().cpu().tolist(), skip_special_tokens=True)
        questions_meta = meta_info.get("questions")
        question_meta = meta_info.get("question")
        trajectories = []
        for idx in range(batch_size):
            question_text = None
            if isinstance(questions_meta, list) and idx < len(questions_meta):
                question_text = questions_meta[idx]
            elif isinstance(question_meta, list) and idx < len(question_meta):
                question_text = question_meta[idx]
            elif isinstance(question_meta, str):
                question_text = question_meta
            if not question_text:
                question_text = self._extract_question_from_prompt(decoded_inputs[idx])
            trajectories.append({
                "question": question_text,
                "prompt": decoded_inputs[idx],
                "steps": [],
                "final_answer": None
            })
        return trajectories
    
    def _get_valid_confidence(self, value1, value2) -> float:
        def count_decimal_places(number):
            num_str = str(number)
            if '.' in num_str:
                return len(num_str.split('.')[1])
            else:
                return 0
        decimal_value1 = count_decimal_places(value1) if value1 is not None else 0
        decimal_value2 = count_decimal_places(value2) if value2 is not None else 0
        if decimal_value1 > decimal_value2:
            return value1
        elif decimal_value2 > decimal_value1:
            return value2
        if value1 > value2:
            return value1
        else:
            return value2
    
    def _parse_response_segments_vllm(
        self,
        response: str,
        responses_ids: torch.Tensor,
        token_conf: torch.Tensor,
    ) -> Dict[str, Optional[Any]]:
        parsed: Dict[str, Optional[Any]] = {
            "think": None,
            "think_confidence": None,
            "action_confidence": None,
            "action_verbal_confidence": None,
            "action": None,
            "action_type": None,
        }

        if not response:
            return parsed

        raw_text = response  

        normalized = response.strip()
        if self.config.is_verbal:
            normalized = fix_tags_tokenizer_verbal(self.tokenizer, normalized)
        else:
            normalized = fix_tags_tokenizer_non_verbal(self.tokenizer, normalized)

        triples = extract_triples_robust(normalized)
        think_content = triples[0].get("think") if triples else None
        action_content = triples[0].get("action") if triples else None
        action_type = triples[0].get("action_type") if triples else None

        action_verbal_confidence = None
        if triples:
            c_text = triples[0].get("confidence")
            if c_text is not None and str(c_text).strip() != "":
                action_verbal_confidence = str(c_text).strip()

        if action_verbal_confidence is None:
            m = re.search(r"<confidence>(.*?)</confidence>", normalized, re.DOTALL)
            if not m:
                m = re.search(r"<confidence>(.*?)</confidence>", raw_text, re.DOTALL)
            if m and m.group(1).strip():
                action_verbal_confidence = m.group(1).strip()

        if action_verbal_confidence is not None:
            parsed["action_confidence"] = self._safe_confidence_to_float(action_verbal_confidence)
        else:
            parsed["action_confidence"] = 0.0   

        parsed["think"] = think_content
        parsed["action"] = action_content
        parsed["action_type"] = action_type
        parsed["action_verbal_confidence"] = action_verbal_confidence

        def _effective_len(row: torch.Tensor) -> int:
            pad_id = self.tokenizer.pad_token_id
            if pad_id is None:
                return int((row != row.new_tensor(0)).sum().item()) or row.numel()
            return int((row != pad_id).sum().item())

        B = 1 if responses_ids.dim() == 1 else responses_ids.size(0)
        T_conf = token_conf.numel()
        if responses_ids.dim() == 1:
            ids_row = responses_ids
        else:
            ids_row = None
            for b in range(B):
                if _effective_len(responses_ids[b]) == T_conf:
                    ids_row = responses_ids[b]
                    break
            if ids_row is None:
                ids_row = responses_ids[0][:T_conf]

        ids = ids_row.tolist()
        probs_np = token_conf.detach().cpu().float().numpy()  

        def _encode(text: str) -> list[int]:
            return self.tokenizer.encode(text, add_special_tokens=False)

        def _find_subseq(hay: list[int], needle: list[int]) -> Optional[Tuple[int, int]]:
            m, n = len(hay), len(needle)
            if n == 0 or n > m:
                return None
            for s in range(m - n + 1):
                if hay[s:s + n] == needle:
                    return (s, s + n)
            return None

        def _find_best_span(hay: list[int], needle: list[int]) -> Optional[Tuple[int, int]]:
            m, n = len(hay), len(needle)
            best_start, best_len = -1, 0
            for s in range(m):
                max_k = min(n, m - s)
                k = 0
                while k < max_k and hay[s + k] == needle[k]:
                    k += 1
                if k == n:
                    return (s, s + n)
                if k > best_len:
                    best_len, best_start = k, s
            if best_len > 0:
                return (best_start, best_start + best_len)
            return None

        def _find_best_overlap_span(hay: list[int], needle: list[int], min_match: int = 1) -> Optional[Tuple[int, int]]:
            m, n = len(hay), len(needle)
            if m == 0 or n == 0:
                return None
            best_len, best_end_i = 0, -1
            prev = [0] * (n + 1)
            for i in range(m):
                curr = [0] * (n + 1)
                for j in range(1, n + 1):
                    if hay[i] == needle[j - 1]:
                        curr[j] = prev[j - 1] + 1
                        if curr[j] > best_len:
                            best_len, best_end_i = curr[j], i
                prev = curr
            if best_len >= min_match:
                start = best_end_i - best_len + 1
                return (start, start + best_len)
            return None

        def _span_confidence(span: Optional[Tuple[int, int]]) -> Optional[float]:
            if not span:
                return None
            s, e = span
            if e <= s or e > len(probs_np):
                return None
            p = probs_np[s:e]
            if p.size == 0:
                return None
            return float(p.mean())

        think_conf = 0.5
        if think_content:
            tid = _encode(think_content.strip())
            sp = _find_subseq(ids, tid) or _find_best_span(ids, tid) or _find_best_overlap_span(ids, tid, 1)
            c = _span_confidence(sp)
            if c is not None:
                think_conf = c

        # ACTION（search / answer）
        action_conf = 0.5
        if action_type in ("search", "answer") and action_content:
            aid = _encode(action_content.strip())
            sp = _find_subseq(ids, aid) or _find_best_span(ids, aid) or _find_best_overlap_span(ids, aid, 1)
            c = _span_confidence(sp)
            if c is not None:
                action_conf = c

        # THINK Use SpecialToken To Get
        think_conf_2 = 0.5
        if think_content:
            think_conf_2 = self._extract_tagged_spans_with_token_conf(
                generated_tokens=ids, token_conf=token_conf, open_tag="<think>", close_tag="</think>"
            )
            if think_conf_2 is not None:
                think_conf_2 = (
                    think_conf_2.get("parts", [])[0].get("confidence", 0.5)
                    if think_conf_2.get("parts")
                    else 0.5
                )

        # ACTION Use SpecialToken To Get
        action_conf_2 = 0.5
        if action_type in ("search", "answer") and action_content:
            action_conf_2 = self._extract_tagged_spans_with_token_conf(
                generated_tokens=ids, token_conf=token_conf, open_tag=f"<{action_type}>", close_tag=f"</{action_type}>"
            )
            if action_conf_2 is not None:
                action_conf_2 = (
                    action_conf_2.get("parts", [])[0].get("confidence", 0.5)
                    if action_conf_2.get("parts")
                    else 0.5
                )

        if not self.config.is_verbal:
            parsed["think_confidence"] = self._get_valid_confidence(think_conf, think_conf_2)
            parsed["action_confidence"] = self._get_valid_confidence(action_conf, action_conf_2)
        else:
            parsed["think_confidence"] = self._get_valid_confidence(think_conf, think_conf_2)
            
        # print(f"response is : {response}\n")
        # print(f"parsed is : {parsed}\n")

        return parsed


    def _parse_response_segments(
        self,
        response: str,
        response_ids: Optional[torch.Tensor] = None,
        token_logits: Optional[torch.Tensor] = None,
    ) -> Dict[str, Optional[Any]]:
        parsed: Dict[str, Optional[Any]] = {
            "think": None,
            "think_confidence": None,
            "action_confidence": None,
            "action": None,
            "action_type": None,
        }

        if not response:
            return parsed

        normalized = response.strip()
        #print(f"Raw response: {response}")
        if self.config.is_verbal:
            normalized = fix_tags_tokenizer_verbal(self.tokenizer, normalized)
        else:
            normalized = fix_tags_tokenizer_non_verbal(self.tokenizer, normalized)
        #print(f"Fix response: {normalized}")
        # if self.config.is_verbal:
        #     try:
        #         print(f"before fix is : {normalized}")
        #         fixed_text = fix_tags_tokenizer(self.tokenizer, normalized)
        #         print(f"fixed text is : {fixed_text}")
        #         triples = extract_triples(fixed_text)
        #         if triples:
        #             sample = triples[0]
        #             parsed["think"] = sample.get("think")
        #             parsed["confidence_text"] = sample.get("confidence")
        #             parsed["action"] = sample.get("action")
        #             parsed["action_type"] = sample.get("action_type")
        #     except Exception:
        #         pass
        # if parsed["action"] is None or parsed["action_type"] is None:
        #     try:
        #         actions, contents = self.postprocess_predictions([response])
        #         parsed["action_type"] = actions[0]
        #         parsed["action"] = contents[0]
        #     except Exception:
        #         parsed["action_type"] = None
        #         parsed["action"] = None
        parsed_dict = extract_triples_robust(normalized)
        think_content = parsed_dict[0].get("think") if parsed_dict else None
        #confidence_content = parsed_dict[0].get("confidence") if parsed_dict else None
        action_content = parsed_dict[0].get("action") if parsed_dict else None
        action_type = parsed_dict[0].get("action_type") if parsed_dict else None
        parsed["think"] = think_content
        #parsed["confidence_text"] = confidence_content
        parsed["action"] = action_content
        parsed["action_type"] = action_type
        
        #fix_inputs_ids = self.tokenizer.encode(normalized, return_tensors="pt")

        if self.config.is_verbal:
            match = re.search(r"<confidence>(.*?)</confidence>", normalized, re.DOTALL)
            if match and match.group(1).strip() is not None:
                parsed["action_confidence"] = self._safe_confidence_to_float(match.group(1).strip())
            else:
                print("Warning: No confidence tag found in verbal mode, defaulting to 0.5")
                parsed["action_confidence"] = 0.5
        else:
            if token_logits is not None:
                think_confidence, search_confidence, answer_confidence = self._logits_to_confidence(response_ids, token_logits)
                
                # parsed["think_confidence"] = think_confidence
                # if action_type == "answer":
                #     parsed["action_confidence"] = answer_confidence
                # elif action_type == "search":
                #     parsed["action_confidence"] = search_confidence
                
                t_conf, a_conf = self._confidence_from_contents_via_ids(
                    response_ids=response_ids[0],          
                    token_logits=token_logits,                 
                    think_content=think_content,
                    action_content=action_content,
                    action_type=action_type,
                )
                # parsed["think_confidence"] = t_conf if t_conf is not None else parsed.get("think_confidence")
                # parsed["action_confidence"] = a_conf if a_conf is not None else parsed.get("action_confidence")

                # valid_think_conf = think_confidence if think_confidence != 0.5 else t_conf
                # valid_action_conf = 0.5
                # if action_type == "answer":
                #     valid_action_conf = answer_confidence if answer_confidence != 0.5 else a_conf
                # else:
                #     valid_action_conf = search_confidence if search_confidence != 0.5 else a_conf
                parsed["think_confidence"] = self._get_valid_confidence(think_confidence, t_conf)
                if action_type == "answer":
                    parsed["action_confidence"] = self._get_valid_confidence(answer_confidence, a_conf)
                else:
                    parsed["action_confidence"] = self._get_valid_confidence(search_confidence, a_conf)

        # if parsed["confidence_value"] is None and parsed.get("confidence_text") is not None:
        #     parsed["confidence_value"] = self._safe_confidence_to_float(parsed["confidence_text"])
        # if parsed.get("confidence_text") is None and parsed.get("confidence_value") is not None:
        #     parsed["confidence_text"] = f"{parsed['confidence_value']:.6f}"

        return parsed

    def _safe_confidence_to_float(self, value: Optional[str]) -> Optional[float]:
        if value is None:
            return None
        try:
            numeric = float(value)
            if not np.isfinite(numeric):
                return None
            return float(min(max(numeric, 0.0), 1.0))
        except (TypeError, ValueError):
            return None

    def _logits_to_confidence(self, response_ids, token_logits: List[torch.Tensor]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        # if token_log_probs is None or token_log_probs.numel() == 0:
        #     return None
        # if torch.isnan(token_log_probs).any():
        #     return None
        # mean_log_prob = token_log_probs.mean().item()
        # confidence = float(np.clip(np.exp(mean_log_prob), 0.0, 1.0))
        # return confidence
        think_scores  = self._extract_tagged_spans_with_probs(response_ids[0], token_logits, "<think>", "</think>")
        search_scores = self._extract_tagged_spans_with_probs(response_ids[0], token_logits, "<search>", "</search>")
        answer_scores = self._extract_tagged_spans_with_probs(response_ids[0], token_logits, "<answer>", "</answer>")

        think_confidence  = get_confidence_safe(think_scores)
        search_confidence = get_confidence_safe(search_scores)
        answer_confidence = get_confidence_safe(answer_scores)
        
        return (think_confidence, search_confidence, answer_confidence)
    
    def _extract_tagged_spans_with_token_conf(
        self,
        generated_tokens: List[int],
        token_conf: torch.Tensor | List[float],   # [T], 每步被选中 token 的概率 ∈ [0,1]
        open_tag: str,
        close_tag: str,
        lookahead: int = 8,
    ) -> Optional[Dict[str, Any]]:

        ids = list(generated_tokens)
        if isinstance(token_conf, torch.Tensor):
            probs_all = token_conf.detach().cpu().float().tolist()
        else:
            probs_all = [float(x) for x in token_conf]
        T = min(len(ids), len(probs_all))
        ids = ids[:T]
        probs_all = probs_all[:T]

        def _find_tag_hits(tag: str, relaxed: bool = True):
 
            hits = []
            for pos in range(len(ids)):
                for k in range(1, min(lookahead, len(ids) - pos) + 1):
                    s = self.tokenizer.decode(ids[pos:pos + k], skip_special_tokens=False)
                    check_s = s.strip() if relaxed else s
                    if check_s.startswith(tag):
                        hits.append((pos, k))
                        break
            return hits

        def _pair_by_stack(opens, closes):
            events = [(p, "open", c) for p, c in opens] + [(p, "close", c) for p, c in closes]
            events.sort(key=lambda x: (x[0], 0 if x[1] == "open" else 1))
            spans, stack = [], []
            for pos, typ, cons in events:
                if typ == "open":
                    stack.append((pos, cons))
                else:
                    if not stack:
                        continue
                    o_pos, o_cons = stack.pop()
                    start, end = o_pos + o_cons, pos
                    if start < end:
                        spans.append((start, end))
            return spans

        open_hits = _find_tag_hits(open_tag)
        close_hits = _find_tag_hits(close_tag)
        if not open_hits or not close_hits:
            return None

        spans = _pair_by_stack(open_hits, close_hits)
        if not spans:
            return None

        parts = []
        for start, end in spans:
            s = max(0, min(start, T))
            e = max(0, min(end, T))
            if e <= s:
                continue

            toks = ids[s:e]
            texts = [self.tokenizer.decode([tid], skip_special_tokens=False) for tid in toks]
            probs = probs_all[s:e]  
            conf = float(sum(probs) / len(probs)) if len(probs) > 0 else None

            parts.append({
                "span": (s, e),
                "tokens": toks,
                "texts": texts,
                "logits": [None] * len(probs),  
                "probs": probs,                 
                "confidence": conf,            
            })

        return {"parts": parts} if parts else None

        
    def _extract_tagged_spans_with_probs(
        self,     
        generated_tokens: List[int],
        token_logits: List[torch.Tensor],
        open_tag: str,
        close_tag: str,
        lookahead: int = 8,
        ) -> Optional[Dict[str, Any]]:

        ids = list(generated_tokens)

        def _find_tag_hits(tag: str,relaxed: bool = True):

            hits = []
            for pos in range(len(ids)):
                for k in range(1, min(lookahead, len(ids) - pos) + 1):
                    s = self.tokenizer.decode(ids[pos:pos+k], skip_special_tokens=False)
                    check_s = s.strip() if relaxed else s   
                    if check_s.startswith(tag):
                        hits.append((pos, k))
                        break
            return hits

        def _pair_by_stack(opens, closes):
            events = [(p,"open",c) for p,c in opens] + [(p,"close",c) for p,c in closes]
            events.sort(key=lambda x: (x[0], 0 if x[1]=="open" else 1))
            spans, stack = [], []
            for pos,typ,cons in events:
                if typ=="open":
                    stack.append((pos,cons))
                else:
                    if not stack: continue
                    o_pos,o_cons = stack.pop()
                    start, end = o_pos+o_cons, pos
                    if start < end:
                        spans.append((start,end))
            return spans

        open_hits = _find_tag_hits(open_tag)
        close_hits = _find_tag_hits(close_tag)
        if not open_hits or not close_hits:
            return None

        spans = _pair_by_stack(open_hits, close_hits)
        if not spans:
            return None

        results = []
        for start, end in spans:
            toks, texts, logits_list, probs_list = [], [], [], []
            token_conf_list = []    

            for t in range(start, end):
                vec = token_logits[t]
                if isinstance(vec, torch.Tensor) and vec.dim() == 2:
                    vec = vec[0]

                tok_id = ids[t]
                toks.append(tok_id)
                texts.append(self.tokenizer.decode([tok_id], skip_special_tokens=False))

                if isinstance(vec, torch.Tensor) and vec.numel() > 1:
                    probs_full = F.softmax(vec, dim=-1)
                    logits_list.append(float(vec[tok_id].item()))
                    probs_list.append(float(probs_full[tok_id].item()))

                    if self.config.topk is None or self.config.topk <= 0:
                        ci = float((-probs_full[tok_id].clamp_min(1e-12).log()).item())
                    else:
                        k = min(self.config.topk, vec.numel())
                        logits_k = vec
                        v, _ = torch.topk(logits_k, k)
                        masked = logits_k.clone()
                        masked[masked < v[-1]] = float("-inf")
                        probs_k = F.softmax(masked, dim=-1)
                        topk_probs, _ = torch.topk(probs_k, k)
                        ci = float((-topk_probs.clamp_min(1e-12).log().mean()).item())
                    token_conf_list.append(ci)

                else:
                    lp = float(vec.item()) if isinstance(vec, torch.Tensor) else float(vec)
                    logits_list.append(None)
                    probs_list.append(float(np.exp(lp)))
                    ci = -lp
                    token_conf_list.append(ci)

            confidence = float(np.mean(token_conf_list)) if token_conf_list else None

            results.append({
                "span": (start, end),
                "tokens": toks,
                "texts": texts,
                "logits": logits_list,
                "probs": probs_list,
                "confidence": confidence,      
            })

        return {"parts": results}

    def _confidence_from_contents_via_ids(
        self,
        response_ids: Optional[torch.Tensor],                 
        token_logits: Optional[torch.Tensor],             
        think_content: Optional[str],
        action_content: Optional[str],
        action_type: Optional[str],
        eps: float = 1e-12,
    ) -> Tuple[Optional[float], Optional[float]]:


        if isinstance(response_ids, torch.Tensor):
            if response_ids.dim() == 2:
                assert response_ids.size(0) == 1, ""
                ids = response_ids[0].tolist()
            else:
                ids = response_ids.view(-1).tolist()
        else:
            ids = list(response_ids)

        if isinstance(token_logits, torch.Tensor):
            if token_logits.dim() == 3:
                assert token_logits.size(0) == 1, "token_logits 应是当前样本的 1 条响应"
                token_logits_seq = [token_logits[0, t] for t in range(token_logits.size(1))]
            elif token_logits.dim() == 2:
                token_logits_seq = [token_logits[t] for t in range(token_logits.size(0))]
            else:
                raise ValueError(f"token_logits 维度不支持: {token_logits.shape}")
        else:
            token_logits_seq = token_logits  

        def _encode(text: str) -> List[int]:
            return self.tokenizer.encode(text, add_special_tokens=False)

        def _find_subseq(hay: List[int], needle: List[int]) -> Optional[Tuple[int, int]]:
            if not hay or not needle or len(needle) > len(hay):
                return None
            m, n = len(hay), len(needle)
            for s in range(0, m - n + 1):
                if hay[s:s + n] == needle:
                    return (s, s + n)
            return None
        
        def _find_best_span(hay: List[int], needle: List[int]) -> Optional[Tuple[int, int]]:

            if not hay or not needle:
                return None

            m, n = len(hay), len(needle)
            best_start, best_len = -1, 0

            for s in range(m):
                max_k = min(n, m - s)
                k = 0
                while k < max_k and hay[s + k] == needle[k]:
                    k += 1

                if k == n:
                    return (s, s + n)

                if k > best_len:
                    best_len = k
                    best_start = s

            if best_len > 0:
                return (best_start, best_start + best_len)
            return None

        def _find_best_overlap_span(hay: list[int], needle: list[int], min_match: int = 1) -> Optional[tuple[int, int]]:

            if not hay or not needle:
                return None

            m, n = len(hay), len(needle)
            best_len, best_end_i = 0, -1

            prev = [0] * (n + 1)
            for i in range(m):
                curr = [0] * (n + 1)
                hi = hay[i]
                for j in range(1, n + 1):
                    if hi == needle[j - 1]:
                        curr[j] = prev[j - 1] + 1
                        if curr[j] > best_len:
                            best_len = curr[j]
                            best_end_i = i
                prev = curr

            if best_len >= min_match:
                start = best_end_i - best_len + 1
                return (start, start + best_len)
            return None

        def _span_confidence(start: int, end: int) -> Optional[float]:
            if start is None or end is None or end <= start:
                return None
            if end > len(ids) or end > len(token_logits_seq):
                return None

            nll_vals = []
            for t in range(start, end):
                vec = token_logits_seq[t]
             
                if vec.dim() == 2:
                    vec = vec[0]
                probs = F.softmax(vec, dim=-1)           
                tok_id = ids[t]
                p_t = float(probs[tok_id].item())           
                nll_vals.append(-np.log(max(p_t, eps)))  

            return float(np.mean(nll_vals)) if nll_vals else None

        # ---- THINK ----
        think_conf: Optional[float] = 0.5
        if think_content:
            think_ids = _encode(think_content.strip())
            span = _find_subseq(ids, think_ids)
            if span is None:
                span = _find_best_span(ids, think_ids)
            if span is None:
                span = _find_best_overlap_span(ids, think_ids, min_match=1)
            if span is not None:
                think_conf = _span_confidence(*span)

        action_conf: Optional[float] = 0.5
        if action_type in ("search", "answer") and action_content:
            action_ids = _encode(action_content.strip())
            span = _find_subseq(ids, action_ids)
            if span is None:
                span = _find_best_span(ids, action_ids)
            if span is None:
                span = _find_best_overlap_span(ids, action_ids, min_match=1)
            if span is not None:
                action_conf = _span_confidence(*span)

        return think_conf, action_conf

    def _append_trajectory_step(
        self,
        trajectory_buffer: Dict[str, Any],
        parsed_segments: Dict[str, Optional[str]],
        observation_text: str,
        done: bool,
        response_text: str,
    ) -> None:
        think_text = (parsed_segments.get("think") or "").strip()
        action_text = (parsed_segments.get("action") or "").strip()
        action_type = parsed_segments.get("action_type")
        think_confidence = parsed_segments.get("think_confidence")
        action_confidence = parsed_segments.get("action_confidence")
        action_verbal_confidence = parsed_segments.get("action_verbal_confidence", 0.5)

        step_index = len(trajectory_buffer["steps"])
        question_text = trajectory_buffer.get("question") or ""
        if step_index == 0 and question_text:
            state_text = f"question : {question_text}\nthink : {think_text}"
        else:
            state_text = f"think : {think_text}"


        format_ok = self._is_format_valid(response_text)

        trajectory_buffer["steps"].append({
            "step": step_index,
            "state": state_text,
            "think": think_text,
            "action": action_text,
            "action_type": action_type,
            "observation": (observation_text or "").strip(),
            "think_confidence": think_confidence,
            "action_confidence": action_confidence,
            "action_verbal_confidence": action_verbal_confidence,
            "done": bool(done),
            "raw_response": response_text,
            "format_ok": format_ok,          
        })

        if action_type == "answer":
            trajectory_buffer["final_answer"] = action_text
            trajectory_buffer["final_think_confidence"] = think_confidence
            trajectory_buffer["final_answer_confidence"] = action_confidence
            trajectory_buffer["final_action_verbal_confidence"] = action_verbal_confidence

        
    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type(
            (Exception, ValueError)
        ), 
    )
    def _rpc_calibrate_trajectory(self, trajectory: dict, golden_answer: Optional[str]) -> Optional[Dict[str, Any]]:
        #print("call rpc calibrate trajectory...")
        if not trajectory or not trajectory.get("steps"):
            return None
        try:
            #print("rpc calibrate trajectory in progress...")
            result = self.calibrate_rpc_client.call_calibrate_trajectory(trajectory=trajectory, golden_answer=golden_answer)
            required_keys = {
                "forward_marginals",
                "forward_no_evidence_marginals",
                "posterior_marginals",
            }
            if not isinstance(result, dict) or not required_keys.issubset(result.keys()):
                print(f"RPC return format error：{list(result.keys()) if isinstance(result, dict) else type(result)}")
                return None
            #print(f"rpc calibrate trajectory done with result : {result}")
            return result
        except Exception as e:
            print(f"RPC call exception: {e}")
            return None

    def _calibrate_trajectory(self, trajectory: Dict[str, Any], golden_answer: Optional[str]) -> Optional[Dict[str, Any]]:
        print("enter into calibrate trajectory...")
        if not trajectory.get("steps"):
            return None

        gb = self.graph_builder
        if gb is None:
            if self.calibrate_rpc_client is None:
                raise RuntimeError("No Local GraphBuilder，either rpc client must be provided for remote calibration.")
            print("rpc calibrate trajectory in progress...")
            rpc_res = self._rpc_calibrate_trajectory(trajectory, golden_answer)
            return rpc_res  

        print("local calibrate trajectory in progress...")
        sanitized_steps = []
        for step in trajectory["steps"]:
            t = step.get("step", len(sanitized_steps))
            sanitized_steps.append({
                "step": t,
                "state_text": step.get("state", "") or "",
                "action_text": step.get("action", "") or "",
                "obs_text": step.get("observation", "") or "",
                "action_type": step.get("action_type"),
                "think_conf": float(step.get("think_confidence") or 0.5),
                "action_conf": float(step.get("action_confidence") or 0.5),
            })

        def build_graph_inplace(prior_mode: str):
            gb.reset()
            prev_o = None
            for st in sanitized_steps:
                t = st["step"]
                s, a, o = f"S_{t}", f"A_{t}", f"O_{t}"
                gb.add_rv(s, 2, st["state_text"])
                gb.add_rv(a, 2, st["action_text"])
                gb.add_rv(o, 2, st["obs_text"])
                if t == 0:
                    p_s0 = np.array([1.0 - st["think_conf"], st["think_conf"]], np.float32) if prior_mode == "real" else np.array([0.5, 0.5], np.float32)
                    gb.add_factor([s], f"phi_prior_{s}", potential=p_s0)
                p_at = np.array([1.0 - st["action_conf"], st["action_conf"]], np.float32) if prior_mode == "real" else np.array([0.5, 0.5], np.float32)
                gb.add_factor([a], f"phi_prior_{a}", potential=p_at)
                gb.add_factor([s, a], f"phi_sa_{t}", potential=None)
                gb.add_factor([a, o], f"phi_ao_{t}", potential=make_identity_obs(eps=self.graph_obs_eps))
                if prev_o is not None:
                    gb.add_factor([prev_o, s], f"phi_os_{t-1}", potential=None)
                prev_o = o

        # forward
        build_graph_inplace(prior_mode="real")
        gb.do_lbp(init=True, normalize=True, max_iters=self.graph_lbp_iters)
        forward_marginals = {name: p.tolist() for name, p in gb.get_marginals(normalize=True).items()}
        # forward_no_evidence
        build_graph_inplace(prior_mode="uniform")
        gb.do_lbp(init=True, normalize=True, max_iters=self.graph_lbp_iters)
        forward_no_evidence_marginals = {name: p.tolist() for name, p in gb.get_marginals(normalize=True).items()}
        build_graph_inplace(prior_mode="real")
        agent_answer = trajectory.get("final_answer")
        answer_true_or_false = compare_answers(expected_answer=golden_answer, agent_answer=agent_answer)
        self._inject_exponential_back_evidence(
            gb=gb,
            num_steps=len(trajectory["steps"]),
            answer_true_or_false=answer_true_or_false,
            s_max=float(getattr(self, "graph_evidence_strength", 0.95)),
            gamma=0.75,
            s_min=0.55,
            inject_A=False,
            inject_S=False,
            alpha_A=0.50,
            alpha_S=0.25,
        )
        gb.do_lbp(init=True, normalize=True, max_iters=self.graph_lbp_iters)
        posterior_marginals = {name: p.tolist() for name, p in gb.get_marginals(normalize=True).items()}

        return {
            "question": trajectory.get("question"),
            "prompt": trajectory.get("prompt"),
            "steps": [{
                "step": st["step"],
                "state": st["state_text"],
                "action": st["action_text"],
                "action_type": st["action_type"],
                "observation": st["obs_text"],
                "think_confidence": st["think_conf"],
                "action_confidence": st["action_conf"],
            } for st in sanitized_steps],
            "forward_marginals": forward_marginals,
            "forward_no_evidence_marginals": forward_no_evidence_marginals,
            "posterior_marginals": posterior_marginals,
            "final_answer": trajectory.get("final_answer"),
            "golden_answer": golden_answer,
            "answer_state": True if answer_true_or_false is True else False if answer_true_or_false is False else None,
        }
    
    def _inject_exponential_back_evidence(
        self,
        gb,
        num_steps: int,
        answer_true_or_false: bool | None,
        s_max: float = 0.95,    
        gamma: float = 0.75,     
        s_min: float = 0.55,    
        inject_A: bool = False,  
        inject_S: bool = False,  
        alpha_A: float = 0.50,  
        alpha_S: float = 0.25,   
    ):

        if answer_true_or_false is None:
            return 

        def make_e(s: float) -> np.ndarray:
            s = max(s_min, min(s, 0.999))
            if answer_true_or_false is True:
                return np.array([1.0 - s, s], np.float32)
            else:
                return np.array([s, 1.0 - s], np.float32)

        k = num_steps - 1
        for t in range(k, -1, -1):
            base = s_max * (gamma ** (k - t))

            eO = make_e(base)
            gb.add_factor([f"O_{t}"], f"evidence_O_{t}", potential=eO)

            if inject_A:
                eA = make_e(base * alpha_A)
                gb.add_factor([f"A_{t}"], f"evidence_A_{t}", potential=eA)
            if inject_S:
                eS = make_e(base * alpha_S)
                gb.add_factor([f"S_{t}"], f"evidence_S_{t}", potential=eS)

    def _do_backward_lbp(self, builder : GraphBuilder, trajectory, golden_answer: Optional[str]) -> Tuple[Dict[str, Any], bool]:
        final_answer = trajectory.get("golden_answer")
        agent_answer = trajectory.get("final_answer")
        
        print(f"Final answer: {final_answer}, Golden answer: {golden_answer}, Agent answer: {agent_answer}")
        
        EVIDENCE_STRENGTHS = 0.9
        answer_true_or_false = compare_answers(expected_answer=golden_answer, agent_answer=agent_answer)
        if answer_true_or_false is True:
            evidence = np.array([1.0 - EVIDENCE_STRENGTHS, EVIDENCE_STRENGTHS], dtype=np.float32)
        elif answer_true_or_false is False:
            evidence = np.array([EVIDENCE_STRENGTHS, 1.0 - EVIDENCE_STRENGTHS], dtype=np.float32)
        
        answer_state = builder.get_answer_state()
        print("Answer state is :", answer_state)
        builder.add_factor([answer_state], f"evid_{answer_state}", potential=evidence)

        builder.do_lbp(init = True, normalize=True)
        
        posterior_marginals = {
            name: marg.tolist()
            for name, marg in builder.get_marginals(normalize=True).items()
        }
        
        return posterior_marginals, answer_true_or_false

    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(
            responses, 
            add_special_tokens=False, 
            return_tensors='pt', 
            padding="longest"
        )['input_ids']
        
    def truncate_to_confidence(self,text: str) -> str:

        end_conf_match = re.search(r'</confidence>', text)
        if not end_conf_match:
            return text.strip()

        end_pos = end_conf_match.end()  
        truncated = text[:end_pos]

        truncated = truncated.strip()

        return truncated

    def _postprocess_responses(self, responses: torch.Tensor) -> torch.Tensor:
        """Process responses to stop at search operation or answer operation."""
        responses_str = self.tokenizer.batch_decode(
            responses, 
            skip_special_tokens=True
        )

        # responses_str = [resp.split('</search>')[0] + '</search>'
        #          if '</search>' in resp 
        #          else resp.split('</answer>')[0] + '</answer>'
        #          if '</answer>' in resp 
        #          else resp
        #          for resp in responses_str]
        
        responses_str = [self.truncate_to_confidence(resp) for resp in responses_str]
        print("RESPONSES:", responses_str)

        if self.config.no_think_rl:
            raise ValueError('stop')
            # if no_think_rl is enabled, only keep action in the str
            actions, _ = self.env.postprocess_predictions(responses_str)
            responses_str=[f"<answer>{envs[idx].ACTION_LOOKUP[action]}</answer>" for idx, action in enumerate(actions)]
            print("RESPONSES:", responses_str)
        responses = self._batch_tokenize(responses_str)
        return responses, responses_str

    
    def _process_next_obs(self, next_obs: List[str], current_step: int) -> torch.Tensor:

        # -------- English formatting instruction (must-keep) --------
        format_instr = (
            "I must answer using this format:\n"
            "<think> my reasoning </think>\n"
            "<search> search query if needed </search>\n"
            "<answer> final answer if no search needed </answer>\n"
            "<confidence> 0~1 number Reflect how likely my confidence in answer or reasoning is correct </confidence>\n"
            "Exactly one of <search> or <answer> must appear.\n"
        )
        
        if current_step >= self.config.max_turns - 1:
            format_instr += "Since this is the final step, I must provide an <answer> and cannot provide a <search>.\n"


        next_obs = list(next_obs or [])

        instr_ids = self.tokenizer(
            [format_instr],
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"][0]                   # shape: [L_instr]
        L_instr = instr_ids.shape[0]
        max_L = int(self.config.max_obs_length)

        info_close = "</information>"
        info_close_ids = self.tokenizer(
            [info_close],
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"][0]                  # shape: [L_close]
        L_close = info_close_ids.shape[0]

        processed_ids = []
        for obs in next_obs:
            obs = (obs or "")

            obs_ids = self.tokenizer(
                [obs],
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"][0]              # shape: [L_obs_total]

            keep_L = max(0, max_L - L_instr)

            if obs_ids.shape[0] > keep_L:
                if keep_L >= L_close:
                    body_L = keep_L - L_close
                    body_ids = obs_ids[:body_L]
                    obs_ids = torch.cat([body_ids, info_close_ids], dim=0)
                else:

                    obs_ids = obs_ids[:keep_L]
            else:
                pass

            combined = torch.cat([obs_ids, instr_ids], dim=0)   # shape: <= max_L
            processed_ids.append(combined)

        if len(processed_ids) == 0:
            return torch.zeros((1, 1), dtype=torch.long)

        batch_max = min(max(x.shape[0] for x in processed_ids), max_L)

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = 0

        out = torch.full((len(processed_ids), batch_max), pad_id, dtype=torch.long)
        for i, ids in enumerate(processed_ids):
            L = min(ids.shape[0], batch_max)
            out[i, :L] = ids[:L]

        if batch_max == max_L:
            print(
                f"[WARNING] OBSERVATION was truncated to fit format instruction. "
                f"final_len={batch_max}, instr_len={L_instr}, max_len={max_L}"
            )

        return out


    def _update_rolling_state(self, rollings: DataProto, cur_responses: torch.Tensor, 
                            next_obs_ids: torch.Tensor) -> Dict:
        """Update rolling state with new responses and observations."""
        # Concatenate and handle padding        
        new_input_ids = self.tensor_fn.concatenate_with_padding([
            rollings.batch['input_ids'],
            cur_responses,
            next_obs_ids
        ])
        
        # Create attention mask and position ids
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # Cut to appropriate length
        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)

        new_rollings = DataProto.from_dict({
            'input_ids': new_input_ids[:, -max_len:],
            'position_ids': new_position_ids[:, -max_len:],
            'attention_mask': new_attention_mask[:, -max_len:]
        })
        new_rollings.meta_info.update(rollings.meta_info)
        
        return new_rollings

    def _info_masked_concatenate_with_padding(self, 
                prompt: torch.Tensor, 
                prompt_with_mask: torch.Tensor, 
                response: torch.Tensor, 
                info: Optional[torch.Tensor] = None,
                pad_to_left: bool = True
            ) -> torch.Tensor:
        """Concatenate tensors and handle padding. Additionally, create a mask (info_mask) to cover the information block if it exists."""
        pad_id = self.tokenizer.pad_token_id
        tensors = [prompt, response]
        tensors_with_mask = [prompt_with_mask, response]
        if info is not None:
            tensors.append(info)
            info_mask = torch.full(info.size(), pad_id, dtype=info.dtype, device=info.device) # information mask
            tensors_with_mask.append(info_mask)
        
        concatenated = torch.cat(tensors, dim=1)
        concatenated_with_info = torch.cat(tensors_with_mask, dim=1)
        mask = concatenated != pad_id if pad_to_left else concatenated == pad_id
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        padded_tensor = concatenated.gather(1, sorted_indices)
        padded_tensor_with_info = concatenated_with_info.gather(1, sorted_indices)

        return padded_tensor, padded_tensor_with_info

    def _update_right_side(self, right_side: Dict, 
                          cur_responses: torch.Tensor,
                          next_obs_ids: Optional[torch.Tensor] = None) -> Dict:
        """Update right side state."""
        if next_obs_ids != None:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    next_obs_ids, 
                    pad_to_left=False
                )
        else:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    pad_to_left=False
                )
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        return {'responses': responses[:, :max_len], 'responses_with_info_mask': responses_with_info_mask[:, :max_len]}

    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """
            Wrapper for generation that handles multi-GPU padding requirements.
            if num_gpus <= 1, return self.actor_rollout_wg.generate_sequences(active_batch)
            if active_batch size is not divisible by num_gpus, pad with first sequence
            then remove padding from output
        """
        if self.test_vllm_engine is not None:
            return self.test_vllm_engine.generate_sequences(active_batch)
            
        num_gpus = self.config.num_gpus
        if num_gpus <= 1:
            return self.actor_rollout_wg.generate_sequences(active_batch)
            
        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus
        
        for key in active_batch.batch.keys():
            active_batch.batch[key] = active_batch.batch[key].long()
        if remainder == 0:
            return self.actor_rollout_wg.generate_sequences(active_batch)
        
        # Add padding sequences
        padding_size = num_gpus - remainder
        padded_batch = {}
        
        for k, v in active_batch.batch.items():
            # Use first sequence as padding template
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)

        padded_active_batch = DataProto.from_dict(padded_batch)
        for key in padded_active_batch.batch.keys():
            padded_active_batch.batch[key] = padded_active_batch.batch[key].long()

        # Generate with padded batch
        padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)

        # Remove padding from output
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
        
        # Handle meta_info if present
        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size]
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta
            
        padded_output.batch = trimmed_batch
        return padded_output

    def run_llm_loop(self, gen_batch, initial_input_ids: torch.Tensor) -> DataProto:
        """Run main LLM generation loop."""

        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
        original_right_side = {'responses': initial_input_ids[:, []], 'responses_with_info_mask': initial_input_ids[:, []]}

        active_mask = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.bool)
        turns_stats = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        valid_action_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        valid_search_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        active_num_list = [active_mask.sum().item()]
        rollings : DataProto = gen_batch
        trajectory_buffers = self._initialize_trajectories(gen_batch, original_left_side['input_ids'])

        # Main generation loop
        for step in range(self.config.max_turns):
            if not active_mask.sum():
                break
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )
            
            # gen_output = self.actor_rollout_wg.generate_sequences(rollings)
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })      
            rollings_active.meta_info.update({k: v for k, v in rollings.meta_info.items()})
            rollings_active.non_tensor_batch = {k: v[active_mask] for k, v in rollings.non_tensor_batch.items()}      
            gen_output = self._generate_with_gpu_padding(rollings_active)

            meta_info = gen_output.meta_info            
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

            # log_prob_list: List[Optional[torch.Tensor]] = [None] * active_mask.shape[0]
            # if 'old_log_probs' in gen_output.batch:
            #     log_probs_active = gen_output.batch['old_log_probs'].detach().cpu()
            #     active_indices = torch.nonzero(active_mask, as_tuple=True)[0].tolist()
            #     for pos, batch_idx in enumerate(active_indices):
            #         log_prob_list[batch_idx] = log_probs_active[pos]
            
            parsed_segments: List[Dict[str, Optional[Any]]] = []

            # run logits path (Naive_rollout_path)
            if 'logits' in gen_output.batch:
                logits_list: List[Optional[torch.Tensor]] = [None] * active_mask.shape[0]
                logits_active = gen_output.batch['logits'].detach().cpu()
                active_indices = torch.nonzero(active_mask, as_tuple=True)[0].tolist()
                for pos, batch_idx in enumerate(active_indices):
                    logits_list[batch_idx] = logits_active[pos]
        
                for idx_resp, resp in enumerate(responses_str):
                    token_logits = logits_list[idx_resp]
                    parsed_segments.append(self._parse_response_segments(resp, responses_ids, token_logits))

            # run vllm_rollout_path 
            elif "rollout_log_probs" in gen_output.batch:
                # [B, T]
                token_confidence, _ = token_confidence_from_dataproto(gen_output)
                confidence_list: List[Optional[torch.Tensor]] = [None] * active_mask.shape[0]
                
                active_indices = torch.nonzero(active_mask, as_tuple=True)[0].tolist()
                for pos, batch_idx in enumerate(active_indices):
                    confidence_list[batch_idx] = token_confidence[pos]
                
                for idx_resp, resp in enumerate(responses_str):
                    token_conf = confidence_list[idx_resp]
                    parsed_segments.append(self._parse_response_segments_vllm(resp, responses_ids, token_conf))
                    
            else:
                raise ValueError("No logits or rollout_log_probs in gen_output")
                
            # Execute in environment and process observations
            next_obs, dones, valid_action, is_search = self.execute_predictions(
                responses_str, self.tokenizer.pad_token, active_mask
            )

            for idx, parsed in enumerate(parsed_segments):
                if idx >= len(trajectory_buffers):
                    break
                mask_value = active_mask[idx]
                if isinstance(mask_value, torch.Tensor):
                    is_active = bool(mask_value.item())
                else:
                    is_active = bool(mask_value)
                if not is_active:
                    continue
                observation_text = next_obs[idx] if idx < len(next_obs) else ""
                self._append_trajectory_step(
                    trajectory_buffers[idx],
                    parsed,
                    observation_text,
                    dones[idx] if idx < len(dones) else False,
                    responses_str[idx],
                )
            
            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            turns_stats[curr_active_mask] += 1
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)

            next_obs_ids = self._process_next_obs(next_obs, current_step = step)
            
            # Update states
            rollings = self._update_rolling_state(
                rollings,
                responses_ids,
                next_obs_ids
            )
            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
                next_obs_ids
            )
            
        # final LLM rollout
        if active_mask.sum():
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )
            
            # gen_output = self.actor_rollout_wg.generate_sequences(rollings)
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })      
            rollings_active.meta_info.update({k: v for k, v in rollings.meta_info.items()})
            rollings_active.non_tensor_batch = {k: v[active_mask] for k, v in rollings.non_tensor_batch.items()}      
            gen_output = self._generate_with_gpu_padding(rollings_active)

            meta_info = gen_output.meta_info            
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

            parsed_segments: List[Dict[str, Optional[Any]]] = []

            # run logits path (Naive_rollout_path)
            if 'logits' in gen_output.batch:
                logits_list: List[Optional[torch.Tensor]] = [None] * active_mask.shape[0]
                logits_active = gen_output.batch['logits'].detach().cpu()
                active_indices = torch.nonzero(active_mask, as_tuple=True)[0].tolist()
                for pos, batch_idx in enumerate(active_indices):
                    logits_list[batch_idx] = logits_active[pos]
        
                for idx_resp, resp in enumerate(responses_str):
                    token_logits = logits_list[idx_resp]
                    parsed_segments.append(self._parse_response_segments(resp, responses_ids, token_logits))

            # run vllm_rollout_path 
            elif "rollout_log_probs" in gen_output.batch:
                # [B, T]
                token_confidence, _ = token_confidence_from_dataproto(gen_output)
                confidence_list: List[Optional[torch.Tensor]] = [None] * active_mask.shape[0]
                
                active_indices = torch.nonzero(active_mask, as_tuple=True)[0].tolist()
                for pos, batch_idx in enumerate(active_indices):
                    confidence_list[batch_idx] = token_confidence[pos]
                
                for idx_resp, resp in enumerate(responses_str):
                    token_conf = confidence_list[idx_resp]
                    parsed_segments.append(self._parse_response_segments_vllm(resp, responses_ids, token_conf))
            else:
                raise ValueError("No logits or rollout_log_probs in gen_output")
            
            # Execute in environment and process observations
            next_obs, dones, valid_action, is_search = self.execute_predictions(
                responses_str, self.tokenizer.pad_token, active_mask
            )

            for idx, parsed in enumerate(parsed_segments):
                if idx >= len(trajectory_buffers):
                    break
                mask_value = active_mask[idx]
                if isinstance(mask_value, torch.Tensor):
                    is_active = bool(mask_value.item())
                else:
                    is_active = bool(mask_value)
                if not is_active:
                    continue
                observation_text = next_obs[idx] if idx < len(next_obs) else ""
                self._append_trajectory_step(
                    trajectory_buffers[idx],
                    parsed,
                    observation_text,
                    dones[idx] if idx < len(dones) else False,
                    responses_str[idx],
                )
            
            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            turns_stats[curr_active_mask] += 1
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)

            next_obs_ids = self._process_next_obs(next_obs, current_step = self.config.max_turns)
            
            # Update states
            rollings = self._update_rolling_state(
                rollings,
                responses_ids,
                next_obs_ids
            )
            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
                next_obs_ids
            )
        
        meta_info['turns_stats'] = turns_stats.tolist()
        meta_info['active_mask'] = active_mask.tolist()
        meta_info['valid_action_stats'] = valid_action_stats.tolist()
        meta_info['valid_search_stats'] = valid_search_stats.tolist()

       
        golden_answers = gen_batch.non_tensor_batch.get("answers", None)
        if golden_answers is None or all(x is None for x in golden_answers):
            golden_answers = gen_batch.non_tensor_batch.get("golden_answers", None)
        if golden_answers is None:
            golden_answers = [None] * len(trajectory_buffers)
        
        uids = gen_batch.non_tensor_batch.get("rollout_uid", None)

        import uuid
        if uids is None:
            uids = [str(uuid.uuid4()) for i in range(len(trajectory_buffers))]
        else:
            try:
                uids = list(uids)
            except TypeError:
                uids = [uids]
            uids = [str(u) for u in uids]

        if len(uids) < len(trajectory_buffers):
            pad = [uids[-1]] * (len(trajectory_buffers) - len(uids))
            uids = uids + pad
        elif len(uids) > len(trajectory_buffers):
            uids = uids[:len(trajectory_buffers)]

        sanitized_trajectories = []
        for i, (traj, golden_answer) in enumerate(zip(trajectory_buffers, golden_answers)):
            uid_i = uids[i]

            sanitized_steps = []
            for step in traj.get('steps', []):
                sanitized_steps.append({
                    'step': step.get('step'),
                    'state': step.get('state'),
                    'think': step.get('think'),
                    'action': step.get('action'),
                    'action_type': step.get('action_type'),
                    'observation': step.get('observation'),
                    'think_confidence': step.get('think_confidence'),
                    'action_confidence': step.get('action_confidence'),
                    'action_verbal_confidence': step.get('action_verbal_confidence'),
                    'format_ok': step.get('format_ok', False),
                    'done': step.get('done'),
                    'raw_response': step.get('raw_response'),
                })

            sanitized_trajectories.append({
                'rollout_uid': uid_i,  
                'question': traj.get('question'),
                'prompt': traj.get('prompt'),
                'steps': sanitized_steps,
                'final_answer': traj.get('final_answer'),
                'final_think_confidence': traj.get('final_think_confidence'),
                'final_answer_confidence': traj.get('final_answer_confidence'),
                'final_action_verbal_confidence': traj.get('final_action_verbal_confidence'),
                'golden_answer': golden_answer,
            })

        calibrations = []
        if self.graph_builder is not None or self.calibrate_rpc_client is not None:
            for i, (traj, golden_answer) in enumerate(zip(trajectory_buffers, golden_answers)):
                uid_i = uids[i]
                calibration = self._calibrate_trajectory(traj, golden_answer)
                if calibration is None:
                    calibration = {"rollout_uid": uid_i}
                elif isinstance(calibration, dict):
                    calibration = dict(calibration)
                    calibration.setdefault("rollout_uid", uid_i)
                else:
                    calibration = {"rollout_uid": uid_i, "raw": calibration}
                calibrations.append(calibration)
        else:
            calibrations = [{"rollout_uid": uids[i]} for i in range(len(sanitized_trajectories))]

        if len(calibrations) < len(sanitized_trajectories):
            for i in range(len(calibrations), len(sanitized_trajectories)):
                calibrations.append({"rollout_uid": uids[i]})
        elif len(calibrations) > len(sanitized_trajectories):
            calibrations = calibrations[:len(sanitized_trajectories)]

        print("ACTIVE_TRAJ_NUM:", active_num_list)

        non_tensor_batch = {}
        if getattr(gen_batch, "non_tensor_batch", None) is not None:
            for k, v in gen_batch.non_tensor_batch.items():
                if isinstance(v, np.ndarray):
                    non_tensor_batch[k] = v
                else:
                    non_tensor_batch[k] = np.array(v, dtype=object)

        non_tensor_batch["rollout_uid"] = np.array(uids, dtype=object)

        non_tensor_batch["trajectories"] = np.array(sanitized_trajectories, dtype=object)
        non_tensor_batch["graph_calibrations"] = np.array(calibrations, dtype=object)

        return self._compose_final_output(
            original_left_side,
            original_right_side,
            meta_info,
            non_tensor_batch,
        )
    def _compose_final_output(
        self,
        left_side: Dict,
        right_side: Dict,
        meta_info: Dict,
        non_tensor_batch: Dict,
    ) -> DataProto:
        """Compose final generation output."""
        final_output = right_side.copy()
        final_output['prompts'] = left_side['input_ids']
        
        # Combine input IDs
        final_output['input_ids'] = torch.cat([
            left_side['input_ids'],
            right_side['responses']
        ], dim=1)
        
        # Create attention mask and position ids
        final_output['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses'])
        ], dim=1)
        final_output['info_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses_with_info_mask'])
        ], dim=1)
        
        final_output['position_ids'] = self.tensor_fn.create_position_ids(
            final_output['attention_mask']
        )
        
        final_output_dp = DataProto.from_dict(final_output)
        final_output_dp.meta_info.update(meta_info)

        if non_tensor_batch:
            final_output_dp.non_tensor_batch = non_tensor_batch

        return final_output_dp

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type(
            (Exception, ValueError)
        ),
    )
    def execute_predictions(self, predictions: List[str], pad_token: str, active_mask=None, do_search=True):

        def _log(level: str, msg: str):
            try:
                lg = getattr(self, "logger", None) or globals().get("logger", None)
                if lg is not None and hasattr(lg, level):
                    getattr(lg, level)(msg)
            except Exception:
                pass 
        def _to_int01(x: Any) -> int:
            try:
                return 1 if bool(x) else 0
            except Exception:
                return 0

        def _safe_strip(s: Any) -> str:
            try:
                return (s or "").strip()
            except Exception:
                return ""

        try:
            cur_actions, contents = self.postprocess_predictions(predictions)
        except Exception as e:
            _log("exception", f"postprocess_predictions failed: {e}")
            n = len(predictions) if isinstance(predictions, list) else 0
            cur_actions = ["invalid"] * n
            contents = [""] * n

        n = max(len(cur_actions), len(contents))
        if len(cur_actions) != n:
            cur_actions = (cur_actions or []) + ["invalid"] * (n - len(cur_actions))
        if len(contents) != n:
            contents = (contents or []) + [""] * (n - len(contents))

        if not isinstance(active_mask, list):
            active_mask = [1] * n
        if len(active_mask) != n:
            _log("warning", f"active_mask len {len(active_mask)} != actions len {n}, auto-fixing.")
            if len(active_mask) < n:
                active_mask = active_mask + [1] * (n - len(active_mask))
            else:
                active_mask = active_mask[:n]
        active_mask = [_to_int01(x) for x in active_mask]

        search_indices = []
        search_queries = []
        try:
            for i, (a, act) in enumerate(zip(cur_actions, active_mask)):
                if act and (a == "search"):
                    q = _safe_strip(contents[i])
                    search_indices.append(i)
                    search_queries.append(q)
        except Exception as e:
            _log("exception", f"build search queries failed: {e}")
            search_indices, search_queries = [], []

        search_results = []
        try:
            if do_search and len(search_queries) > 0:
                try:
                    search_results = self.batch_search(search_queries)
                    if not isinstance(search_results, list):
                        _log("warning", "batch_search returned non-list, coercing to empty results.")
                        search_results = []
                except Exception as e:
                    _log("exception", f"batch_search failed: {e}")
                    search_results = []
            else:
                search_results = []
        except Exception as e:
            _log("exception", f"batch_search outer failed: {e}")
            search_results = []

        if len(search_results) < len(search_queries):
            search_results += [""] * (len(search_queries) - len(search_results))
        elif len(search_results) > len(search_queries):
            search_results = search_results[:len(search_queries)]

        sr_map = {}
        try:
            for i, res in zip(search_indices, search_results):
                sr_map[i] = _safe_strip(res)
        except Exception as e:
            _log("exception", f"build sr_map failed: {e}")
            sr_map = {}

        next_obs, dones, valid_action, is_search = [], [], [], []
        try:
            for i, (action, act) in enumerate(zip(cur_actions, active_mask)):
                if not act:
                    next_obs.append("")
                    dones.append(1)
                    valid_action.append(0)
                    is_search.append(0)
                    continue

                a = (action or "").strip().lower()
                if a == "answer":
                    next_obs.append("")
                    dones.append(1)
                    valid_action.append(1)
                    is_search.append(0)
                elif a == "search":
                    info = _safe_strip(sr_map.get(i, ""))
                    next_obs.append(f"\n\n<information>{info}</information>\n\n")
                    dones.append(0)
                    valid_action.append(1)
                    is_search.append(1)
                else:
                    next_obs.append(
                        "\nMy previous action is invalid. "
                        "If I want to search, I should put the query between <search> and </search>. "
                        "If I want to give the final answer, I should put the answer between <answer> and </answer>. "
                        "Let me try again.\n"
                    )
                    dones.append(0)
                    valid_action.append(0)
                    is_search.append(0)
        except Exception as e:
            _log("exception", f"build outputs failed midway: {e}")
            while len(next_obs) < n:
                next_obs.append("")
                dones.append(1)
                valid_action.append(0)
                is_search.append(0)

        L = max(len(next_obs), len(dones), len(valid_action), len(is_search), n)
        def _pad(lst, fill, L):
            return (lst or []) + [fill] * (L - len(lst))
        next_obs      = _pad(next_obs, "", L)
        dones         = _pad(dones, 1, L)
        valid_action  = _pad(valid_action, 0, L)
        is_search     = _pad(is_search, 0, L)

        return next_obs[:n], dones[:n], valid_action[:n], is_search[:n]

    def postprocess_predictions(self, predictions: List[Any]) -> Tuple[List[int], List[bool]]:
        """
        Process (text-based) predictions from llm into actions and validity flags.
        
        Args:
            predictions: List of raw predictions
            
        Returns:
            Tuple of (actions list, validity flags list)
        """
        actions = []
        contents = []
                
        for prediction in predictions:
            if isinstance(prediction, str): # for llm output
                pattern = r'<(search|answer)>(.*?)</\1>'
                match = re.search(pattern, prediction, re.DOTALL)
                if match:
                    content = match.group(2).strip()  # Return only the content inside the tags
                    action = match.group(1)
                else:
                    content = ''
                    action = None
            else:
                raise ValueError(f"Invalid prediction type: {type(prediction)}")
            
            actions.append(action)
            contents.append(content)
            
        return actions, contents

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type(
            (Exception, ValueError)
        ), 
    )
    def batch_search(self, queries: Optional[List[str]] = None) -> List[str]:
        """
        Batchified search for queries.
        Args:
            queries: queries to call the search engine
        Returns:
            search results which is concatenated into a string
        """
        if not self.config.online_search:
            try:
                results = self._batch_search(queries)['result']
                return [self._passages2string(result) for result in results]
            except Exception as e:
                return []
        else:
            if queries is not None:
                # here is a not sync function with result
                try:
                    results : List[List[Dict[str, str]]] = search_pipeline.run_batch_search(queries, self.online_search_config.to_dicts())
                    return search_pipeline.format_to_str(results)
                except Exception as e:
                    return []

    def _batch_search(self, queries):
        
        payload = {
            "queries": queries,
            "topk": self.config.topk,
            "return_scores": True
        }
        
        return requests.post(self.config.search_url, json=payload).json()

    def _passages2string(self, retrieval_result):
        return results_to_string(retrieval_result)
    
    def _is_format_valid(self, resp: str) -> bool:
        if not resp:
            return False
        text = resp.strip()

        has_think = bool(re.search(r"<think>.*?</think>", text, re.DOTALL))
        has_conf  = bool(re.search(r"<confidence>.*?</confidence>", text, re.DOTALL))

        has_search = bool(re.search(r"<search>.*?</search>", text, re.DOTALL))
        has_answer = bool(re.search(r"<answer>.*?</answer>", text, re.DOTALL))

        # exactly one of search/answer
        if not (has_think and has_conf):
            return False
        if has_search == has_answer:
            return False
        return True

