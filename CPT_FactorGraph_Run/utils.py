import sys
import os
os.environ['HF_ENDPOINT'] = ""
os.environ['HF_HOME'] = ''
os.environ['HF_HUB_CACHE'] = ""
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import transformers
import torch
from datasets import load_dataset
import requests
import re
from copy import deepcopy
from itertools import zip_longest
import torch.nn.functional as F

class StopOnSequence(transformers.StoppingCriteria):
    def __init__(self, target_sequences, tokenizer):
        self.target_ids = [tokenizer.encode(target_sequence, add_special_tokens=False) for target_sequence in target_sequences]
        self.target_lengths = [len(target_id) for target_id in self.target_ids]
        self._tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        targets = [torch.as_tensor(target_id, device=input_ids.device) for target_id in self.target_ids]
        if input_ids.shape[1] < min(self.target_lengths):
            return False
        for i, target in enumerate(targets):
            if torch.equal(input_ids[0, -self.target_lengths[i]:], target):
                return True
        return False
    
def get_confidence_safe(result, default=0.5):
    if result and "parts" in result and len(result["parts"]) > 0:
        return result["parts"][0].get("confidence", default)
    return default

def compare_answers(expected_answer: Optional[str | List[str]], agent_answer: str) -> bool:
    if not agent_answer or not expected_answer:
        return False
    
    #print(expected_answer)
    #print(type(expected_answer))

    def clean_text(text):
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text

    agent_clean = clean_text(agent_answer)

    if isinstance(expected_answer, list):
        for ans in expected_answer:
            expected_clean = clean_text(ans)
            if expected_clean in agent_clean or agent_clean in expected_clean:
                return True
        return False
    elif isinstance(expected_answer, np.ndarray):
        for ans in expected_answer.tolist():
            expected_clean = clean_text(ans)
            if expected_clean in agent_clean or agent_clean in expected_clean:
                return True
        return False
    else:
        expected_clean = clean_text(expected_answer)
        return expected_clean in agent_clean or agent_clean in expected_clean

def get_query(text):
    pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL)
    matches = pattern.findall(text)
    if matches:
        return matches[-1]
    else:
        return None

def get_answer(text):
    pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
    matches = pattern.findall(text)
    if matches:
        return matches[-1].strip()
    else:
        return None

def flatten_results(results):
    if isinstance(results, list) and results and isinstance(results[0], list):
        flat = []
        for sub in results:
            flat.extend(sub)
        return flat
    return results if isinstance(results, list) else []

def extract_title_text(doc):
    if isinstance(doc, str):
        lines = doc.splitlines()
        title = (lines[0].strip().strip('"\'')) if lines else ""
        text = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""
        return title, text

    if isinstance(doc, dict):
        title = (doc.get("title") or "").strip()
        contents = doc.get("contents") or doc.get("content") or ""
        if not title:
            if isinstance(contents, str) and contents:
                lines = contents.splitlines()
                title = (lines[0].strip().strip('"\'')) if lines else ""
                text = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""
            else:
                text = ""
        else:
            text = contents.strip() if isinstance(contents, str) else ""
        return title, text

    return "", str(doc)

def results_to_string(results):
    flat = flatten_results(results)
    out = []
    for i, item in enumerate(flat, 1):
        doc = item.get("document")
        score = item.get("score")
        title, text = extract_title_text(doc)
        header = f"Doc {i} (Title: {title})"
        if isinstance(score, (int, float)):
            header += f" [Score: {score:.3f}]"
        block = header + ("\n" + text if text else "")
        out.append(block)
    return "\n\n".join(out)

def search(query: str, topk: int = 3):
    payload = {
        "queries": [query],
        "topk": topk,
        "return_scores": True
    }
    results = requests.post("http://127.0.0.1:8100/retrieve", json=payload).json()['result']
    return results_to_string(results)

def extract_triples(text: str) -> List[Dict[str, Optional[str]]]:
    think_list = re.findall(r"<think>(.*?)</think>", text, re.S)
    confidence_list = re.findall(r"<confidence>(.*?)</confidence>", text, re.S)
    search_list = re.findall(r"<search>(.*?)</search>", text, re.S)
    answer_list = re.findall(r"<answer>(.*?)</answer>", text, re.S)

    results = []
    si, ai = 0, 0
    for i in range(len(confidence_list)):
        think = think_list[i].strip() if i < len(think_list) else None
        confidence = confidence_list[i].strip()
        if si < len(search_list):
            action = search_list[si].strip()
            action_type = "search"
            si += 1
        elif ai < len(answer_list):
            action = answer_list[ai].strip()
            action_type = "answer"
            ai += 1
        else:
            action = None
            action_type = None
        results.append({
            "think": think,
            "confidence": confidence,
            "action": action,
            "action_type": action_type
        })
    return results

def extract_triples_robust(text: str) -> List[Dict[str, Optional[str]]]:
    flags = re.S | re.I  

    def grab(tag: str):
        return [m.strip() for m in re.findall(fr"<{tag}>(.*?)</{tag}>", text, flags=flags)]

    thinks      = grab("think")
    confidences = grab("confidence")
    searches    = grab("search")
    answers     = grab("answer")

    results: List[Dict[str, Optional[str]]] = []

    for t, c, s, a in zip_longest(thinks, confidences, searches, answers, fillvalue=None):
        action, action_type = (s, "search") if s is not None else ((a, "answer") if a is not None else (None, None))
        results.append({
            "think": t,
            "confidence": c,
            "action": action,
            "action_type": action_type
        })
    return results


def fix_tags_tokenizer_verbal(in_tokenizer, text: str) -> str:
    # special_tokens = ["<think>", "</think>", "<confidence>", "</confidence>",
    #               "<answer>", "</answer>", "<search>", "</search>"]
    
    tokenizer = deepcopy(in_tokenizer)
    if "<answer>" in text or "</answer>" in text:
        tags = ["think", "confidence", "answer"]
    else:
        tags = ["think", "confidence", "search"]
        
    special_tokens = [f"<{tag}>" for tag in tags] + [f"</{tag}>" for tag in tags]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    input_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    token_ids = {tok: tokenizer.convert_tokens_to_ids(tok) for tok in special_tokens}

    positions = {}
    for tag in tags:
        open_id, close_id = token_ids[f"<{tag}>"], token_ids[f"</{tag}>"]
        open_pos = [i for i, t in enumerate(input_ids) if t == open_id]
        close_pos = [i for i, t in enumerate(input_ids) if t == close_id]
        positions[tag] = (open_pos, close_pos)

    new_ids = []
    last_pos = 0  

    for i, tag in enumerate(tags):
        open_id, close_id = token_ids[f"<{tag}>"], token_ids[f"</{tag}>"]
        open_pos_list, close_pos_list = positions[tag]

        start = next((p for p in open_pos_list if p >= last_pos), -1)
        end = next((p for p in close_pos_list if p > start), -1)

        if start == -1 and end != -1:
            content_ids = input_ids[last_pos:end]
            new_ids += [open_id] + content_ids + [close_id]
            last_pos = end + 1
        elif start != -1 and end == -1:
            next_tag_start = len(input_ids)
            for j in range(i + 1, len(tags)):
                next_open_id = token_ids[f"<{tags[j]}>"]
                next_pos = next((p for p, t in enumerate(input_ids[last_pos:], last_pos) if t == next_open_id), len(input_ids))
                next_tag_start = min(next_tag_start, next_pos)
            content_ids = input_ids[start + 1:next_tag_start]
            new_ids += [open_id] + content_ids + [close_id]
            last_pos = next_tag_start
        elif start == -1 and end == -1:
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            line_content = lines[0] if lines else ""
            content_ids = tokenizer(line_content, add_special_tokens=False)["input_ids"]
            new_ids += [open_id] + content_ids + [close_id]
        else:
            content_ids = input_ids[start + 1:end]
            new_ids += [open_id] + content_ids + [close_id]
            last_pos = end + 1

    return tokenizer.decode(new_ids)

def fix_tags_tokenizer_non_verbal(in_tokenizer, text: str) -> str:
    # special_tokens = ["<think>", "</think>", "<confidence>", "</confidence>",
    #               "<answer>", "</answer>", "<search>", "</search>"]
    
    tokenizer = deepcopy(in_tokenizer)
    if "<answer>" in text or "</answer>" in text:
        tags = ["think", "answer"]
    else:
        tags = ["think", "search"]
        
    special_tokens = [f"<{tag}>" for tag in tags] + [f"</{tag}>" for tag in tags]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    input_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    token_ids = {tok: tokenizer.convert_tokens_to_ids(tok) for tok in special_tokens}

    positions = {}
    for tag in tags:
        open_id, close_id = token_ids[f"<{tag}>"], token_ids[f"</{tag}>"]
        open_pos = [i for i, t in enumerate(input_ids) if t == open_id]
        close_pos = [i for i, t in enumerate(input_ids) if t == close_id]
        positions[tag] = (open_pos, close_pos)

    new_ids = []
    last_pos = 0  

    for i, tag in enumerate(tags):
        open_id, close_id = token_ids[f"<{tag}>"], token_ids[f"</{tag}>"]
        open_pos_list, close_pos_list = positions[tag]

        start = next((p for p in open_pos_list if p >= last_pos), -1)
        end = next((p for p in close_pos_list if p > start), -1)

        if start == -1 and end != -1:
            content_ids = input_ids[last_pos:end]
            new_ids += [open_id] + content_ids + [close_id]
            last_pos = end + 1
        elif start != -1 and end == -1:
            next_tag_start = len(input_ids)
            for j in range(i + 1, len(tags)):
                next_open_id = token_ids[f"<{tags[j]}>"]
                next_pos = next((p for p, t in enumerate(input_ids[last_pos:], last_pos) if t == next_open_id), len(input_ids))
                next_tag_start = min(next_tag_start, next_pos)
            content_ids = input_ids[start + 1:next_tag_start]
            new_ids += [open_id] + content_ids + [close_id]
            last_pos = next_tag_start
        elif start == -1 and end == -1:
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            line_content = lines[0] if lines else ""
            content_ids = tokenizer(line_content, add_special_tokens=False)["input_ids"]
            new_ids += [open_id] + content_ids + [close_id]
        else:
            content_ids = input_ids[start + 1:end]
            new_ids += [open_id] + content_ids + [close_id]
            last_pos = end + 1

    return tokenizer.decode(new_ids)

EPS = 1e-6

def make_identity_obs(eps: float = 1e-6) -> np.ndarray:
    P = np.array([[1.0, 0.0],
                  [0.0, 1.0]], dtype=np.float64)
    P = (1 - 2*eps) * P + eps           # [[1-eps, eps], [eps, 1-eps]]
    P = P / P.sum(axis=1, keepdims=True)
    return P.astype(np.float32)

def row_to_symmetric_cpt(row: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    row = np.asarray(row, dtype=np.float64)
    if row.shape != (2,):
        raise ValueError("row 必须是 (2,) 形状，如 [0.2, 0.8]")
    row = np.clip(row, eps, 1-eps); row = row / row.sum()
    cpt = np.stack([row[::-1], row], axis=0)
    cpt = cpt / cpt.sum(axis=1, keepdims=True)
    return cpt.astype(np.float32)

def ensure_cpt_2x2(pot: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    arr = np.asarray(pot, dtype=np.float64)
    if arr.shape == (2,):
        arr = row_to_symmetric_cpt(arr, eps=eps)
    elif arr.shape == (2,2):
        arr = np.clip(arr, eps, 1-eps)
        arr = arr / arr.sum(axis=1, keepdims=True)
    else:
        raise ValueError(f"期望 (2,) 或 (2,2)，得到 {arr.shape}")
    return arr.astype(np.float32)

def _to_float01(x, default=0.5):
    try:
        if x is None:
            return float(default)
        if isinstance(x, (int, float)):
            v = float(x)
            return float(np.clip(v, 0.0, 1.0))
        s = str(x).strip()
        if s.endswith("%"):
            v = float(s[:-1].strip()) / 100.0
        else:
            v = float(s)
        return float(np.clip(v, 0.0, 1.0))
    except Exception:
        return float(default)

def _encode_seq(tokenizer, text: str) -> List[int]:
    return tokenizer.encode(text, add_special_tokens=False)

def _find_all_occurrences(seq: List[int], pat: List[int]) -> List[int]:
    if len(pat) == 0 or len(seq) < len(pat):
        return []
    res = []
    Ls, Lp = len(seq), len(pat)
    for i in range(Ls - Lp + 1):
        if seq[i:i+Lp] == pat:
            res.append(i)
    return res

def _pair_spans(open_pos: List[int], close_pos: List[int], open_len: int, close_len: int) -> List[Tuple[int, int]]:
    open_pos = sorted(open_pos)
    close_pos = sorted(close_pos)
    spans = []
    stack = []

    events = []
    for p in open_pos:
        events.append((p, "open"))
    for p in close_pos:
        events.append((p, "close"))
    events.sort(key=lambda x: (x[0], 0 if x[1]=="open" else 1))

    for pos, typ in events:
        if typ == "open":
            stack.append(pos)
        else:
            if not stack:
                continue
            s = stack.pop()               
            e = pos                       
            start_idx = s + open_len      
            end_idx = e                   
            if start_idx < end_idx:
                spans.append((start_idx, end_idx))
    spans.sort()
    return spans


def _find_prefix_hits(ids: List[int], prefix_ids: List[int]) -> List[int]:
    if not prefix_ids or len(ids) < len(prefix_ids):
        return []
    hits = []
    Ls, Lp = len(ids), len(prefix_ids)
    for i in range(Ls - Lp + 1):
        if ids[i:i+Lp] == prefix_ids:
            hits.append(i)
    return hits

def _min_tokens_to_cover_tag(tokenizer, ids: List[int], start: int, tag: str, lookahead: int = 8) -> Optional[int]:

    for n in range(1, lookahead + 1):
        s = tokenizer.decode(ids[start:start+n], skip_special_tokens=False)
        if s.startswith(tag):
            return n
    return None

def _pair_by_stack(opens: List[Tuple[int, int]], closes: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    events = []
    for pos, cons in opens:
        events.append((pos, "open", cons))
    for pos, cons in closes:
        events.append((pos, "close", cons))
    events.sort(key=lambda x: (x[0], 0 if x[1] == "open" else 1))

    spans, stack = [], []
    for pos, typ, cons in events:
        if typ == "open":
            stack.append((pos, cons))
        else:
            if not stack:
                continue
            o_pos, o_cons = stack.pop()
            start = o_pos + o_cons
            end = pos
            if start < end:
                spans.append((start, end))
    spans.sort()
    return spans


def split_tokens_and_scores(
    tokenizer,
    generated_tokens: List[int],
    token_scores: List[torch.Tensor],   
    special_tokens: List[str],
) -> Optional[Dict[str, Any]]:

    if len(special_tokens) != 2:
        return None
    open_tag, close_tag = special_tokens
    ids = list(generated_tokens)

    open_prefix_ids = tokenizer.encode(open_tag[:-1], add_special_tokens=False)
    close_prefix_ids = tokenizer.encode(close_tag[:-1], add_special_tokens=False)

    open_hits = _find_prefix_hits(ids, open_prefix_ids)
    close_hits = _find_prefix_hits(ids, close_prefix_ids)
    if not open_hits or not close_hits:
        return None

    opens, closes = [], []
    for pos in open_hits:
        n = _min_tokens_to_cover_tag(tokenizer, ids, pos, open_tag)
        if n is not None:
            opens.append((pos, n))
    for pos in close_hits:
        n = _min_tokens_to_cover_tag(tokenizer, ids, pos, close_tag)
        if n is not None:
            closes.append((pos, n))

    spans = _pair_by_stack(opens, closes)
    if not spans:
        return None

    start_idx, end_idx = spans[0]  
    tokens, texts, logprobs, probs = [], [], [], []

    for t in range(start_idx, end_idx):
        vec = token_scores[t]
        if vec.dim() == 2:
            vec = vec[0]
        logp_all = F.log_softmax(vec, dim=-1)
        p_all = torch.exp(logp_all)
        tok_id = ids[t]
        tokens.append(tok_id)
        texts.append(tokenizer.decode([tok_id], skip_special_tokens=False))
        logprobs.append(float(logp_all[tok_id].item()))
        probs.append(float(p_all[tok_id].item()))

    return {
        "tokens": tokens,
        "texts": texts,
        "logprobs": logprobs,
        "probs": probs,
        "span": (start_idx, end_idx),
    }


def split_tokens_and_scores_with_confidence(
    tokenizer,
    generated_tokens: List[int],
    token_scores: List[torch.Tensor],   
    special_tokens: List[str],
) -> Optional[Dict[str, Any]]:
    if len(special_tokens) != 2:
        return None
    open_tag, close_tag = special_tokens
    ids = list(generated_tokens)

    open_prefix_ids = tokenizer.encode(open_tag[:-1], add_special_tokens=False)
    close_prefix_ids = tokenizer.encode(close_tag[:-1], add_special_tokens=False)

    open_hits = _find_prefix_hits(ids, open_prefix_ids)
    close_hits = _find_prefix_hits(ids, close_prefix_ids)
    if not open_hits or not close_hits:
        return None

    opens, closes = [], []
    for pos in open_hits:
        n = _min_tokens_to_cover_tag(tokenizer, ids, pos, open_tag)
        if n is not None:
            opens.append((pos, n))
    for pos in close_hits:
        n = _min_tokens_to_cover_tag(tokenizer, ids, pos, close_tag)
        if n is not None:
            closes.append((pos, n))

    spans = _pair_by_stack(opens, closes)
    if not spans:
        return None

    parts = []
    for (start_idx, end_idx) in spans:
        tokens, texts, logprobs, probs = [], [], [], []
        for t in range(start_idx, end_idx):
            vec = token_scores[t]
            if vec.dim() == 2:
                vec = vec[0]
            logp_all = F.log_softmax(vec, dim=-1)
            p_all = torch.exp(logp_all)
            tok_id = ids[t]
            tokens.append(tok_id)
            texts.append(tokenizer.decode([tok_id], skip_special_tokens=False))
            logprobs.append(float(logp_all[tok_id].item()))
            probs.append(float(p_all[tok_id].item()))

        neg_logps = [-lp for lp in logprobs]
        confidence = float(np.mean(neg_logps)) if neg_logps else None

        parts.append({
            "span": (start_idx, end_idx),
            "tokens": tokens,
            "texts": texts,
            "logprobs": logprobs,
            "probs": probs,
            "confidence": confidence,
        })

    return {"parts": parts}


def split_tokens_and_logits_with_probs(
    tokenizer,
    generated_tokens: List[int],
    token_logits: List[torch.Tensor],  
) -> Optional[Dict[str, Any]]:

    if len(special_tokens) != 2:
        return None
    open_tag, close_tag = special_tokens
    ids = list(generated_tokens)

    open_prefix_ids = tokenizer.encode(open_tag[:-1], add_special_tokens=False)
    close_prefix_ids = tokenizer.encode(close_tag[:-1], add_special_tokens=False)

    open_hits = _find_prefix_hits(ids, open_prefix_ids)
    close_hits = _find_prefix_hits(ids, close_prefix_ids)
    if not open_hits or not close_hits:
        return None

    opens, closes = [], []
    for pos in open_hits:
        n = _min_tokens_to_cover_tag(tokenizer, ids, pos, open_tag)
        if n is not None:
            opens.append((pos, n))
    for pos in close_hits:
        n = _min_tokens_to_cover_tag(tokenizer, ids, pos, close_tag)
        if n is not None:
            closes.append((pos, n))

    spans = _pair_by_stack(opens, closes)
    if not spans:
        return None

    parts = []
    for (start_idx, end_idx) in spans:
        tokens, texts, logits_list, probs_list = [], [], [], []
        for t in range(start_idx, end_idx):
            vec = token_logits[t]
            if vec.dim() == 2:
                vec = vec[0]
            probs = F.softmax(vec, dim=-1)
            tok_id = ids[t]
            tokens.append(tok_id)
            texts.append(tokenizer.decode([tok_id], skip_special_tokens=False))
            logits_list.append(float(vec[tok_id].item()))
            probs_list.append(float(probs[tok_id].item()))

        if probs_list:
            neg_logps = [-float(np.log(max(p, 1e-12))) for p in probs_list]
            confidence = float(sum(neg_logps) / len(neg_logps))
        else:
            confidence = None

        parts.append({
            "span": (start_idx, end_idx),
            "tokens": tokens,
            "texts": texts,
            "logits": logits_list,
            "probs": probs_list,
            "confidence": confidence,
        })

    return {"parts": parts}

def extract_tagged_spans_with_probs(
    tokenizer,
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
                s = tokenizer.decode(ids[pos:pos+k], skip_special_tokens=False)
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
    for start,end in spans:
        toks, texts, logits_list, probs_list = [], [], [], []
        for t in range(start,end):
            vec = token_logits[t]
            if vec.dim()==2: vec = vec[0]
            probs = F.softmax(vec, dim=-1)
            tok_id = ids[t]
            toks.append(tok_id)
            texts.append(tokenizer.decode([tok_id], skip_special_tokens=False))
            logits_list.append(float(vec[tok_id].item()))
            probs_list.append(float(probs[tok_id].item()))

        if probs_list:
            nll = [-np.log(max(p,1e-12)) for p in probs_list]
            confidence = float(np.mean(nll))
        else:
            confidence = None

        results.append({
            "span": (start,end),
            "tokens": toks,
            "texts": texts,
            "logits": logits_list,
            "probs": probs_list,
            "confidence": confidence,
        })

    return {"parts": results}

def compute_token_confidence(
    step_logprob_dicts: List[Dict[int, Any]],
    token_ids: List[int],
    k: Optional[int] = 4,
    method: str = "top1",  
) -> List[Tuple[float, dict]]:
    results: List[Tuple[float, dict]] = []
    ln2 = math.log(2)

    for t, d in enumerate(step_logprob_dicts):
        if not d:
            results.append((float("nan"), {"k_used": 0}))
            continue

        pairs = []
        for tid, v in d.items():
            lp = getattr(v, "logprob", v)
            if lp is not None:
                pairs.append((int(tid), float(lp)))
        if not pairs:
            results.append((float("nan"), {"k_used": 0}))
            continue

        pairs.sort(key=lambda x: x[1], reverse=True)
        k_used = len(pairs) if k is None else max(1, min(k, len(pairs)))
        topk = pairs[:k_used]
        topk_logp = [lp for _, lp in topk]
        topk_p = [math.exp(lp) for lp in topk_logp]

        sel_tid = token_ids[t] if t < len(token_ids) else None
        sel_logp = None
        if sel_tid is not None:
            v_sel = d.get(sel_tid, None)
            if v_sel is not None:
                sel_logp = getattr(v_sel, "logprob", v_sel)

        score = float("nan")

        if method == "top1":
            score = math.exp(sel_logp) if sel_logp is not None else 0.0

        elif method == "margin":
            if sel_logp is None:
                score = -float("inf")
            else:
                sec_logp = topk_logp[1] if k_used >= 2 else -float("inf")
                score = float(sel_logp) - sec_logp

        elif method == "mass":
            score = sum(topk_p)

        elif method == "entropy":

            if k_used == 1:
                score = 1.0
            else:
                H = -sum(p * (lp) for p, lp in zip(topk_p, topk_logp))  # 以 e 为底
                score = 1.0 - H / math.log(k_used)

        elif method == "geom_mean":
            avg_logp = sum(topk_logp) / k_used
            score = math.exp(avg_logp)

        else:
            raise ValueError(f"Unknown method: {method}")

        results.append((score, {
            "k_used": k_used,
            "sel_logp": sel_logp,
            "top1_logp": topk_logp[0],
            "top2_logp": topk_logp[1] if k_used >= 2 else None,
            "mass_topk": sum(topk_p),
        }))

    return results


def token_confidence_from_dataproto(
    dp, 
    pad_fill_value: float = -1.0,
    eos_token_id: Optional[int] = None,  
) -> Tuple[torch.Tensor, torch.Tensor]:

    batch = dp.batch 
    device = batch["responses"].device
    responses: torch.Tensor = batch["responses"]  

    if "rollout_log_probs" in batch.keys():
        logp: torch.Tensor = batch["rollout_log_probs"]  
        valid_mask = logp.ne(pad_fill_value)
        logp = torch.where(valid_mask, logp, torch.full_like(logp, -1e9))
        probs = torch.exp(logp).to(torch.float32)  # [B, T]
        return probs, valid_mask

    elif "logits" in batch.keys():
        logits: torch.Tensor = batch["logits"]          # [B, T, V]
        probs_full = torch.softmax(logits, dim=-1)      # [B, T, V]
        gather_idx = responses.unsqueeze(-1)            # [B, T, 1]
        probs = probs_full.gather(-1, gather_idx).squeeze(-1).to(torch.float32)  # [B, T]

        valid_mask = torch.ones_like(responses, dtype=torch.bool, device=device)

        if eos_token_id is not None:
            B, T = responses.size()
            is_eos = (responses == eos_token_id)        # [B, T]
            eos_pos = torch.where(
                is_eos.any(dim=1, keepdim=True),
                is_eos.float().argmax(dim=1, keepdim=True),               
                torch.full((B, 1), T, device=device, dtype=torch.long),   
            )  # [B,1]
            steps = torch.arange(T, device=device).unsqueeze(0).expand(B, T)  # [B, T]
            after_eos = steps >= eos_pos
            valid_mask = valid_mask & (~after_eos)

            probs = torch.where(valid_mask, probs, torch.zeros_like(probs))

        return probs, valid_mask

    else:
        raise KeyError(
            "DataProto.batch 中既没有 'rollout_log_probs' 也没有 'logits'；"
            "无法计算 token 置信度。请确认上游是否开启了 logprobs 或返回 logits。"
        )
