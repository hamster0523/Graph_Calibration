"""
Unified termination policy system for MCTS agent.
Provides clean separation of termination logic and reward calculation.
"""
from dataclasses import dataclass
from typing import Optional, Protocol, Tuple, List
from abc import ABC, abstractmethod
import re
from app.schema import Memory
from app.logger import logger

@dataclass
class TerminationOutcome:
    """Result of termination judgment"""
    terminated: bool
    reward: float = 0.0              # Terminal reward (normalized)
    reason: str = ""                 # Human readable reason
    success: Optional[bool] = None   # Optional: whether successful

class TerminationPolicy:
    """Single decision point: determines termination and reward based on tool name, result, depth, and state"""
    
    def decide(self, *, tool_name: Optional[str], result: str, depth: int, max_depth: int, state: Memory) -> TerminationOutcome:
        """
        Decide whether to terminate and calculate reward
        
        Args:
            tool_name: Name of the tool used (if any)
            result: Tool execution result
            depth: Current node depth
            max_depth: Maximum allowed depth
            state: Current memory state
            
        Returns:
            TerminationOutcome with termination decision and reward
        """
        # Default policy: only terminate on max depth; no reward
        if depth >= max_depth:
            return TerminationOutcome(True, 0.5, reason="max_depth", success=None)
        
        # Check for keyword-based termination (fallback strategy)
        result_lower = result.lower()
        if any(keyword in result_lower for keyword in ["success", "completed", "finished"]):
            return TerminationOutcome(True, 1.0, reason="success_keyword", success=True)
        elif any(keyword in result_lower for keyword in ["failed", "error", "impossible"]):
            return TerminationOutcome(True, 0.1, reason="failure_keyword", success=False)

        if tool_name and tool_name.lower() == "terminate":
            return TerminationOutcome(True, 1.0, reason="terminate_called", success=True)

        return TerminationOutcome(False)

from typing import Optional, Tuple, Union, List
from collections import Counter
import re
import string
import unicodedata

class HotpotQATerminationPolicy(TerminationPolicy):
    """HotpotQA task-specific termination policy using EM/F1 evaluation (answers from state)"""
    
    def __init__(self, gold_answer: Union[str, List[str]]):
        """
        Args:
            gold_answer: 标准答案字符串，或多个可接受答案的列表
        """
        self.gold_answer = gold_answer

    # -------- 核心：从 state 中提取最近一次 create_chat_completion 的答案 --------
    def _extract_answer_from_state(self, state: Memory) -> Optional[str]:
        """
        在 state.messages 中逆序查找最近一条:
          - role == "tool"
          - name == "create_chat_completion"
        的消息，并从其 content 中抽取自然语言答案。
        """
        if not state or not getattr(state, "messages", None):
            return None
        
        # 逆序遍历，找到最近的 create_chat_completion 工具输出
        for msg in reversed(state.messages):
            # 你的 Message 结构看起来类似：Message(role, content, name=tool_name, ...)
            if getattr(msg, "role", None) == "tool" and getattr(msg, "name", "") == "create_chat_completion":
                raw = getattr(msg, "content", "") or ""
                if not raw:
                    continue
                # 去掉框架前缀，例如：Observed output of cmd `create_chat_completion` executed:\n...
                raw = re.sub(
                    r"^Observed output of cmd `?create_chat_completion`? executed:\s*\n", 
                    "", raw, flags=re.IGNORECASE
                ).strip()
                # 有些实现会把答案直接就是 raw，也有的会带 "Answer:" / "Final answer:" 等
                ans = self._extract_answer_like_human(raw)
                if ans:
                    return ans
        # 若没找到工具输出，可选：回退到最近一条 assistant 明文答案（非必须）
        for msg in reversed(state.messages):
            if getattr(msg, "role", None) == "assistant":
                text = (getattr(msg, "content", "") or "").strip()
                if text:
                    ans = self._extract_answer_like_human(text)
                    if ans:
                        return ans
        return None

    # -------- 更鲁棒的“从自然语言段落里抠出最终答案”的启发式 --------
    def _extract_answer_like_human(self, text: str) -> str:
        """
        先匹配 'Answer:' / 'Final answer:' / '答案：' 等关键词；否则取最后一句/最后一行。
        你也可以把 Agent 最终回答统一标准化为结构化 JSON，这里就不需要启发式了。
        """
        # 1) 常见前缀
        m = re.search(r"(?:final\s*answer|answer|答案)\s*[:：]\s*(.+)$", text, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).strip()
        # 2) 取最后一行/最后一句（尽量不拿“References/Citations”之类）
        lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
        if lines:
            cand = lines[-1]
            # 若最后一行是引用或提示，尝试上一行
            if re.search(r"^(references|citations|source|结束|终止)\b", cand, re.IGNORECASE):
                if len(lines) >= 2:
                    cand = lines[-2]
            # 切句号兜底
            sentences = re.split(r"[。\.!?！？]+", cand)
            if sentences:
                return sentences[-1].strip() or cand
        return text.strip()

    # -------- 规范化 + EM/F1 计算（F1 用多重计数交集）--------
    def _normalize(self, s: str) -> str:
        s = unicodedata.normalize("NFKC", s.lower().strip())
        # 去冠词（英文），保留最常见规则；中文一般无此步
        s = re.sub(r"\b(a|an|the)\b", " ", s)
        # 去标点
        s = re.sub("[" + re.escape(string.punctuation) + "]", " ", s)
        # 合并空白
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _em_f1_pair(self, pred: str, gold: str) -> Tuple[float, float]:
        pred_norm = self._normalize(pred)
        gold_norm = self._normalize(gold)

        # EM
        em = 1.0 if pred_norm == gold_norm else 0.0

        # F1（多重词频交集）
        if not pred_norm or not gold_norm:
            return em, 0.0
        p_toks = pred_norm.split()
        g_toks = gold_norm.split()
        if not p_toks or not g_toks:
            return em, 0.0

        p_cnt = Counter(p_toks)
        g_cnt = Counter(g_toks)
        common = sum((p_cnt & g_cnt).values())
        if common == 0:
            return em, 0.0
        precision = common / len(p_toks)
        recall = common / len(g_toks)
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        return em, f1

    def _compute_em_f1(self, pred: str, gold: Union[str, List[str]]) -> Tuple[float, float]:
        """
        gold 可为 string 或候选列表：返回与任一 gold 的最佳 (EM, F1)。
        """
        if isinstance(gold, str):
            return self._em_f1_pair(pred, gold)
        best_em, best_f1 = 0.0, 0.0
        for g in gold:
            em, f1 = self._em_f1_pair(pred, g)
            if (f1 > best_f1) or (f1 == best_f1 and em > best_em):
                best_em, best_f1 = em, f1
        return best_em, best_f1

    # --------------------- 终止策略 ---------------------
    def decide(self, *, tool_name: Optional[str], result: str, depth: int, max_depth: int, state: Memory) -> TerminationOutcome:
        """
        HotpotQA-specific termination decision
        仅当 tool_name == "terminate" 时做评估；答案来自 state（而非 result）。
        """
        # 1) 仅在显式终止时评估
        if tool_name and tool_name.lower() == "terminate":
            try:
                pred = self._extract_answer_from_state(state)
                if not pred:
                    logger.warning("HotpotQA: 未能从 state 中提取 create_chat_completion 的答案。")
                    # 没有答案但已终止：给个温和奖励，提示流程需修正
                    return TerminationOutcome(True, 0.2, reason="terminate_no_pred_in_state", success=None)

                gold = self.gold_answer
                if gold:
                    em, f1 = self._compute_em_f1(pred, gold)
                    reward = max(0.0, min(1.0, 0.7 * f1 + 0.3 * em))
                    return TerminationOutcome(True, reward, reason="terminate_with_eval", success=(em == 1.0))
                else:
                    return TerminationOutcome(True, 0.2, reason="terminate_no_gold", success=None)
            except Exception as e:
                logger.warning(f"Error in HotpotQA evaluation: {e}")
                return TerminationOutcome(True, 0.1, reason="terminate_eval_error", success=False)

        # 2) 深度截断
        if depth >= max_depth:
            return TerminationOutcome(True, 0.0, reason="max_depth", success=None)

        return TerminationOutcome(False)


class GSM8KTerminationPolicy(TerminationPolicy):
    """GSM8K math problem termination policy"""
    
    def __init__(self, gold_answer_getter):
        self.get_gold = gold_answer_getter
    
    def decide(self, *, tool_name: Optional[str], result: str, depth: int, max_depth: int, state: Memory) -> TerminationOutcome:
        if tool_name and tool_name.lower() == "terminate":
            pred_num = self._extract_number_from_result(result)
            gold_num = self._extract_number_from_text(self.get_gold(state))
            
            if pred_num is not None and gold_num is not None:
                # Exact numerical match
                reward = 1.0 if abs(pred_num - gold_num) < 1e-6 else 0.0
                return TerminationOutcome(True, reward, reason="math_eval", success=(reward == 1.0))
            else:
                # Fallback to text comparison
                reward = 0.5 if pred_num is not None else 0.0
                return TerminationOutcome(True, reward, reason="math_partial", success=False)
        
        if depth >= max_depth:
            return TerminationOutcome(True, 0.0, reason="max_depth", success=False)
        
        return TerminationOutcome(False)
    
    def _extract_number_from_result(self, text: str) -> Optional[float]:
        """Extract numerical answer from result"""
        # Look for number patterns
        patterns = [
            r"answer[:\s]+([+-]?\d*\.?\d+)",
            r"final answer[:\s]+([+-]?\d*\.?\d+)",
            r"solution[:\s]+([+-]?\d*\.?\d+)",
            r"([+-]?\d*\.?\d+)\s*$"  # Number at end of text
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
        
        return None
    
    def _extract_number_from_text(self, text: str) -> Optional[float]:
        """Extract number from gold answer text"""
        numbers = re.findall(r'[+-]?\d*\.?\d+', text)
        if numbers:
            try:
                return float(numbers[-1])  # Take last number
            except ValueError:
                pass
        return None
