import numpy as np
from verl import DataProto
from verl.workers.reward_manager import AbstractRewardManager
import torch
import math

class Hamster_Reward_Manager_Step_Wise(AbstractRewardManager):

    def __init__(
        self,
        tokenizer,
        num_examine: int = 2,
        format_score: float = 0.0,
        # 三个子指标权重
        lambda1: float = 0.6,   # 校准一致性
        lambda2: float = 0.2,   # 确定性（低熵）
        lambda3: float = 0.2,   # 结构增益
        # 折扣与终局/步级权重
        gamma: float = 1.5,    # 折扣
        w_final: float = 0.7,   # 终局项权重
        w_steps: float = 0.3,   # 步级项权重（建议 w_final + w_steps = 1）
        # 指标到 [0,1] 的映射温度
        tau_calib: float = 0.7,
        tau_struct: float = 0.7,
        # 组内差距放大
        step_gain: float = 1.0,     # 步级奖励增益（越大差距越大）
        step_mode: str = "tanh",   # 'logit' | 'tanh' | 'linear'
        step_kappa: float = 3.0,    # tanh 模式的斜率
        step_margin: float = 0.0,   # 额外边际：s>0.5 加 +m，s<0.5 加 -m
        logit_eps: float = 1e-6,    # logit 的数值安全
        lookahead: int = 8,
        step_clip_min: float = -5.0,
        step_clip_max: float =  5.0,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.format_score = format_score

        self.lambda1 = float(lambda1)
        self.lambda2 = float(lambda2)
        self.lambda3 = float(lambda3)

        self.gamma = float(gamma)
        self.w_final = float(w_final)
        self.w_steps = float(w_steps)

        self.tau_calib = float(tau_calib)
        self.tau_struct = float(tau_struct)

        self.step_gain = float(step_gain)
        self.step_mode = str(step_mode).lower()
        self.step_kappa = float(step_kappa)
        self.step_margin = float(step_margin)
        self.logit_eps = float(logit_eps)
        self.lookahead = int(lookahead)
        self.step_clip_min = float(step_clip_min); 
        self.step_clip_max = float(step_clip_max)

    # ---------- 工具 ----------
    @staticmethod
    def _safe_kl(p, q, eps=1e-12):
        p = np.asarray(p, dtype=np.float64); q = np.asarray(q, dtype=np.float64)
        p = np.clip(p, eps, 1.0); q = np.clip(q, eps, 1.0)
        p /= p.sum(); q /= q.sum()
        return float(np.sum(p * np.log(p / q)))

    @staticmethod
    def _entropy(p, eps=1e-12):
        p = np.asarray(p, dtype=np.float64); p = np.clip(p, eps, 1.0); p /= p.sum()
        return float(-np.sum(p * np.log(p)))

    @staticmethod
    def _avg3(p, q, r):
        return (np.asarray(p, dtype=np.float64) +
                np.asarray(q, dtype=np.float64) +
                np.asarray(r, dtype=np.float64)) / 3.0

    @staticmethod
    def _num_steps(marginals: dict) -> int:
        t = 0
        while (f"S_{t}" in marginals and f"A_{t}" in marginals and f"O_{t}" in marginals):
            t += 1
        return t

    # ---------- 指标 -> [0,1] ----------
    def _score_from_kl(self, kl_value: float, tau: float) -> float:
        kl_value = max(0.0, float(kl_value))
        return float(math.exp(-tau * kl_value))

    def _score_from_entropy_binary(self, p) -> float:
        H = self._entropy(p)
        return float(1.0 - H / math.log(2.0))

    def _amplify(self, s01: float) -> float:
        """
        把 s∈[0,1] → 有符号实数并控制在 [-5, +5]。
        """
        s = float(s01)
        if self.step_mode == "logit":
            s = np.clip(s, self.logit_eps, 1.0 - self.logit_eps)
            val = math.log(s / (1.0 - s))
            # logit输出范围取决于eps，比如eps=1e-6时约[-13.8,+13.8]
            # 缩放到[-1,+1]
            val = val / math.log((1.0 - self.logit_eps) / self.logit_eps)
        elif self.step_mode == "tanh":
            val = math.tanh(self.step_kappa * (s - 0.5))  # 本身就在(-1,+1)
        elif self.step_mode == "linear":
            val = 2.0 * (s - 0.5)  # [-1, +1]
        else:
            raise ValueError(f"Unknown step_mode={self.step_mode}")

        # 额外边际项（使正负进一步拉开）
        margin = self.step_margin if s01 >= 0.5 else -self.step_margin

        return self.step_gain * val + margin

    def _find_tag_starts(self, response_ids: torch.Tensor, tag: str):
        """返回所有 <tag> 的 [start,end)（end 为标签自身的结束下标），按出现顺序排序（相对 responses）。"""
        ids = response_ids.view(-1).tolist()
        look = self.lookahead

        def _find_tag_hits(tag_text: str, relaxed: bool = True):
            hits = []
            for pos in range(len(ids)):
                for k in range(1, min(look, len(ids) - pos) + 1):
                    s = self.tokenizer.decode(ids[pos:pos + k], skip_special_tokens=False)
                    if (s.strip() if relaxed else s).startswith(tag_text):
                        hits.append((pos, k)); break
            return hits

        hits = _find_tag_hits(f"<{tag}>")
        spans = [(p, p + k) for (p, k) in hits]
        spans.sort(key=lambda x: x[0])
        return spans

    def _find_search_close_spans(self, response_ids: torch.Tensor):
        """返回所有 </search> 关闭标签的 [start,end)（相对 responses）。"""
        ids = response_ids.view(-1).tolist()
        look = self.lookahead

        def _find_tag_hits(tag: str, relaxed: bool = True):
            hits = []
            for pos in range(len(ids)):
                for k in range(1, min(look, len(ids) - pos) + 1):
                    s = self.tokenizer.decode(ids[pos:pos + k], skip_special_tokens=False)
                    if (s.strip() if relaxed else s).startswith(tag):
                        hits.append((pos, k)); break
            return hits

        close_spans = _find_tag_hits("</search>")
        close_spans = [(p, p + k) for (p, k) in close_spans]
        close_spans.sort(key=lambda x: x[0])
        return close_spans

    def __call__(self, data: DataProto, *args, **kwargs):
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        assert all(k in data.batch for k in ('responses', 'prompts', 'attention_mask')), \
            "DataProto.batch 需要包含 'responses'、'prompts'、'attention_mask'"

        responses = data.batch['responses']      # [B, T_resp]
        prompts   = data.batch['prompts']        # [B, T_prompt]
        attn      = data.batch['attention_mask'] # [B, T_prompt+T_resp]
        device    = responses.device

        B = responses.size(0)
        T_resp = responses.size(1)
        reward_tensor = torch.zeros_like(responses, dtype=torch.float32, device=device)

        meta_info = getattr(data, "meta_info", {}) or {}
        calibs = meta_info.get("graph_calibrations", [])
        trajectories = meta_info.get("trajectories", [])

        λ1, λ2, λ3 = float(self.lambda1), float(self.lambda2), float(self.lambda3)
        sumλ = max(1e-8, λ1 + λ2 + λ3)
        γ = float(self.gamma)
        
        
        batch_s_calib = []
        batch_s_cert  = []
        batch_s_struct= []
        batch_r_steps = []
        batch_r_final = []
        batch_r_total = []

        printed = 0

        for i in range(B):
            calib_i = calibs[i] if i < len(calibs) else None
            if calib_i:
                fwd  = calib_i.get("forward_marginals", {})
                noe  = calib_i.get("forward_no_evidence_marginals", {})
                post = calib_i.get("posterior_marginals", {})
                answer_state = calib_i.get("answer_state")  # True/False/None
            else:
                fwd, noe, post, answer_state = {}, {}, {}, None

            # 有效 response 长度（相对 responses）
            prompt_len = prompts.shape[1]
            valid_resp_len = int(attn[i, prompt_len:].sum().item())
            if valid_resp_len <= 0:
                # 没有可写入的响应位，直接跳过（或视需要写到 index 0）
                continue
            last_resp_pos = valid_resp_len - 1

            step_rewards = []

             # -------- 步级奖励：先算 s∈[0,1]，再放大为有符号实数 --------
            if fwd and noe and post:
                T = min(self._num_steps(fwd), self._num_steps(noe), self._num_steps(post))
                for t in range(T):
                    kS, kA, kO = f"S_{t}", f"A_{t}", f"O_{t}"
                    if not (kS in fwd and kA in fwd and kO in fwd): break
                    if not (kS in noe and kA in noe and kO in noe): break
                    if not (kS in post and kA in post and kO in post): break

                    b_minus = self._avg3(fwd[kS],  fwd[kA],  fwd[kO])
                    b_plus  = self._avg3(post[kS], post[kA], post[kO])
                    b_zero  = self._avg3(noe[kS],  noe[kA],  noe[kO])

                    # kl is small is good
                    s_calib  = self._score_from_kl(self._safe_kl(b_plus, b_minus), tau=self.tau_calib)   # ∈[0,1]
                    batch_s_calib.append(s_calib)
                    s_cert   = self._score_from_entropy_binary(b_plus)                                    # ∈[0,1]
                    batch_s_cert.append(s_cert)
                    # s_struct = self._score_from_kl(self._safe_kl(b_plus, b_zero),  tau=self.tau_struct)  # ∈[0,1]
                    # KL_struct：结构变化越大越好
                    raw_kl_struct = self._safe_kl(b_plus, b_zero)
                    s_struct = 1.0 - math.exp(-self.tau_struct * raw_kl_struct)
                    batch_s_struct.append(s_struct)

                    s01 = (λ1 * s_calib + λ2 * s_cert + λ3 * s_struct) / sumλ   # ∈[0,1]
                    # s01 = (λ1 * s_calib + λ3 * s_struct) / (λ1 + λ3)   # ∈[0,1]
                    r_step = self._amplify(s01)                                  # 有符号实数
                    step_rewards.append(r_step)
                    
            # ---- 将每步奖励写到对应位置 ----
            if len(step_rewards) > 0:
                close_spans = self._find_search_close_spans(responses[i])
                think_starts = self._find_tag_starts(responses[i], "think")

                for t, r in enumerate(step_rewards):
                    if t < len(close_spans):
                        _, end = close_spans[t]            # </search> 的 [start,end)
                        pos = min(end - 1, T_resp - 1)
                        # 若超过有效响应范围，回退到最后有效位
                        pos = min(pos, last_resp_pos)
                        reward_tensor[i, pos] += float(np.clip(r, self.step_clip_min, self.step_clip_max))
                        continue

                    if (t + 1) < len(think_starts):
                        next_think_start, _ = think_starts[t + 1]
                        pos = max(0, min(next_think_start - 1, T_resp - 1))
                        pos = min(pos, last_resp_pos)
                        reward_tensor[i, pos] += float(np.clip(r, self.step_clip_min, self.step_clip_max))
                        continue

                    # 兜底：最后一个有效 response token
                    reward_tensor[i, last_resp_pos] += float(np.clip(r, self.step_clip_min, self.step_clip_max))

                # 折扣聚合步级奖励（仅用于统计/总奖励）
                weights = np.array([γ ** t for t in range(len(step_rewards))], dtype=np.float64)
                R_steps = float(np.sum(weights * np.array(step_rewards)) / np.sum(weights))
            else:
                R_steps = 0.0

            # ---- 终局奖励 ----
            if answer_state is True:
                R_final = 1.0
            elif answer_state is False:
                R_final = -1.0
            else:
                R_final = 0.0   # 未知=0.0 与注释一致
                if i < len(trajectories):
                    traj_i = trajectories[i] or {}
                    final_ans = traj_i.get("final_answer")
                    golden = traj_i.get("golden_answer", traj_i.get("golden_answers"))
                    try:
                        from CPT_FactorGraph_Run.utils import compare_answers
                        cmp = compare_answers(expected_answer=golden, agent_answer=final_ans)
                        if cmp is True:
                            R_final = 1.0
                        elif cmp is False:
                            R_final = -1.0
                    except Exception:
                        R_final = 0.0
            batch_r_final.append(R_final)

            # ---- 最后 token 写入凸组合总奖励 ----
            R_total = self.w_steps * R_steps + self.w_final * R_final
            reward_tensor[i, last_resp_pos] += float(R_total)

            # ---- 调试打印 ----
            if self.num_examine > 0 and printed < self.num_examine:
                printed += 1
                import numpy as _np
                if len(step_rewards) > 0:
                    print(
                        f"[Reward] sample#{i}  steps={len(step_rewards)}  "
                        f"R_steps={R_steps:+.4f}  R_final={R_final:+.2f}  "
                        f"R_total(last)={R_total:+.4f}  first3 r_step={_np.array(step_rewards[:3])}"
                    )
                else:
                    print(f"[Reward] sample#{i}  (no step rewards)  R_final={R_final:+.2f}  R_total={R_total:+.4f}")

        # 可选打印：详细调试信息
            if self.num_examine > 0 and printed < self.num_examine:
                printed += 1
                print(f"[Reward Debug] ▶ Sample #{i}")
                print(f"  Step count (T): {len(step_rewards)}")
                print(f"  Final answer state: {answer_state}")

                if len(step_rewards) > 0:
                    import numpy as _np
                    # 展示每个 step 的详细 reward 构成
                    print(f"  γ = {γ:.3f} | w_steps = {self.w_steps:.2f} | w_final = {self.w_final:.2f}")
                    print(f"  Step rewards detail:")
                    for t, r_step in enumerate(step_rewards):
                        weight = γ ** t
                        print(f"    • t={t:<2d} | r_step={r_step:+.4f} | weight={weight:.4f} | weighted={r_step*weight:.4f}")
                    weights = np.array([γ ** t for t in range(len(step_rewards))])
                    normalized_weights = weights / weights.sum()
                    print(f"  Normalized weights: {np.round(normalized_weights, 4)}")

                    # step 级指标统计
                    print(f"  mean(r_step)={_np.mean(step_rewards):+.4f}, std(r_step)={_np.std(step_rewards):.4f}")
                    print(f"  R_steps (discounted aggregate) = {R_steps:+.4f}")
                else:
                    print("  (no step rewards available)")

                # 终局奖励信息
                print(f"  R_final = {R_final:+.4f}")
                print(f"  R_total = {R_total:+.4f}")

                # 展示子指标（校准、一致性、结构）
                if len(batch_s_calib) > 0:
                    print(f"  Last batch sub-metrics (mean ± std):")
                    print(f"    s_calib:  {np.mean(batch_s_calib):.4f} ± {np.std(batch_s_calib):.4f}")
                    print(f"    s_cert :  {np.mean(batch_s_cert):.4f} ± {np.std(batch_s_cert):.4f}")
                    print(f"    s_struct: {np.mean(batch_s_struct):.4f} ± {np.std(batch_s_struct):.4f}")

        def _m(x):  
            return float(np.mean(x)) if len(x) > 0 else 0.0
        def _s(x):  
            return float(np.std(x)) if len(x) > 0 else 0.0
        
        batch_s_calib_mean, batch_s_calib_std   = _m(batch_s_calib),  _s(batch_s_calib)
        batch_s_cert_mean,  batch_s_cert_std    = _m(batch_s_cert),   _s(batch_s_cert)
        batch_s_struct_mean,batch_s_struct_std  = _m(batch_s_struct), _s(batch_s_struct)
        batch_r_steps_mean, batch_r_steps_std   = _m(batch_r_steps),  _s(batch_r_steps)
        batch_r_final_mean, batch_r_final_std   = _m(batch_r_final),  _s(batch_r_final)
        batch_r_total_mean, batch_r_total_std   = _m(batch_r_total),  _s(batch_r_total)  

        reward_extra_infos_dict = {
            # s_* 指标
            "reward/batch_s_calib_mean":  batch_s_calib_mean,
            "reward/batch_s_calib_std":   batch_s_calib_std,
            "reward/batch_s_cert_mean":   batch_s_cert_mean,
            "reward/batch_s_cert_std":    batch_s_cert_std,
            "reward/batch_s_struct_mean": batch_s_struct_mean,
            "reward/batch_s_struct_std":  batch_s_struct_std,
            # 分项奖励
            "reward/batch_r_steps_mean":  batch_r_steps_mean,
            "reward/batch_r_steps_std":   batch_r_steps_std,
            "reward/batch_r_final_mean":  batch_r_final_mean,
            "reward/batch_r_final_std":   batch_r_final_std,
            # 总奖励（关键）
            "reward/batch_r_total_mean":  batch_r_total_mean,  # NEW
            "reward/batch_r_total_std":   batch_r_total_std,   # NEW
        }
        
        return {"reward_tensor": reward_tensor, "reward_extra_infos": reward_extra_infos_dict}
