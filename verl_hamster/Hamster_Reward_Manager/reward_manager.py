import numpy as np
from verl import DataProto
from verl.workers.reward_manager import AbstractRewardManager
import torch
import math

class Hamster_Reward_Manager(AbstractRewardManager):

    def __init__(
        self,
        tokenizer,
        num_examine: int = 2,
        format_score: float = 0.0,
        lambda1: float = 0.6,   
        lambda2: float = 0.2,   
        lambda3: float = 0.2,   

        gamma: float = 1.5,    
        w_final: float = 0.7,   
        w_steps: float = 0.3,   

        tau_calib: float = 0.7,
        tau_struct: float = 0.7,

        step_gain: float = 1.0,     
        step_mode: str = "tanh",  
        step_kappa: float = 3.0,   
        step_margin: float = 0.0,   
        logit_eps: float = 1e-6,    
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

    # ---------- 工具 ----------
    @staticmethod
    def _safe_kl(p, q, eps=1e-12):
        p = np.asarray(p, dtype=np.float64)
        q = np.asarray(q, dtype=np.float64)
        p = np.clip(p, eps, 1.0)
        q = np.clip(q, eps, 1.0)
        p /= p.sum()
        q /= q.sum()
        return float(np.sum(p * np.log(p / q)))  # nats

    @staticmethod
    def _entropy(p, eps=1e-12):
        p = np.asarray(p, dtype=np.float64)
        p = np.clip(p, eps, 1.0)
        p /= p.sum()
        return float(-np.sum(p * np.log(p)))  # nats

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

    def _score_from_kl(self, kl_value: float, tau: float) -> float:
        kl_value = max(0.0, float(kl_value))
        return float(math.exp(-tau * kl_value))

    def _score_from_entropy_binary(self, p) -> float:
        H = self._entropy(p)
        return float(1.0 - H / math.log(2.0))

    def _amplify(self, s01: float) -> float:
 
        s = float(s01)
        if self.step_mode == "logit":
            s = np.clip(s, self.logit_eps, 1.0 - self.logit_eps)
            val = math.log(s / (1.0 - s))

            val = val / math.log((1.0 - self.logit_eps) / self.logit_eps)
        elif self.step_mode == "tanh":
            val = math.tanh(self.step_kappa * (s - 0.5)) 
        elif self.step_mode == "linear":
            val = 2.0 * (s - 0.5)  # [-1, +1]
        else:
            raise ValueError(f"Unknown step_mode={self.step_mode}")


        margin = self.step_margin if s01 >= 0.5 else -self.step_margin

        return self.step_gain * val + margin

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
        reward_tensor = torch.zeros_like(responses, dtype=torch.float32, device=device)

        meta_info = getattr(data, "meta_info", {}) or {}
        calibs = meta_info.get("graph_calibrations", [])
        trajectories = meta_info.get("trajectories", [])

        λ1, λ2, λ3 = float(self.lambda1), float(self.lambda2), float(self.lambda3)
        sumλ = max(1e-8, λ1 + λ2 + λ3)
        γ = float(self.gamma)

        printed = 0
        
        batch_s_calib = []
        batch_s_cert  = []
        batch_s_struct= []
        batch_r_steps = []
        batch_r_final = []
        batch_r_total = []   

        for i in range(B):
            calib_i = calibs[i] if i < len(calibs) else None
            if calib_i:
                fwd  = calib_i.get("forward_marginals", {})
                noe  = calib_i.get("forward_no_evidence_marginals", {})
                post = calib_i.get("posterior_marginals", {})
                answer_state = calib_i.get("answer_state")  # True/False/None
            else:
                fwd, noe, post, answer_state = {}, {}, {}, None

            step_rewards = []


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

                    raw_kl_struct = self._safe_kl(b_plus, b_zero)
                    s_struct = 1.0 - math.exp(-self.tau_struct * raw_kl_struct)
                    batch_s_struct.append(s_struct)

                    s01 = (λ1 * s_calib + λ2 * s_cert + λ3 * s_struct) / sumλ   # ∈[0,1]
                    r_step = self._amplify(s01)                                  # 有符号实数
                    step_rewards.append(r_step)

            if len(step_rewards) > 0:
                weights = np.array([γ ** t for t in range(len(step_rewards))], dtype=np.float64)
                R_steps = float(np.sum(weights * np.array(step_rewards)) / np.sum(weights)) 
            else:
                R_steps = 0.0
            batch_r_steps.append(R_steps)

            if answer_state is True:
                R_final = 1.0
            elif answer_state is False:
                R_final = 0.0
            else:
                R_final = 0.0
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
                            R_final = 0.0
                    except Exception:
                        R_final = 0.0
            batch_r_final.append(R_final)

            R_total = self.w_steps * R_steps + self.w_final * R_final
            batch_r_total.append(R_total)

            prompt_len = prompts.shape[1]
            valid_resp_len = int(attn[i, prompt_len:].sum().item())
            if valid_resp_len > 0:
                reward_tensor[i, valid_resp_len - 1] = R_total


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

