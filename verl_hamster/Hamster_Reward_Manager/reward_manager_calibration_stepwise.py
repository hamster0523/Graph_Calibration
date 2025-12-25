import numpy as np
import torch
from verl import DataProto
from verl.workers.reward_manager import AbstractRewardManager
import math


class Hamster_Step_Conf_Reward_Manager(AbstractRewardManager):

    def __init__(
        self,
        tokenizer,
        num_examine: int = 2,
        lambda_fmt: float = 0.4,
        lambda_calib: float = 0.4,
        lambda_sharp: float = 0.2,
        use_entropy: bool = True,
        step_gain: float = 1.0,
        lookahead: int = 8,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = int(num_examine)

        self.lambda_fmt = float(lambda_fmt)
        self.lambda_calib = float(lambda_calib)
        self.lambda_sharp = float(lambda_sharp)
        self.use_entropy = bool(use_entropy)
        self.step_gain = float(step_gain)
        self.lookahead = int(lookahead)


    @staticmethod
    def _binary_entropy(p: float, eps: float = 1e-8) -> float:
        p = float(max(min(p, 1.0 - eps), eps))
        return float(-p * math.log(p) - (1.0 - p) * math.log(1.0 - p))

    @staticmethod
    def _to_float_01(value, default: float = 0.5) -> float:

        try:
            x = float(value)
        except Exception:
            x = float(default)
        x = max(0.0, min(1.0, x))
        return x

    def _find_tag_hits(self, ids, tag: str, lookahead: int, relaxed: bool = True):

        hits = []
        n = len(ids)
        for pos in range(n):
            for k in range(1, min(lookahead, n - pos) + 1):
                s = self.tokenizer.decode(ids[pos:pos + k], skip_special_tokens=False)
                check_s = s.strip() if relaxed else s
                if check_s.startswith(tag):
                    hits.append((pos, k))
                    break
        return hits

    def _find_search_close_spans(self, response_ids: torch.Tensor):

        ids = response_ids.view(-1).tolist()
        hits = self._find_tag_hits(ids, "</search>", lookahead=self.lookahead, relaxed=True)
        spans = [(p, p + k) for (p, k) in hits]
        spans.sort(key=lambda x: x[0])
        return spans

    def _find_tag_starts(self, response_ids: torch.Tensor, tag: str):

        ids = response_ids.view(-1).tolist()
        hits = self._find_tag_hits(ids, f"<{tag}>", lookahead=self.lookahead, relaxed=True)
        spans = [(p, p + k) for (p, k) in hits]
        spans.sort(key=lambda x: x[0])
        return spans


    def __call__(self, data: DataProto, *args, **kwargs):
        if "rm_scores" in data.batch:
            return data.batch["rm_scores"]

        assert all(
            k in data.batch
            for k in ("responses", "prompts", "attention_mask")
        ), "DataProto.batch 需要包含 'responses'、'prompts'、'attention_mask'"

        responses = data.batch["responses"]      # [B, T_resp]
        prompts = data.batch["prompts"]          # [B, T_prompt]
        attn = data.batch["attention_mask"]      # [B, T_prompt+T_resp]
        device = responses.device
        B = responses.size(0)
        T_resp = responses.size(1)

        reward_tensor = torch.zeros_like(
            responses, dtype=torch.float32, device=device
        )

        meta_info = getattr(data, "meta_info", {}) or {}
        trajectories = meta_info.get("trajectories", []) or []
        calibrations = meta_info.get("graph_calibrations", []) or []

        calib_map = {}
        for cal in calibrations:
            if not isinstance(cal, dict):
                continue
            key = (cal.get("question"), cal.get("final_answer"))
            calib_map[key] = cal

        all_R_total = []
        all_R_steps = []
        all_R_final = []
        all_step_scores = []
        printed = 0

        λ_fmt = self.lambda_fmt
        λ_calib = self.lambda_calib
        λ_sharp = self.lambda_sharp

        for i in range(B):
            traj = trajectories[i] if i < len(trajectories) else None

            # 有效 response 长度（相对 responses）
            prompt_len = prompts.shape[1]
            valid_resp_len = int(attn[i, prompt_len:].sum().item())
            if valid_resp_len <= 0:
                all_R_total.append(0.0)
                all_R_steps.append(0.0)
                all_R_final.append(0.0)
                continue
            last_resp_pos = valid_resp_len - 1

            # 对应该样本的 calibration
            calib = None
            if traj and isinstance(traj, dict):
                key = (traj.get("question"), traj.get("final_answer"))
                calib = calib_map.get(key)

            posterior_marginals = {}
            answer_state = None
            if isinstance(calib, dict):
                posterior_marginals = calib.get("posterior_marginals", {}) or {}
                answer_state = calib.get("answer_state", None)

            # 终局奖励：answer_state True → 1.0，否则 0.0
            R_final = 1.0 if answer_state is True else 0.0

            step_scores_i = []   # s_step_j（∈[0,1]，方便统计）
            step_rewards_i = []  # r_step_j（∈{0,1}，实际写入）

            if traj and isinstance(traj, dict):
                steps = traj.get("steps", []) or []
                for st in steps:
                    if not isinstance(st, dict):
                        continue

                    step_idx = st.get("step", len(step_scores_i))

                    # 1) 格式 gating
                    format_ok = bool(st.get("format_ok", False))
                    if not format_ok:
                        s_step = 0.0
                        step_scores_i.append(s_step)
                        step_rewards_i.append(0.0)
                        continue
                    r_fmt = 1.0

                    # 2) verbal 置信度：优先 action_verbal_confidence
                    raw_c_verbal = st.get("action_verbal_confidence", None)
                    fallback = st.get("action_confidence", 0.5)
                    c_verbal = self._to_float_01(raw_c_verbal, default=fallback)

                    # 3) posterior 置信度（图上的 A_t 或 S_t）
                    has_post = False
                    r_calib = 0.0
                    if posterior_marginals:
                        var_name_A = f"A_{step_idx}"
                        var_name_S = f"S_{step_idx}"
                        dist = (
                            posterior_marginals.get(var_name_A)
                            or posterior_marginals.get(var_name_S)
                        )
                        if isinstance(dist, (list, tuple)) and len(dist) >= 2:
                            try:
                                p_post = float(dist[1])
                                p_post = max(0.0, min(1.0, p_post))
                                r_calib = 1.0 - abs(c_verbal - p_post)
                                r_calib = max(0.0, min(1.0, r_calib))
                                has_post = True
                            except Exception:
                                has_post = False

                    # 4) 锐度奖励
                    if self.use_entropy:
                        H = self._binary_entropy(c_verbal)
                        r_sharp = 1.0 - H / math.log(2.0)
                        r_sharp = max(0.0, min(1.0, r_sharp))
                    else:
                        r_sharp = 0.0

                    # 5) 合成 s_step ∈[0,1]
                    num = λ_fmt * r_fmt
                    den = λ_fmt

                    if has_post and λ_calib > 0.0:
                        num += λ_calib * r_calib
                        den += λ_calib

                    if self.use_entropy and λ_sharp > 0.0:
                        num += λ_sharp * r_sharp
                        den += λ_sharp

                    s_step = float(num / den) if den > 0.0 else 0.0
                    step_scores_i.append(s_step)

                    # 6) step 奖励
                    raw_r = self.step_gain * s_step
                    r_step = raw_r
                    step_rewards_i.append(r_step)

            # -------- 把每步奖励写在 </search> 等位置 --------
            R_steps = float(sum(step_rewards_i)) if len(step_rewards_i) > 0 else 0.0

            if len(step_rewards_i) > 0:
                search_close_spans = self._find_search_close_spans(responses[i])
                think_starts = self._find_tag_starts(responses[i], "think")

                for t, r_step in enumerate(step_rewards_i):
                    # 这里 r_step ∈ {0,1}，写 0 其实是 no-op，但逻辑上也没问题
                    if r_step == 0.0:
                        continue

                    # 1) 优先写在第 t 个 </search> 的末尾
                    if t < len(search_close_spans):
                        _, end = search_close_spans[t]
                        pos = min(end - 1, T_resp - 1)
                        pos = min(pos, last_resp_pos)
                        reward_tensor[i, pos] += float(r_step)
                        continue

                    # 2) 不够的话写到“下一步 <think> 起点 - 1”
                    if (t + 1) < len(think_starts):
                        next_think_start, _ = think_starts[t + 1]
                        pos = max(0, min(next_think_start - 1, last_resp_pos))
                        reward_tensor[i, pos] += float(r_step)
                        continue

                    # 3) 兜底：最后一个有效 response token
                    reward_tensor[i, last_resp_pos] += float(r_step)

            # -------- 写入终局奖励：最后一个有效 token --------
            if R_final != 0.0:
                reward_tensor[i, last_resp_pos] += float(R_final)

            # -------- 统计：总奖励 R_total --------
            R_total = R_steps + R_final
            all_R_total.append(R_total)
            all_R_steps.append(R_steps)
            all_R_final.append(R_final)
            all_step_scores.extend(step_scores_i)

            # debug 打印
            if self.num_examine > 0 and printed < self.num_examine:
                printed += 1
                num_steps = len(traj.get("steps", [])) if traj else 0
                print(
                    f"[StepConfReward] sample #{i} | steps={num_steps} "
                    f"| R_steps={R_steps:.4f} | R_final={R_final:.4f} | "
                    f"R_total={R_total:.4f}"
                )
                if traj:
                    print(f"  question: {traj.get('question')}")
                    fmt_flags = [
                        bool(st.get("format_ok", False))
                        for st in traj.get("steps", [])
                    ]
                    print(f"  format_ok per step: {fmt_flags}")
                    print(f"  s_step: {np.round(step_scores_i, 4)}")
                    print(f"  r_step: {np.round(step_rewards_i, 4)}")
                    print(f"  answer_state: {answer_state}")

        # -------- 统计信息（方便 wandb / tb 可视化） --------
        def _m(x):
            return float(np.mean(x)) if len(x) > 0 else 0.0

        def _s(x):
            return float(np.std(x)) if len(x) > 0 else 0.0

        R_total_mean, R_total_std = _m(all_R_total), _s(all_R_total)
        R_steps_mean, R_steps_std = _m(all_R_steps), _s(all_R_steps)
        R_final_mean, R_final_std = _m(all_R_final), _s(all_R_final)

        if all_step_scores:
            arr_s = np.asarray(all_step_scores, dtype=np.float32)
            s_mean = float(arr_s.mean())
            s_std = float(arr_s.std())
        else:
            s_mean = 0.0
            s_std = 0.0

        reward_extra_infos_dict = {
            "reward/step_conf_R_total_mean": R_total_mean,
            "reward/step_conf_R_total_std": R_total_std,
            "reward/step_conf_R_steps_mean": R_steps_mean,
            "reward/step_conf_R_steps_std": R_steps_std,
            "reward/step_conf_R_final_mean": R_final_mean,
            "reward/step_conf_R_final_std": R_final_std,
            "reward/step_conf_s_step_mean": s_mean,
            "reward/step_conf_s_step_std": s_std,
        }

        return {
            "reward_tensor": reward_tensor,
            "reward_extra_infos": reward_extra_infos_dict,
        }
