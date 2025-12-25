import numpy as np
import torch
from verl import DataProto
from verl.workers.reward_manager import AbstractRewardManager
import math


class Hamster_Global_Conf_Reward_Manager(AbstractRewardManager):

    def __init__(
        self,
        tokenizer,
        num_examine: int = 2,
        lambda_fmt: float = 0.4,
        lambda_calib: float = 0.4,
        lambda_sharp: float = 0.2,
        use_entropy: bool = False,
        step_gain: float = 1.0,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = int(num_examine)

        self.lambda_fmt = float(lambda_fmt)
        self.lambda_calib = float(lambda_calib)
        self.lambda_sharp = float(lambda_sharp)
        self.use_entropy = bool(use_entropy)
        self.step_gain = float(step_gain)

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
        return max(0.0, min(1.0, x))

    def __call__(self, data: DataProto, *args, **kwargs):


        # 如果已经有 rm_scores，就直接复用，避免重复算
        if "rm_scores" in data.batch:
            return data.batch["rm_scores"]

        responses = data.batch["responses"]       # [B, T_resp]
        prompts = data.batch["prompts"]           # [B, T_total]（prompt+resp），这里只用长度
        attn = data.batch["attention_mask"]       # [B, T_total]
        device = responses.device
        B = responses.size(0)

        reward_tensor = torch.zeros_like(
            responses, dtype=torch.float32, device=device
        )

        # ===== 从 non_tensor_batch 读取信息 =====
        nt = getattr(data, "non_tensor_batch", {}) or {}

        # uid 列表（与当前 batch 对齐）
        raw_uids = nt.get("rollout_uid", None)
        if raw_uids is None:
            # 兜底：没有 uid 就按索引造一个
            uids = [str(i) for i in range(B)]
        else:
            print("get rollout_uid from non_tensor_batch")
            if isinstance(raw_uids, np.ndarray):
                uids = raw_uids.tolist()
            elif isinstance(raw_uids, (list, tuple)):
                uids = list(raw_uids)
            else:
                # 单个值也转成 list
                uids = [raw_uids]
            uids = [str(u) for u in uids]

            # 长度兜底到至少 B
            if len(uids) < B:
                # 用最后一个补齐，正常不会走到这里
                pad = [uids[-1]] * (B - len(uids))
                uids = uids + pad
            elif len(uids) > B:
                uids = uids[:B]

        # # trajectories: 可能是 list 或 np.ndarray(dtype=object)
        # raw_trajs = (
        #     nt.get("trajectories")
        #     or nt.get("hamster_trajectories")
        #     or []
        # )
        # if isinstance(raw_trajs, np.ndarray):
        #     traj_list = raw_trajs.tolist()
        # else:
        #     traj_list = list(raw_trajs)

        # # graph_calibrations: 同理
        # raw_calibs = nt.get("graph_calibrations") or []
        # if isinstance(raw_calibs, np.ndarray):
        #     calib_list = raw_calibs.tolist()
        # else:
        #     calib_list = list(raw_calibs)
        # ===== trajectories =====
        if "trajectories" in nt:
            raw_trajs = nt["trajectories"]
        elif "hamster_trajectories" in nt:
            raw_trajs = nt["hamster_trajectories"]
        else:
            raw_trajs = []

        if isinstance(raw_trajs, np.ndarray):
            traj_list = raw_trajs.tolist()
        else:
            traj_list = list(raw_trajs)

        # ===== graph_calibrations =====
        if "graph_calibrations" in nt:
            raw_calibs = nt["graph_calibrations"]
        else:
            raw_calibs = []

        if isinstance(raw_calibs, np.ndarray):
            calib_list = raw_calibs.tolist()
        else:
            calib_list = list(raw_calibs)

        # ===== 建立 uid -> trajectory / calibration 映射 =====
        traj_map = {}
        for t in traj_list:
            if not isinstance(t, dict):
                continue
            uid_t = t.get("rollout_uid", None)
            if uid_t is None:
                continue
            uid_t = str(uid_t)
            # 避免重复覆盖，第一份为准
            if uid_t not in traj_map:
                traj_map[uid_t] = t

        calib_map = {}
        for c in calib_list:
            if not isinstance(c, dict):
                continue
            uid_c = c.get("rollout_uid", None)
            if uid_c is None:
                continue
            uid_c = str(uid_c)
            if uid_c not in calib_map:
                calib_map[uid_c] = c

        # 统计量
        all_R_total, all_R_steps, all_R_final = [], [], []
        all_step_scores = []
        printed = 0

        λ_fmt = self.lambda_fmt
        λ_calib = self.lambda_calib
        λ_sharp = self.lambda_sharp

        # ===== 遍历当前 batch 的每个样本，按 uid 对齐取 traj / calib =====
        for i in range(B):
            uid_i = uids[i]

            traj = traj_map.get(uid_i, None)
            calib = calib_map.get(uid_i, None)

            R_steps_sum = 0.0
            R_final = 0.0
            step_scores_i = []

            posterior_marginals = {}
            answer_state = None
            if isinstance(calib, dict):
                posterior_marginals = calib.get("posterior_marginals", {}) or {}
                answer_state = calib.get("answer_state", None)

            # ==== 终局奖励：根据 answer_state ==== #
            # answer_state 为 True 表示最终答案正确；否则 0
            R_final = 1.0 if answer_state is True else 0.0

            # ===== Step-level 部分 =====
            if traj and isinstance(traj, dict):
                steps = traj.get("steps", []) or []
                len_steps = len(steps)

                for st in steps:
                    if not isinstance(st, dict):
                        step_scores_i.append(0.0)
                        continue

                    step_idx = st.get("step", len(step_scores_i))

                    # 1) format gating：如果格式不过关，直接 0
                    format_ok = bool(st.get("format_ok", False))
                    if not format_ok:
                        s_step = 0.0
                        step_scores_i.append(s_step)
                        continue

                    r_fmt = 1.0  # 只要 format_ok，就给 1

                    # 2) verbal confidence：优先用 verbal，缺失就 fallback 到 action_confidence
                    raw_c_verbal = st.get("action_verbal_confidence", None)
                    fallback = st.get("action_confidence", 0.0)
                    c_verbal = self._to_float_01(raw_c_verbal, default=fallback)

                    # 3) posterior 对齐（图模型的后验边缘）
                    has_post = False
                    r_calib = 0.0

                    if posterior_marginals:
                        varA = f"A_{step_idx}"
                        varS = f"S_{step_idx}"
                        dist = (
                            posterior_marginals.get(varA)
                            or posterior_marginals.get(varS)
                        )
                        # dist 应该是 [p_wrong, p_right] 或类似
                        if isinstance(dist, (list, tuple)) and len(dist) >= 2:
                            try:
                                p_post = max(0.0, min(1.0, float(dist[1])))
                                r_calib = 1.0 - abs(c_verbal - p_post)
                                has_post = True
                            except Exception:
                                pass

                    # 4) sharpness（是否鼓励更尖锐的 verbal 分布）
                    if self.use_entropy:
                        H = self._binary_entropy(c_verbal)
                        r_sharp = max(0.0, min(1.0, 1.0 - H / math.log(2.0)))
                    else:
                        r_sharp = 0.0

                    # 5) 综合 step 奖励
                    num = λ_fmt * r_fmt
                    den = λ_fmt

                    if has_post and λ_calib > 0:
                        num += λ_calib * r_calib
                        den += λ_calib

                    if self.use_entropy and λ_sharp > 0:
                        num += λ_sharp * r_sharp
                        den += λ_sharp

                    s_step = float(num / den) if den > 0 else 0.0

                    step_scores_i.append(s_step)
                    R_steps_sum += s_step

                # ==== 按总步数平均 ==== #
                if len_steps > 0:
                    R_steps = self.step_gain * (R_steps_sum / len_steps)
                else:
                    R_steps = 0.0

            else:
                R_steps = 0.0
                len_steps = 0

            R_total = R_steps + R_final

            all_R_total.append(R_total)
            all_R_steps.append(R_steps)
            all_R_final.append(R_final)
            all_step_scores.extend(step_scores_i)

            # ==== 写入 reward 到最后一个有效 token ==== #
            prompt_len = prompts.shape[1]
            valid_resp_len = int(attn[i, prompt_len:].sum().item())
            if valid_resp_len > 0:
                reward_tensor[i, valid_resp_len - 1] = R_total

            # ==== Debug 打印 ==== #
            if printed < self.num_examine:
                printed += 1
                print(
                    f"[StepConfReward] sample #{i} (uid={uid_i}) | steps={len_steps} "
                    f"| R_steps={R_steps:.4f} | R_final={R_final:.4f} | "
                    f"R_total={R_total:.4f} (gain={self.step_gain})"
                )
                if traj:
                    fmt_flags = [bool(st.get("format_ok", False)) for st in steps]
                    print(f"  format_ok per step: {fmt_flags}")
                    print(f"  step_scores: {np.round(step_scores_i, 4)}")
                    print(f"  answer_state: {answer_state}")

        # ===== 统计信息（放到 reward_extra_infos） =====
        def mean(x): return float(np.mean(x)) if x else 0.0
        def std(x): return float(np.std(x)) if x else 0.0

        reward_extra_infos_dict = {
            "reward/step_conf_R_total_mean": mean(all_R_total),
            "reward/step_conf_R_total_std": std(all_R_total),
            "reward/step_conf_R_steps_mean": mean(all_R_steps),
            "reward/step_conf_R_steps_std": std(all_R_steps),
            "reward/step_conf_R_final_mean": mean(all_R_final),
            "reward/step_conf_R_final_std": std(all_R_final),
            "reward/step_conf_s_step_mean": mean(all_step_scores),
            "reward/step_conf_s_step_std": std(all_step_scores),
        }

        return {
            "reward_tensor": reward_tensor,
            "reward_extra_infos": reward_extra_infos_dict,
        }
