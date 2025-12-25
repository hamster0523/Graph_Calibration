import numpy as np
from verl import DataProto
from verl.workers.reward_manager import AbstractRewardManager
import torch

class Hamster_Format_Reward_Manager(AbstractRewardManager):


    def __init__(
        self,
        tokenizer,
        num_examine: int = 2,   # 打印多少个样本的 debug
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = int(num_examine)

    def __call__(self, data: DataProto, *args, **kwargs):
        # 若已存在 rm_scores，直接返回
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

        meta_info   = getattr(data, "meta_info", {}) or {}
        trajectories = meta_info.get("trajectories", [])  # 你在 run_llm_loop 里塞的

        all_R = []
        printed = 0

        for i in range(B):
            traj_i = trajectories[i] if i < len(trajectories) else None
            R_total = 0.0

            if traj_i and isinstance(traj_i, dict):
                steps = traj_i.get("steps", []) or []
                for st in steps:
                    # 关键：这里假定你在每个 step 里写入了 st["format_ok"]: bool
                    if st.get("format_ok", False):
                        R_total += 1.0

            all_R.append(R_total)

            # 把奖励打在「最后一个有效 response token」上
            prompt_len = prompts.shape[1]
            valid_resp_len = int(attn[i, prompt_len:].sum().item())
            if valid_resp_len > 0:
                reward_tensor[i, valid_resp_len - 1] = R_total

            # debug 打印
            if self.num_examine > 0 and printed < self.num_examine:
                printed += 1
                num_steps = len(traj_i.get("steps", [])) if traj_i else 0
                print(f"[FormatReward] sample #{i} | steps={num_steps} | R_total={R_total:.1f}")
                if traj_i:
                    print(f"  question: {traj_i.get('question')}")
                    # 也可以打印每一步的 format_ok
                    fmt_flags = [st.get("format_ok", False) for st in traj_i.get("steps", [])]
                    print(f"  format_ok per step: {fmt_flags}")

        # 统计信息，方便 wandb / tensorboard 看
        if len(all_R) > 0:
            arr = np.array(all_R, dtype=np.float32)
            R_mean = float(arr.mean())
            R_std  = float(arr.std())
        else:
            R_mean = 0.0
            R_std  = 0.0

        reward_extra_infos_dict = {
            "reward/format_R_total_mean": R_mean,
            "reward/format_R_total_std":  R_std,
        }

        return {
            "reward_tensor": reward_tensor,
            "reward_extra_infos": reward_extra_infos_dict,
        }
