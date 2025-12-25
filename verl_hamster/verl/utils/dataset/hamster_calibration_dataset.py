# jsonl_to_rlhf_parquet_adapter.py
import os
from pathlib import Path
from typing import Optional, List, Union

import torch
from omegaconf import DictConfig
from transformers import PreTrainedTokenizer, ProcessorMixin
from datasets import load_dataset
from verl.utils.dataset.rl_dataset import RLHFDataset
from CPT_FactorGraph_Run.prompt import prompt_no_confidence, prompt_with_confidence, prompt_confidence_guided
from CPT_FactorGraph_Run.prompt import calibration_prompt, calibration_prompt_with_example

class QAJSONLAsParquetRLHF_Calibration(torch.utils.data.Dataset):
    """
    适配器数据集：
    - 接收 jsonl/json/parquet 路径（或列表）
    - 对于 jsonl/json：转换成 RLHFDataset 期望 schema -> 旁路保存 *.cached.parquet
    - 汇总所有 parquet 路径后，用 RLHFDataset 实例化；本类把 __len__/__getitem__ 代理给内部 RLHFDataset

    输入 JSONL 字段：
      - question: str
      - golden_answer: List[str] (可选)
      - index/tools_kwargs/interaction_kwargs (可选)

    生成 Parquet 字段：
      - prompt: List[{"role":"user","content": str}]
      - extra_info（仅当非空才写入）:
          - index: int (可选)
          - reward_model: {"ground_truth": List[str]} (可选)
          - tools_kwargs / interaction_kwargs (仅当非空)
    """

    def __init__(
        self,
        data_files: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
        force_rebuild: bool = True,   # 新增：强制重建（默认开启）
    ):
        if not isinstance(data_files, (list, tuple)):
            data_files = [data_files]
        self._tok = tokenizer
        self._cfg = config
        self._proc = processor
        self._force_rebuild = force_rebuild

        parquet_paths = []
        for f in data_files:
            f = str(f)
            ext = os.path.splitext(f)[1].lower()
            if ext in (".parquet", ".pq"):
                # 如果你也想强制覆盖现有 parquet，可在这里把 parquet -> jsonl 回不去，
                # 一般不建议。保持直通。
                parquet_paths.append(f)
            elif ext in (".jsonl", ".json"):
                parquet_paths.append(self._ensure_cached_parquet(f))
            else:
                raise ValueError(f"Unsupported data file: {f}")

        self._inner = RLHFDataset(
            data_files=parquet_paths,
            tokenizer=self._tok,
            config=self._cfg,
            processor=self._proc,
        )

    # ---------- 将 jsonl/json 转 parquet（RLHF schema） ----------
    def _ensure_cached_parquet(self, jsonl_path: str) -> str:
        p = Path(jsonl_path)
        parquet_path = str(p.with_suffix(p.suffix + ".cached.parquet"))  # e.g. train.jsonl.cached.parquet
        tmp_path = parquet_path + ".tmp"

        # 每次都重建（忽略缓存）
        print(f"[QAJSONLAsParquetRLHF] Rebuilding Parquet from JSON[L]: {jsonl_path} -> {parquet_path}")

        # 读 JSON/JSONL
        ds = load_dataset("json", data_files=jsonl_path, split="train")

        def to_prompt(ex):
            q = ex.get("question")
            if q is None:
                raise ValueError(f"Missing 'question' in example: {ex}")
            
            q = calibration_prompt_with_example.format(question=q)

            # 兼容字段名 golden_answers / golden_answer（二选一都可）
            gts = ex.get("golden_answers")
            if gts is None:
                gts = ex.get("golden_answer")

            if gts is not None:
                if isinstance(gts, str):
                    gts = [gts]
                elif isinstance(gts, list):
                    gts = [str(x) for x in gts]
                else:
                    gts = [str(gts)]

            idx = ex.get("index")
            tools_kwargs = ex.get("tools_kwargs") or None
            interaction_kwargs = ex.get("interaction_kwargs") or None

            out = {"prompt": [{"role": "user", "content": q}]}

            extra = {}
            if idx is not None:
                try:
                    extra["index"] = int(idx)
                except Exception:
                    extra["index"] = idx
            if gts:
                # 同时写两份，方便训练侧按任意路径读取
                extra["reward_model"] = {"ground_truth": list(gts)}
                extra["golden_answers"] = list(gts)
                out["golden_answers"] = list(gts)
            if isinstance(tools_kwargs, dict) and len(tools_kwargs) > 0:
                extra["tools_kwargs"] = tools_kwargs
            if isinstance(interaction_kwargs, dict) and len(interaction_kwargs) > 0:
                extra["interaction_kwargs"] = interaction_kwargs

            if extra:
                out["extra_info"] = extra
            return out

        keep = ["prompt", "extra_info", "golden_answers"]
        ds = ds.map(to_prompt, remove_columns=[c for c in ds.column_names if c not in keep])

        # 原子覆盖：先写 tmp，再替换为正式文件
        out_dir = os.path.dirname(parquet_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        # 如果 tmp 已存在，先删
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

        ds.to_parquet(tmp_path)
        os.replace(tmp_path, parquet_path)

        print(f"[QAJSONLAsParquetRLHF] Saved parquet: {parquet_path}")
        return parquet_path

    # ---------- 代理 ----------
    def __len__(self):
        return len(self._inner)

    def __getitem__(self, idx):
        return self._inner[idx]

    def resume_dataset_state(self):
        if hasattr(self._inner, "resume_dataset_state"):
            return self._inner.resume_dataset_state()
