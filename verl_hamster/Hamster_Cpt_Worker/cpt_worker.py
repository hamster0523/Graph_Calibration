import sys
from typing import List, Dict, Any, Optional
import numpy as np
import torch

from CPT_FactorGraph_Run.build_trajectory_to_graph import GraphBuilder
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register, Execute
from CPT_FactorGraph_Run.utils import make_identity_obs

def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    try:
        from omegaconf import OmegaConf
        if isinstance(cfg, dict):
            cur = cfg
            for part in key.split("."):
                if not isinstance(cur, dict) or part not in cur:
                    return default
                cur = cur[part]
            return cur
        else:
            val = OmegaConf.select(cfg, key)
            return default if val is None else val
    except Exception:
        cur = cfg
        for part in key.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

class GraphServiceWorker(Worker):

    def __init__(self, config=None, role=None, **kwargs):
        Worker.__init__(self)
        
        self.config = config
        self.role = role
        self.kwargs = kwargs

        self.gb = None
        self.graph_obs_eps = 1e-6
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[GraphServiceWorker.__init__] role={role}, device={self.device}")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        cpt_model_path = (
            _cfg_get(self.config, "hamster.cpt_model_path")
            or _cfg_get(self.config, "cpt_model_path")
        )
        do_calibration = bool(
            _cfg_get(self.config, "hamster.cpt_do_calibration", True)
            or _cfg_get(self.config, "cpt_do_calibration", True)
        )
        sharpen_strength = float(
            _cfg_get(self.config, "hamster.graph_sharpen_strength", 3.0)
            or _cfg_get(self.config, "graph_sharpen_strength", 3.0)
        )
        min_assoc_strength = float(
            _cfg_get(self.config, "hamster.graph_min_assoc_strength", 0.15)
            or _cfg_get(self.config, "graph_min_assoc_strength", 0.15)
        )
        self.graph_obs_eps = float(
            _cfg_get(self.config, "hamster.graph_obs_eps", 1e-6)
            or _cfg_get(self.config, "graph_obs_eps", 1e-6)
        )

        if not cpt_model_path:
            raise ValueError("[GraphServiceWorker.init_model] cpt_model_path 未在配置中找到（hamster.cpt_model_path 或 cpt_model_path）")

        self.gb = GraphBuilder(
            cpt_model_path,
            cpt_do_calibration=do_calibration,
            sharpen_strength=sharpen_strength,
            min_assoc_strength=min_assoc_strength,
        )

        import os
        print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
        print(
            f"[GraphServiceWorker] Initialized on device {self.device} | "
            f"cpt_model_path={cpt_model_path} | do_calibration={do_calibration} | "
            f"sharpen_strength={sharpen_strength} | min_assoc_strength={min_assoc_strength} | "
            f"graph_obs_eps={self.graph_obs_eps}"
        )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, execute_mode=Execute.RANK_ZERO)
    def build_and_infer(
        self,
        sanitized_steps: List[Dict[str, Any]],
        lbp_iters: int,
        posterior_evidence: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:

        if self.gb is None:
            raise RuntimeError("[GraphServiceWorker.build_and_infer] 请先调用 init_model() 完成初始化。")

        def _build(prior_mode: str):
            self.gb.reset()
            prev_o = None
            for st in sanitized_steps:
                t = st["step"]
                s, a, o = f"S_{t}", f"A_{t}", f"O_{t}"

                self.gb.add_rv(s, 2, st["state_text"])
                self.gb.add_rv(a, 2, st["action_text"])
                self.gb.add_rv(o, 2, st["obs_text"])

                # priors
                if t == 0:
                    if prior_mode == "uniform":
                        p_s0 = np.array([0.5, 0.5], np.float32)
                    else:
                        p_s0 = np.array([1.0 - st["think_conf"], st["think_conf"]], np.float32)
                    self.gb.add_factor([s], f"phi_prior_{s}", potential=p_s0)

                if prior_mode == "uniform":
                    p_at = np.array([0.5, 0.5], np.float32)
                else:
                    p_at = np.array([1.0 - st["action_conf"], st["action_conf"]], np.float32)
                self.gb.add_factor([a], f"phi_prior_{a}", potential=p_at)

                # structure
                self.gb.add_factor([s, a], f"phi_sa_{t}", potential=None)
                self.gb.add_factor([a, o], f"phi_ao_{t}", potential=make_identity_obs(eps=self.graph_obs_eps))
                if prev_o is not None:
                    self.gb.add_factor([prev_o, s], f"phi_os_{t-1}", potential=None)
                prev_o = o

        # forward
        _build("real")
        self.gb.do_lbp(init=True, normalize=True, max_iters=lbp_iters)
        forward = {k: v.tolist() for k, v in self.gb.get_marginals(normalize=True).items()}

        # forward_no_evidence
        _build("uniform")
        self.gb.do_lbp(init=True, normalize=True, max_iters=lbp_iters)
        forward_no = {k: v.tolist() for k, v in self.gb.get_marginals(normalize=True).items()}

        # posterior
        _build("real")
        if posterior_evidence and posterior_evidence.get("enabled", False):
            k = len(sanitized_steps) - 1
            ans_is_true = posterior_evidence["answer_true"]
            s_max = float(posterior_evidence.get("s_max", 0.95))
            gamma = float(posterior_evidence.get("gamma", 0.75))
            s_min = float(posterior_evidence.get("s_min", 0.55))

            def make_e(s: float) -> np.ndarray:
                s = max(s_min, min(s, 0.999))
                return np.array([1.0 - s, s], np.float32) if ans_is_true else np.array([s, 1.0 - s], np.float32)

            for t in range(k, -1, -1):
                strength = s_max * (gamma ** (k - t))
                self.gb.add_factor([f"O_{t}"], f"evidence_O_{t}", potential=make_e(strength))

        self.gb.do_lbp(init=True, normalize=True, max_iters=lbp_iters)
        posterior = {k: v.tolist() for k, v in self.gb.get_marginals(normalize=True).items()}

        return {
            "forward_marginals": forward,
            "forward_no_evidence_marginals": forward_no,
            "posterior_marginals": posterior,
        }
