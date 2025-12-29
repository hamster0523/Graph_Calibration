
import binascii
import sys
import os
import asyncio
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field
from pydantic import ConfigDict  
import numpy as np

from GraphBuilder import GraphBuilder
from utils import make_identity_obs, compare_answers

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
GRAPH_LBP_ITERS = int(os.getenv("GRAPH_LBP_ITERS", "100"))
GRAPH_OBS_EPS   = float(os.getenv("GRAPH_OBS_EPS", "1e-6"))
CPT_MODEL_PATH  = os.getenv("CPT_MODEL_PATH", "")
CPT_DO_CALIB    = os.getenv("CPT_DO_CALIB", "false").lower() == "true"


class StepIn(BaseModel):
    model_config = ConfigDict(extra='ignore')  
    step: Optional[int] = None
    state: Optional[str] = ""
    action: Optional[str] = ""
    observation: Optional[str] = ""
    action_type: Optional[Union[str, int, None]] = None 
    think_confidence: Optional[float] = 0.5
    action_confidence: Optional[float] = 0.5

class TrajectoryIn(BaseModel):
    model_config = ConfigDict(extra='ignore')  
    question: Optional[str] = None
    prompt: Optional[str] = None
    final_answer: Optional[str] = None
    steps: List[StepIn] = Field(default_factory=list)

class CalibrateRequest(BaseModel):
    model_config = ConfigDict(extra='ignore', populate_by_name=True)  
    trajectory: TrajectoryIn

    golden_answer: Optional[Union[str, None]] = None
    golden_answers: Optional[List[str]] = Field(default=None, alias="golden_answers")

    def get_normalized_golden_answer(self) -> Optional[Union[str, List[str]]]:
        if self.golden_answer is not None:
            return self.golden_answer
        if self.golden_answers:
            return self.golden_answers  
        return None

app = FastAPI(title="CPT-Graph Calibrate Service")

MAX_PREVIEW = 4096  
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    parsed_body = None
    raw_preview = None
    try:
        parsed_body = await request.json()  
    except Exception:
        try:
            raw = await request.body()  
            raw_preview = raw.decode("utf-8", errors="replace")
            if len(raw_preview) > MAX_PREVIEW:
                raw_preview = raw_preview[:MAX_PREVIEW] + "...<truncated>"
        except Exception:
            try:
                raw = await request.body()
                raw_preview = binascii.hexlify(raw).decode("ascii")
                if len(raw_preview) > MAX_PREVIEW:
                    raw_preview = raw_preview[:MAX_PREVIEW] + "...<truncated-hex>"
            except Exception:
                raw_preview = "<unreadable body>"

    content = {
        "detail": exc.errors(),         
        "path": request.url.path,
        "method": request.method,
        "body_json": parsed_body if parsed_body is not None else None,
        "body_text": None if parsed_body is not None else raw_preview,
    }
    return JSONResponse(status_code=422, content=content)

_global_lock = asyncio.Lock()
_graph_builder = GraphBuilder(
    cpt_model_path=CPT_MODEL_PATH,
    cpt_do_calibration=CPT_DO_CALIB,
)

GRAPH_EVIDENCE_STRENGTH = float(os.getenv("GRAPH_EVIDENCE_STRENGTH", "0.95"))
BACK_GAMMA = float(os.getenv("BACK_EVIDENCE_GAMMA", "0.75"))
BACK_S_MIN = float(os.getenv("BACK_EVIDENCE_S_MIN", "0.55"))

def _inject_exponential_back_evidence(
    gb: GraphBuilder,
    num_steps: int,
    answer_true_or_false: Optional[bool],
    s_max: float,
    gamma: float,
    s_min: float,
    inject_A: bool,
    inject_S: bool,
    alpha_A: float,
    alpha_S: float,
):
    if answer_true_or_false is None:
        return
    for t in range(num_steps):
        k = num_steps - 1 - t
        s = max(s_min, s_max * (gamma ** k))
        o_name = f"O_{t}"
        if answer_true_or_false:
            pot = np.array([1.0 - s, s], dtype=np.float32)
        else:
            pot = np.array([s, 1.0 - s], dtype=np.float32)
        gb.add_factor([o_name], f"phi_evd_{o_name}", potential=pot)

        if inject_A:
            a_name = f"A_{t}"
            if answer_true_or_false:
                pA = np.array([1.0 - alpha_A * s, alpha_A * s], np.float32)
            else:
                pA = np.array([alpha_A * s, 1.0 - alpha_A * s], np.float32)
            gb.add_factor([a_name], f"phi_evd_{a_name}", potential=pA)

        if inject_S:
            s_name = f"S_{t}"
            if answer_true_or_false:
                pS = np.array([1.0 - alpha_S * s, alpha_S * s], np.float32)
            else:
                pS = np.array([alpha_S * s, 1.0 - alpha_S * s], np.float32)
            gb.add_factor([s_name], f"phi_evd_{s_name}", potential=pS)

def _calibrate_trajectory(
    graph_builder: GraphBuilder,
    trajectory: Dict[str, Any],
    golden_answer: Optional[Union[str, List[str]]],  
    graph_lbp_iters: int,
    graph_obs_eps: float,
) -> Optional[Dict[str, Any]]:
    if not trajectory.get("steps"):
        return None

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

    gb: GraphBuilder = graph_builder
    if gb is None:
        return None

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
                p_s0 = np.array(
                    [1.0 - st["think_conf"], st["think_conf"]], np.float32
                ) if prior_mode == "real" else np.array([0.5, 0.5], np.float32)
                gb.add_factor([s], f"phi_prior_{s}", potential=p_s0)

            p_at = np.array(
                [1.0 - st["action_conf"], st["action_conf"]], np.float32
            ) if prior_mode == "real" else np.array([0.5, 0.5], np.float32)
            gb.add_factor([a], f"phi_prior_{a}", potential=p_at)

            gb.add_factor([s, a], f"phi_sa_{t}", potential=None)
            gb.add_factor([a, o], f"phi_ao_{t}", potential=make_identity_obs(eps=graph_obs_eps))
            if prev_o is not None:
                gb.add_factor([prev_o, s], f"phi_os_{t-1}", potential=None)
            prev_o = o

    build_graph_inplace(prior_mode="real")
    gb.do_lbp(init=True, normalize=True, max_iters=graph_lbp_iters)
    forward_marginals = {name: p.tolist() for name, p in gb.get_marginals(normalize=True).items()}

    build_graph_inplace(prior_mode="uniform")
    gb.do_lbp(init=True, normalize=True, max_iters=graph_lbp_iters)
    forward_no_evidence_marginals = {name: p.tolist() for name, p in gb.get_marginals(normalize=True).items()}

    build_graph_inplace(prior_mode="real")
    agent_answer = trajectory.get("final_answer")

    expected = golden_answer
    answer_true_or_false = compare_answers(expected_answer=expected, agent_answer=agent_answer)

    _inject_exponential_back_evidence(
        gb=gb,
        num_steps=len(trajectory["steps"]),
        answer_true_or_false=answer_true_or_false,
        s_max=GRAPH_EVIDENCE_STRENGTH,
        gamma=BACK_GAMMA,
        s_min=BACK_S_MIN,
        inject_A=False,
        inject_S=False,
        alpha_A=0.50,
        alpha_S=0.25,
    )

    gb.do_lbp(init=True, normalize=True, max_iters=graph_lbp_iters)
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

@app.post("/calibrate")
async def calibrate(req: CalibrateRequest):
    async with _global_lock:
        out = _calibrate_trajectory(
            graph_builder=_graph_builder,
            trajectory=req.trajectory.model_dump(),
            golden_answer=req.get_normalized_golden_answer(),
            graph_lbp_iters=GRAPH_LBP_ITERS,
            graph_obs_eps=GRAPH_OBS_EPS,
        )
    return out or {}
