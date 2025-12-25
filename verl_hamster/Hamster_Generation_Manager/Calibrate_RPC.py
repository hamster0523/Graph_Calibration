import httpx
import numpy as np

class CalibrateRPCClient:
    def __init__(self, base_url: str, timeout: float = 3000.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def call_calibrate_trajectory(self, trajectory: dict, golden_answer: str | None):
        if isinstance(golden_answer, np.ndarray):
            if golden_answer.ndim == 0:
                golden_answer = golden_answer.item()
            else:
                golden_answer = golden_answer.tolist()
        payload = {"trajectory": trajectory, "golden_answers": golden_answer}
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(f"{self.base_url}", json=payload)
            resp.raise_for_status()
            return resp.json()
