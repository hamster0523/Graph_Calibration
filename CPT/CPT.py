import sys
import os
os.environ["WANDB_DISABLED"] = "true"
os.environ['HF_ENDPOINT'] = ""
os.environ['HF_HOME'] = ""
os.environ['HF_HUB_CACHE'] = ""
from typing import Tuple, Optional
import random
from pathlib import Path
import re
import numpy as np

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from hamster_tool.tools import open_jsonl
import numpy as np
import json

SA_PROMPT = """[Instruction]
Evaluate the quality of the proposed action given the historical context.

[Historical Context]
{history_text}

[Proposed Action]
{action_text}"""

HS_PROMPT = """[Instruction]
Evaluate the quality of the next state of reasoning given the historical context.

[Historical Context]
{history_text}

[Next State of Reasoning]
{next_state_text}"""

def extract_question(context_text: str) -> Optional[str]:

    if not isinstance(context_text, str) or not context_text.strip():
        return None

    m_outer = re.search(r"""["']context["']\s*:\s*(['"])(.*?)\1\s*$""", context_text, re.DOTALL)
    if m_outer:
        context_text = m_outer.group(2)

    pattern_line = re.compile(
        r'(?im)^[\t >\-•]*question\s*[:：]\s*(.+?)\s*$' 
    )
    m = pattern_line.search(context_text)
    if m:
        return m.group(1).strip()

    pattern_inline = re.compile(
        r'(?is)question\s*[:：]\s*(.+?)(?:\r?\n|$)'
    )
    m = pattern_inline.search(context_text)
    if m:
        return m.group(1).strip()

    return None

def fit_temp_bias(logits_np: np.ndarray, s_np: np.ndarray, steps: int = 800, lr: float = 0.05) -> Tuple[float, float]:
    d = torch.tensor(logits_np[:,1] - logits_np[:,0], dtype=torch.float32)  # [N]
    s = torch.tensor(s_np, dtype=torch.float32)
    
    T_param = torch.nn.Parameter(torch.tensor(1.0))  
    b_param = torch.nn.Parameter(torch.tensor(0.0))
    opt = torch.optim.Adam([T_param, b_param], lr=lr)
    
    for step in range(steps):
        opt.zero_grad()
        T = F.softplus(T_param) + 1e-6  
        p = torch.sigmoid((d + b_param) / T)
        
        loss = -(s * torch.log(p + 1e-12) + (1 - s) * torch.log(1 - p + 1e-12)).mean()
        loss.backward()
        opt.step()
        
        if (step + 1) % 100 == 0:
            print(f"Step {step+1}/{steps}, Loss: {loss.item():.6f}, T: {T.item():.4f}, b: {b_param.item():.4f}")
    
    T = float(F.softplus(T_param).item() + 1e-6)
    b = float(b_param.item())
    return T, b

def apply_calibration(logits: torch.Tensor, T: float, b: float) -> torch.Tensor:
    d = (logits[...,1] - logits[...,0])  # logit差值
    p1 = torch.sigmoid((d + b) / T)  # 校准后的正类概率
    return torch.stack([1 - p1, p1], dim=-1)  # [B,2]

class CPTModel:
    def __init__(self, model_path: str, do_calibrate : Optional[bool] = True ,calibration_datasets_path: Optional[str] = "./calibration_dataset.jsonl", 
                 calibration_params_path: Optional[str] = None, device: Optional[torch.device] = None,
                 rank: int = int(os.environ.get("CPT_MODEL_RANK", 3))):
        
        if not model_path:
            raise ValueError("model_path is required")
        
        if not Path(model_path).exists():
            raise ValueError(f"model_path {model_path} not exists")
        
        print(f"Rank {rank}: Initializing CPTModel with model: {model_path}")
        
        self.model_path = model_path
        self.rank = rank
        self.do_calibrate = do_calibrate
        
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device(f"cuda:{rank}")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device
            
        print(f"Rank {rank}: Using device: {self.device}")
        

        script_dir = Path(__file__).parent
        model_name = Path(model_path).name  
        self.auto_calibration_params_path = script_dir / f"cpt_calibration_params_{model_name}_rank{rank}.json"
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,        
                attn_implementation="sdpa",
                num_labels = 2,
                trust_remote_code=True
            )
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"Rank {rank}: Model loaded to device {self.device}")
        
        self.T = 1.0  
        self.b = 0.0  
        self.is_calibrated = False
        
    
        if calibration_datasets_path and Path(calibration_datasets_path).exists():
            self.calibration_datasets = open_jsonl(calibration_datasets_path)
            if len(self.calibration_datasets) > 500:
                self.calibration_datasets = random.sample(self.calibration_datasets, 500)
        else:
            self.calibration_datasets = None
        
        calibration_loaded = False
        
        if calibration_params_path and Path(calibration_params_path).exists():
            self.load_calibration_params(calibration_params_path)
            calibration_loaded = True
            print(f"Loaded calibration params from user-specified path: {calibration_params_path}")
        elif self.auto_calibration_params_path.exists():
            self.load_calibration_params(str(self.auto_calibration_params_path))
            calibration_loaded = True
            print(f"Loaded calibration params from script directory: {self.auto_calibration_params_path}")
        
        if not calibration_loaded and self.calibration_datasets:
            print("No existing calibration params found, training new calibration parameters...")
            self.train_calibration()
            self.save_calibration_params(str(self.auto_calibration_params_path))
            print(f"Calibration params auto-saved to: {self.auto_calibration_params_path}")
        
        print(f"Rank {rank}: CPTModel initialized successfully!")
    
    def train_calibration(self, steps: int = 800, lr: float = 0.05) -> None:
        if not self.calibration_datasets:
            print(f"Rank {self.rank}: No calibration datasets available for training calibration parameters.")
            return
        
        print(f"Rank {self.rank}: Training calibration parameters on {len(self.calibration_datasets)} samples...")
        
        logits_list = []
        scores_list = []
        
        for item in self.calibration_datasets:
            context = item.get('context', '')
            llm_score = item.get('llm_score', item.get('score', 0.5))  
            
            with torch.no_grad():
                messages = [
                    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                    {"role": "user", "content": context},
                ]
                text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                enc = self.tokenizer([text], return_tensors="pt").to(self.device)
                out = self.model(**enc)
                logits = out.logits.float().cpu().numpy()[0] 
                
                logits_list.append(logits)
                scores_list.append(llm_score)
        
        if len(logits_list) == 0:
            print(f"Rank {self.rank}: No valid calibration samples found.")
            return
        
        logits_np = np.stack(logits_list)  
        scores_np = np.array(scores_list)  
        
        self.T, self.b = fit_temp_bias(logits_np, scores_np, steps=steps, lr=lr)
        self.is_calibrated = True
        
        print(f"Rank {self.rank}: Calibration training completed. T={self.T:.4f}, b={self.b:.4f}")
    
    def save_calibration_params(self, save_path: Optional[str] = None) -> None:
        if save_path is None:
            save_path = str(self.auto_calibration_params_path)
        
        params = {
            'T': self.T,
            'b': self.b,
            'is_calibrated': self.is_calibrated,
            'model_path': self.model_path,  
            'rank': self.rank,
            'device': str(self.device),
            'save_time': str(Path(save_path).stat().st_mtime if Path(save_path).exists() else 'new')
        }
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(params, f, indent=2)
        print(f"Rank {self.rank}: Calibration parameters saved to {save_path}")
    
    def load_calibration_params(self, load_path: str) -> None:
        with open(load_path, 'r') as f:
            params = json.load(f)
        
        saved_model_path = params.get('model_path')
        if saved_model_path and saved_model_path != self.model_path:
            print(f"Rank {self.rank}: Warning: Calibration params were trained on {saved_model_path}, "
                  f"but current model is {self.model_path}")
        
        self.T = params.get('T', 1.0)
        self.b = params.get('b', 0.0)
        self.is_calibrated = params.get('is_calibrated', False)
        print(f"Rank {self.rank}: Calibration parameters loaded: T={self.T:.4f}, b={self.b:.4f}")
    
    def has_cached_calibration(self) -> bool:
        return self.auto_calibration_params_path.exists()
    
    def retrain_calibration(self, steps: int = 800, lr: float = 0.05, auto_save: bool = True) -> None:
        print(f"Rank {self.rank}: Retraining calibration parameters...")
        self.train_calibration(steps, lr)
        
        if auto_save:
            self.save_calibration_params()
            print(f"Rank {self.rank}: Updated calibration params saved to: {self.auto_calibration_params_path}")
    
    def clear_cached_calibration(self) -> None:
        """清除缓存的校准参数文件"""
        if self.auto_calibration_params_path.exists():
            self.auto_calibration_params_path.unlink()
            print(f"Rank {self.rank}: Cleared cached calibration params: {self.auto_calibration_params_path}")
            self.T = 1.0
            self.b = 0.0
            self.is_calibrated = False
        else:
            print(f"Rank {self.rank}: No cached calibration params to clear")

    def predict(self, history_context : str, next_step_context : str, context_type : str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if context_type.lower() == "sa":
            prompt = SA_PROMPT.format(history_text=history_context, action_text=next_step_context)
        elif context_type.lower() == "os":
            prompt = HS_PROMPT.format(history_text=history_context, next_state_text=next_step_context)
        else:
            raise ValueError(f"Unsupported context_type {context_type}")
        
        before_calibrate_logits = None
        after_calibrate_probs = None
        with torch.no_grad():
            messages = [
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            enc = self.tokenizer([text], return_tensors="pt").to(self.device)
            out = self.model(**enc)
            probs = torch.softmax(out.logits.float(), dim=-1)
            before_calibrate_logits = probs.cpu().numpy()
        
        if self.do_calibrate and self.is_calibrated:
            probs = self._calibrate_probs(probs)
            after_calibrate_probs = probs.cpu().numpy()
        
        return (before_calibrate_logits, after_calibrate_probs)
    
    def _calibrate_probs(self, probs: torch.Tensor) -> torch.Tensor:
        if not self.is_calibrated:
            return probs
        
        eps = 1e-8
        log_probs = torch.log(probs + eps)
        d_approx = log_probs[..., 1] - log_probs[..., 0]
        
        p1_calibrated = torch.sigmoid((d_approx + self.b) / self.T)
        calibrated_probs = torch.stack([1 - p1_calibrated, p1_calibrated], dim=-1)
        
        return calibrated_probs
    
    def predict_test(self, context : str) -> np.ndarray:
        with torch.no_grad():
            messages = [
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                {"role": "user", "content": context},
            ]
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            enc = self.tokenizer([text], return_tensors="pt").to(self.device)
            out = self.model(**enc)
            logits = out.logits.float()
        
        if self.is_calibrated and self.do_calibrate:
            probs = torch.softmax(logits, dim=-1)
            calibrated_probs = self._calibrate_probs(probs)
            eps = 1e-8
            calibrated_logits = torch.log(calibrated_probs + eps)
            return calibrated_logits.cpu().numpy()
        else:
            return logits.cpu().numpy()