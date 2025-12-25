import sys
import os
os.environ["WANDB_DISABLED"] = "true"
os.environ['HF_ENDPOINT'] = ""
os.environ['HF_HOME'] = ''
os.environ['HF_HUB_CACHE'] = ""
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
import json

from build_trajectory_to_graph import GraphBuilder
from utils import (StopOnSequence,
                    compare_answers,
                    search,
                    fix_tags_tokenizer, 
                    extract_triples, 
                    make_identity_obs,
                    _to_float01)
from prompt import prompt_with_confidence
from hamster_tool.tools import open_jsonl

class EnhancedGraphBuilder(GraphBuilder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_nodes = {}  # {step_idx: [names...]}

    def remove_step_nodes(self, step_idx: int):
        if step_idx not in self.step_nodes:
            return

        remove_names = set(self.step_nodes.get(step_idx, []))

        try:
            rvs_dict = self.graph.get_rvs()  # {name: RV}
        except AttributeError:
            rvs_dict = self.graph._rvs
        all_names = set(rvs_dict.keys())
        keep_names = all_names - remove_names


        keep_context = {name: self.rv_to_context_map.get(name, "") for name in keep_names}


        new_graph = self.graph.__class__() 
        for name in sorted(keep_names):
            new_graph.rv(name, 2)

        try:
            factors = self.graph.get_factors()
        except AttributeError:
            factors = self.graph._factors

        for f in factors:
            rv_names = [rv.name for rv in f.get_rvs()]
            if all(n in keep_names for n in rv_names):

                pot = f.get_potential()
                new_graph.factor(rv_names, f.name, potential=pot)


        self.graph = new_graph

        new_step_nodes = {}
        for k, names in self.step_nodes.items():
            if k == step_idx:
                continue
            kept = [n for n in names if n in keep_names]
            if kept:
                new_step_nodes[k] = kept
        self.step_nodes = new_step_nodes

        self.rv_to_context_map = keep_context

    def add_rv(self, name: str, n_opts: int, rv_context: str):
        super().add_rv(name, n_opts, rv_context)
        try:
            step_idx = int(name.split('_')[1])
        except Exception:
            return
        self.step_nodes.setdefault(step_idx, []).append(name)

class OnlineInterventionController:
    def __init__(self, theta_S=0.5, theta_A=0.6, max_retries=3):
        self.theta_S = theta_S  
        self.theta_A = theta_A  
        self.max_retries = max_retries
        self.intervention_history = []
    
    def evaluate_step_risk(self, builder: GraphBuilder, step_idx: int) -> dict:
        marginals = builder.get_marginals(normalize=True)
        
        s_name = f"S_{step_idx}"
        a_name = f"A_{step_idx}"
        
        p_state = marginals.get(s_name, np.array([0.5, 0.5]))[1] 
        p_action = marginals.get(a_name, np.array([0.5, 0.5]))[1]  
        
        risk_level = "LOW"
        intervention_type = None
        
        if p_state < self.theta_S and p_action < self.theta_A:
            risk_level = "CRITICAL"
            intervention_type = "STATE_ACTION_INTERVENTION"
        elif p_state < self.theta_S:
            risk_level = "HIGH" 
            intervention_type = "STATE_INTERVENTION"
        elif p_action < self.theta_A:
            risk_level = "MEDIUM"
            intervention_type = "ACTION_INTERVENTION"
        
        return {
            "risk_level": risk_level,
            "intervention_type": intervention_type,
            "p_state": p_state,
            "p_action": p_action,
            "step_idx": step_idx
        }
    
    def generate_intervention_prompt(self, risk_assessment: dict, 
                                   current_action: Optional[str], 
                                   current_think: Optional[str]) -> str:
        intervention_type = risk_assessment["intervention_type"]
        
        if intervention_type == "STATE_INTERVENTION":
            return f"""
System: The current reasoning state appears uncertain (confidence: {risk_assessment['p_state']:.2f}). 
It seems the reasoning path may have deviated. Please reconsider the overall approach and ensure we're on the right track.

Previous thinking: {current_think}
Previous proposed action: {current_action}

Please provide a revised thinking process and action.
"""
        elif intervention_type == "ACTION_INTERVENTION":
            return f"""
System: The proposed action has low confidence (confidence: {risk_assessment['p_action']:.2f}). 
Please reconsider this specific step and propose an alternative action with better justification.

Previous thinking: {current_think}
Previous proposed action: {current_action}

Please think again and provide a better alternative.
"""
        elif intervention_type == "STATE_ACTION_INTERVENTION":
            return f"""
System: Critical uncertainty detected in both reasoning state and proposed action. 
Strong reconsideration is needed. Please step back and reassess the entire approach.

Previous thinking: {current_think}
Previous proposed action: {current_action}

Please provide a completely revised approach.
"""
        
        return "" 
    
def run_demo_with_online_intervention(
    question: str,
    golden_answer: str,
    model: Any,
    tokenizer: Any,
    stopping_criteria: Any,
    curr_eos: List[int],
    curr_search_template: str,
    *,
    builder: GraphBuilder,
    controller: OnlineInterventionController,
    obs_eps: float = 1e-6,
    lbp_iters: int = 10  
):
    prompt = prompt_with_confidence.format(question=question)
    device = model.device

    if getattr(tokenizer, "chat_template", None):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False
        )

    print('\n\n################# [Start Online Intervention Reasoning] ##################\n\n')
    
    sao = []
    step_history = []  
    base_prompt = prompt  
    
    for step_idx in range(20):  
        print(f"\n--- Step {step_idx} ---")
        
        retry_count = 0
        intervention_occurred = False
        
        while retry_count < controller.max_retries:
            current_prompt = base_prompt if retry_count == 0 else prompt
            think_content, action_content, confidence_content, should_terminate = (
                generate_single_step(model, tokenizer, current_prompt, stopping_criteria,
                curr_eos))
            
            if should_terminate:
                break
                
            s_text = f"think: {think_content}" if step_idx > 0 else f"question: {question}\nthink: {think_content}"
            a_text = action_content
            
            risk_assessment = update_graph_and_assess_risk(
                builder, controller, step_idx, s_text, a_text, confidence_content,
                step_history, obs_eps, lbp_iters, is_retry=(retry_count > 0)
            )
            
            print(f"Step {step_idx}, Retry {retry_count}: P(S)={risk_assessment['p_state']:.3f}, P(A)={risk_assessment['p_action']:.3f}")
            
            if risk_assessment["intervention_type"] is None:
                intervention_occurred = (retry_count > 0)  
                break
            else:
                intervention_prompt = controller.generate_intervention_prompt(
                    risk_assessment, a_text, think_content
                )
                prompt += intervention_prompt
                retry_count += 1
                print(f"Intervention needed: {risk_assessment['intervention_type']}")
        
        if should_terminate:
            o_text = ""
            final_answer = action_content
            break
            
        search_results = execute_action(action_content, tokenizer, model, curr_search_template)
        o_text = search_results if search_results else ""
        builder.rv_to_context_map[f"O_{step_idx}"] = o_text
                
        step_info = {
            "step_idx": step_idx,
            "S": s_text,
            "A": action_content,
            "O": o_text,
            "confidence": confidence_content,
            "risk_assessment": risk_assessment,
            "intervention_occurred": intervention_occurred,
            "retry_count": retry_count
        }
        step_history.append(step_info)
        
        search_text = curr_search_template.format(
            output_text=f"<think>{think_content}</think><action>{action_content}</action>",
            search_results=search_results
        )
        prompt += search_text
        
        if should_early_terminate(step_history, controller.theta_S):
            print("Early termination due to persistent low confidence")
            final_answer = "[UNABLE_TO_SOLVE]"
            break
    


def update_graph_and_assess_risk(builder, controller, step_idx, s_text, a_text, confidence_content, 
                               step_history, obs_eps, lbp_iters, is_retry=False):
    if is_retry:
        builder.remove_step_nodes(step_idx)

    s_name, a_name, o_name = f"S_{step_idx}", f"A_{step_idx}", f"O_{step_idx}"
    builder.add_rv(s_name, 2, s_text)
    builder.add_rv(a_name, 2, a_text)
    builder.add_rv(o_name, 2, "") 

    prior_conf = _to_float01(confidence_content, default=0.5)
    action_prior = np.array([1.0 - prior_conf, prior_conf], dtype=np.float32)

    builder.add_factor([s_name, a_name], f"act_{step_idx}", potential=None)
    builder.add_factor([a_name], f"prior_A{step_idx}", potential=action_prior)
    builder.add_factor([a_name, o_name], f"obs_{step_idx}", potential=make_identity_obs(eps=obs_eps))

    if step_idx > 0:
        prev_o_name = f"O_{step_idx-1}"
        builder.add_factor([prev_o_name, s_name], f"trans_{step_idx-1}", potential=None)

    builder.do_lbp(init=True, normalize=True, max_iters=lbp_iters)

    return controller.evaluate_step_risk(builder, step_idx)

def generate_single_step(model, tokenizer, prompt, stopping_criteria, curr_eos: List[int]):
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
    attention_mask = torch.ones_like(input_ids)
    
    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=1024, 
        stopping_criteria=stopping_criteria,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.7
    )
    
    generated_tokens = outputs[0][input_ids.shape[1]:]
    output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    parsed_dict = extract_triples(fix_tags_tokenizer(tokenizer, output_text.strip()))
    think_content = parsed_dict[0].get("think") if parsed_dict else ""
    action_content = parsed_dict[0].get("action") if parsed_dict else ""
    confidence_content = parsed_dict[0].get("confidence") if parsed_dict else ""

    should_terminate = outputs[0][-1].item() in curr_eos
    
    return think_content, action_content, confidence_content, should_terminate

def should_early_terminate(step_history, theta_S, window_size=3):
    if len(step_history) < window_size:
        return False
    
    recent_steps = step_history[-window_size:]
    low_confidence_count = sum(1 for step in recent_steps 
                              if step["risk_assessment"]["p_state"] < theta_S)
    
    return low_confidence_count == window_size

def execute_action(action_content, tokenizer, model, curr_search_template):

    text = action_content.strip().lower()
    if any(k in text for k in ["search"]):
        try:
            res = search(action_content)
        except Exception:
            res = ""
        return res or ""
    return ""


def start_demo(inference_data_path : str):
    data = open_jsonl(inference_data_path)
    model_id = "Qwen/Qwen2.5-14B-Instruct"
    curr_eos = [151645, 151643] # for Qwen2.5 series models
    curr_search_template = '\n\n{output_text}<information>{search_results}</information>\n\n'
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto")
    target_sequences = ["</search>", " </search>", "</search>\n", " </search>\n", "</search>\n\n", " </search>\n\n"]
    stopping_criteria = transformers.StoppingCriteriaList([StopOnSequence(target_sequences, tokenizer)])
    
    cpt_model_path = ""
    cpt_graph : EnhancedGraphBuilder = EnhancedGraphBuilder(cpt_model_path)
    controller = OnlineInterventionController(theta_S=0.5, theta_A=0.6, max_retries=3)
    
    for idx, d in tqdm(enumerate(data)):
        # try:
            cpt_graph.reset()
            question = d['question']
            golden_answers = d['golden_answers']
            output = run_demo_with_online_intervention(
                question, golden_answers,
                model, tokenizer, stopping_criteria, curr_eos, curr_search_template,
                builder=cpt_graph,
                controller=controller,
                obs_eps=1e-6,
                lbp_iters=5,
            )
        # except Exception as e:
        #     print(f"Error processing entry {idx}: {e}")
        #     continue
           
if __name__ == "__main__":
    inference_data_path = "run_data.jsonl"
    try:
        start_demo(inference_data_path)
    except KeyboardInterrupt as e:
        print("Process interrupted by user. Exiting...")
        sys.exit(0)