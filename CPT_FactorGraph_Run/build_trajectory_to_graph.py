import sys
import os
os.environ["WANDB_DISABLED"] = "true"
os.environ['HF_ENDPOINT'] = ""
os.environ['HF_HOME'] = ""
os.environ['HF_HUB_CACHE'] = ""
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, Qwen2ForCausalLM, GenerationConfig
import torch
import numpy as np
import json

from hamster_factor_graph import Graph
from CPT.CPT import CPTModel
from .utils import (StopOnSequence,
                    compare_answers,
                    search,
                    fix_tags_tokenizer_verbal, 
                    fix_tags_tokenizer_non_verbal,
                    extract_triples,
                    get_query,
                    make_identity_obs,
                    ensure_cpt_2x2,
                    _to_float01,
                    split_tokens_and_scores, split_tokens_and_scores_with_confidence,
                    split_tokens_and_logits_with_probs,
                    extract_tagged_spans_with_probs)
from .prompt import prompt_with_confidence, prompt_no_confidence

# RV : Random Variable
class GraphBuilder:
    def __init__(self, cpt_model_path: str, cpt_do_calibration: Optional[bool] = True,
                 sharpen_strength: float = 3.0,  
                 min_assoc_strength: float = 0.15):  
        self.graph: Graph = Graph()
        self.CPT: CPTModel = CPTModel(cpt_model_path, do_calibrate=cpt_do_calibration)
        self.rv_to_context_map: Dict[str, str] = {}
        self.sharpen_strength = sharpen_strength
        self.min_assoc_strength = min_assoc_strength
        self.graph_has_init = False
    
    def add_rv(self, name : str, n_opts : int, rv_context : str):
        self.graph.rv(name, n_opts)
        self.rv_to_context_map[name] = rv_context
    
    def add_factor(self, rv_names : List[str], factor_name : str, potential : Optional[np.ndarray]):
        if len(rv_names) == 1:
            if potential is None:
                raise ValueError("Prior factor must have a specified potential.")
            self.graph.factor(rv_names, factor_name, potential=potential)
        else:
            before_potential = None
            after_potential = None
            if potential is None:
                before_potential, after_potential = self._get_potential_from_CPT(rv_names)
                before_potential = self._format_potential(before_potential)
                potential = before_potential
                if after_potential is not None:
                    after_potential = self._format_potential(after_potential)
                    potential = after_potential
                    
            if (potential[0,0] + potential[1,1]) < (potential[0,1] + potential[1,0]):
                potential = potential[:, ::-1]
                after_potential = potential

            # if self._assoc_strength(potential) < 0.12:           
            #     potential = self._sharpen_rows(potential, gamma=2.0)   
            #     after_potential = potential
            # if self._assoc_strength(potential) < 0.08:          
            #     potential = self._mix_with_identity(potential, lam=0.7, eps=0.05, gamma=2.0) 
            #     after_potential = potential   
            if self._assoc_strength(potential) < self.min_assoc_strength:           
                potential = self._sharpen_rows(potential, gamma=self.sharpen_strength)   
                after_potential = potential
                
            if self._assoc_strength(potential) < 0.08:          
                potential = self._mix_with_identity(potential, lam=0.5, eps=0.02, gamma=self.sharpen_strength) 
                after_potential = potential
            
            #print(f"[DEBUG] {rv_names[0]}->{rv_names[1]}")
            #print(f"before_potential:\n{before_potential}\nafter_potential:\n{after_potential}\n")
            
            self.graph.factor(rv_names, factor_name, potential=potential)
        
    def _sharpen_rows(self, p: np.ndarray, gamma: float = 2.0) -> np.ndarray:
        p = np.clip(p, 1e-12, 1.0)
        p = p ** gamma
        return p / p.sum(axis=1, keepdims=True)
    
    def _assoc_strength(self, p: np.ndarray) -> float:
        return 0.5 * (abs(p[0,0] - p[0,1]) + abs(p[1,1] - p[1,0]))
    
    def _mix_with_identity(self, p: np.ndarray, lam: float = 0.7, eps: float = 0.05, gamma: float = 2.0):
        ident = np.array([[1-eps, eps],[eps,1-eps]], np.float32)
        ident = self._sharpen_rows(ident, gamma)
        mixed = lam * p + (1 - lam) * ident
        return mixed / mixed.sum(axis=1, keepdims=True)

    def _get_potential_from_CPT(self, rv_names : List[str]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        assert len(rv_names) == 2, "Currently only supports binary factors (2 RVs)."
        rv_name_1, rv_name_2 = rv_names[0], rv_names[1]

        l1, l2 = rv_name_1.lower(), rv_name_2.lower()
        if ('s' in l1) and ('a' in l2):
            context_type = "sa"
        elif ('o' in l1) and ('s' in l2):
            context_type = "os"
        else:
            raise ValueError(f"Unsupported factor pattern for CPT: {rv_name_1} -> {rv_name_2}. "
                            f"Only support S->A and O->S.")

        history_context     = self.rv_to_context_map[rv_name_1]
        next_step_context   = self.rv_to_context_map[rv_name_2]

        before_potential, after_potential = self.CPT.predict(history_context, next_step_context, context_type)
        return before_potential, after_potential

    def _format_potential(self, prob_or_cpt : np.ndarray) -> np.ndarray:
        arr = np.squeeze(np.asarray(prob_or_cpt, dtype=np.float32))
        if arr.shape == (2,):
            return ensure_cpt_2x2(arr)     
        if arr.shape == (2, 2):
            return ensure_cpt_2x2(arr)
        raise ValueError(f"Potential shape must be (2,) or (2,2); got {arr.shape}")
    
    def _init_graph(self):
        nodes = self.graph._sorted_nodes()
        self.graph.init_messages(nodes)

    def do_lbp(self, init : bool = False, normalize : bool = True, max_iters : int = 500, progress : bool = False) -> None:
        if not self.graph_has_init:
            self._init_graph()
            self.graph_has_init = True
        self.graph.lbp(init=init, normalize=normalize, max_iters=max_iters, progress=progress)
        
    def get_marginals(self, normalize : bool = True) -> Dict[str, np.ndarray]:
        if not self.graph_has_init:
            self._init_graph()
            self.graph_has_init = True
        return {rv.name: p for rv, p in self.graph.rv_marginals(normalize=normalize)} 
    
    def get_answer_state(self) -> str:
        answer_rvs = [rv for rv in self.rv_to_context_map.keys() if rv.startswith("O_")]
        if not answer_rvs:
            raise ValueError("No answer RVs found in the graph.")
        latest_answer_rv = sorted(answer_rvs, key=lambda rv: int(rv.split("_")[1]))[-1]
        return latest_answer_rv  

    def reset(self):
        self.graph = Graph()
        self.rv_to_context_map = {}
        self.graph_has_init = False
    
def run_demo(
    question: str,
    golden_answer : str,
    model: Any,
    tokenizer: Any,
    stopping_criteria: transformers.StoppingCriteria,
    curr_eos: List[int],
    curr_search_template: str) :
    
    prompt = prompt_with_confidence.format(question=question)
    device = model.device

    if getattr(tokenizer, "chat_template", None):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False
        )

    print('\n\n################# [Start Reasoning + Searching] ##################\n\n')
    
    sao = []
    cnt = 0
    answer = None

    while True:
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        attention_mask = torch.ones_like(input_ids)

        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1024,
            stopping_criteria=stopping_criteria,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.8
        )

        generated_tokens = outputs[0][input_ids.shape[1]:]
        output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        print(f"before fix : {output_text}\n")

        to_parse_output_text = fix_tags_tokenizer(tokenizer, output_text.strip().strip("/n"))
        print(f"after fix: {to_parse_output_text}\n")
        parsed_dict = extract_triples(to_parse_output_text)
        think_content = parsed_dict[0].get("think") if parsed_dict else None
        confidence_content = parsed_dict[0].get("confidence") if parsed_dict else None
        action_content = parsed_dict[0].get("action") if parsed_dict else None
        action_type = parsed_dict[0].get("action_type") if parsed_dict else None
        
        s = "think : " + (think_content if think_content else "")
        a = action_content if action_content else ""
        
        if cnt == 0:
            s = "question : " + question + "\n" + "think : " + (think_content if think_content else "")
        
        if outputs[0][-1].item() in curr_eos:
            prompt += "\n" + output_text
            o = None
            temp_sao = {
                "S" : s,
                "A" : {
                    "action": a,
                    "action_type": action_type,
                    "confidence": confidence_content
                },
                "O" : o
            }
            answer = a
            sao.append(temp_sao)
            break

        tmp_query = get_query(tokenizer.decode(outputs[0], skip_special_tokens=True))
        if tmp_query:
            search_results = search(tmp_query)
        else:
            search_results = ''
        
        o = search_results if search_results else ""

        search_text = curr_search_template.format(
            output_text=output_text,
            search_results=search_results
        )
        
        temp_sao = {
            "S" : s,
            "A" : {
                "action": a,
                "action_type": action_type,
                "confidence": confidence_content
            },
            "O" : o
        }
        
        sao.append(temp_sao)

        prompt += search_text
        cnt += 1
        
def get_confidence_safe(result, default=0.5):
    if result and "parts" in result and len(result["parts"]) > 0:
        return result["parts"][0].get("confidence", default)
    return default

def run_demo_with_graph_inner_confidence(
    question: str,
    golden_answer : str,
    model: Any,
    tokenizer: Any,
    stopping_criteria: Any,
    curr_eos: List[int],
    curr_search_template: str,
    *,
    builder: GraphBuilder,                   
    obs_eps: float = 1e-6,
    lbp_iters: int = 5,
    do_lbp_each_step: bool = True,
    evidence_strength: float = 0.999
):
    prompt = prompt_no_confidence.format(question=question)
    device = model.device

    if getattr(tokenizer, "chat_template", None):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False
        )

    print('\n\n################# [Start Reasoning + Searching] ##################\n\n')

    sao, cnt, answer = [], 0, None
    prev_o_name: Optional[str] = None  

    while True:
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        attention_mask = torch.ones_like(input_ids)

        generation_config = GenerationConfig(
            output_logits = True,
            return_dict_in_generate = True
        )

        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1024,
            stopping_criteria=stopping_criteria,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.8,
            generation_config=generation_config
        )

        generated_ids = outputs['sequences']
        generated_tokens = generated_ids[0][input_ids.shape[1]:]
        output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        print(f"before fix : {output_text}\n")

        token_logits = outputs['logits']
        think_scores  = extract_tagged_spans_with_probs(tokenizer, generated_tokens, token_logits, "<think>", "</think>")
        search_scores = extract_tagged_spans_with_probs(tokenizer, generated_tokens, token_logits, "<search>", "</search>") 
        answer_scores = extract_tagged_spans_with_probs(tokenizer, generated_tokens, token_logits, "<answer>", "</answer>")

        think_confidence  = get_confidence_safe(think_scores)
        search_confidence = get_confidence_safe(search_scores)
        answer_confidence = get_confidence_safe(answer_scores)

        to_parse_output_text = fix_tags_tokenizer_non_verbal(tokenizer, output_text.strip().strip("/n"))
        parsed_dict = extract_triples(to_parse_output_text)
        think_content = parsed_dict[0].get("think") if parsed_dict else None
        action_content = parsed_dict[0].get("action") if parsed_dict else None
        action_type = parsed_dict[0].get("action_type") if parsed_dict else None

        s_text = "think : " + (think_content if think_content else "")
        a_text = action_content if action_content else ""
        if cnt == 0:
            s_text = f"question : {question}\nthink : {think_content or ''}"

        if generated_ids[0][-1].item() in curr_eos:
            prompt += "\n" + output_text
            o_text = ""  

            temp_sao = {"S": s_text, "A": {"action": a_text, "action_type": action_type, "confidence": answer_confidence}, "O": None}
            answer = a_text
            sao.append(temp_sao)

            s_name, a_name, o_name = f"S_{cnt}", f"A_{cnt}", f"O_{cnt}"
            builder.add_rv(s_name, 2, s_text)
            builder.add_rv(a_name, 2, a_text)
            builder.add_rv(o_name, 2, o_text)

            if cnt == 0:
                builder.add_factor([s_name], f"phi_prior_{s_name}", potential=np.array([1.0 - think_confidence, think_confidence], dtype=np.float32))

            builder.add_factor([s_name, a_name], f"act_{cnt}", potential=None)
            builder.add_factor([a_name], f"prior_A{cnt}", potential=np.array([1.0 - answer_confidence, answer_confidence], dtype=np.float32))
            builder.add_factor([a_name, o_name], f"obs_{cnt}", potential=make_identity_obs(eps=obs_eps))

            if prev_o_name is not None:
                builder.add_factor([prev_o_name, s_name], f"trans_{cnt-1}", potential=None)

            if do_lbp_each_step:
                builder.do_lbp(init=True, normalize=True, max_iters=lbp_iters)
            break

        tmp_query = get_query(tokenizer.decode(outputs['sequences'][0], skip_special_tokens=True))
        search_results = search(tmp_query) if tmp_query else ''
        o_text = search_results or ""
        search_text = curr_search_template.format(output_text=output_text, search_results=search_results)

        temp_sao = {"S": s_text, "A": {"action": a_text, "action_type": action_type, "confidence": search_confidence}, "O": o_text}
        sao.append(temp_sao)

        s_name, a_name, o_name = f"S_{cnt}", f"A_{cnt}", f"O_{cnt}"
        builder.add_rv(s_name, 2, s_text)
        builder.add_rv(a_name, 2, a_text)
        builder.add_rv(o_name, 2, o_text)

        if cnt == 0:
            builder.add_factor([s_name], f"phi_prior_{s_name}", potential=np.array([1.0 - think_confidence, think_confidence], dtype=np.float32))

        builder.add_factor([s_name, a_name], f"act_{cnt}", potential=None)
        builder.add_factor([a_name], f"prior_A{cnt}", potential=np.array([1.0 - search_confidence, search_confidence], dtype=np.float32))
        builder.add_factor([a_name, o_name], f"obs_{cnt}", potential=make_identity_obs(eps=obs_eps))

        if prev_o_name is not None:
            builder.add_factor([prev_o_name, s_name], f"trans_{cnt-1}", potential=None)

        if do_lbp_each_step:
            builder.do_lbp(init=True, normalize=True, max_iters=lbp_iters)

        prev_o_name = o_name
        prompt += search_text
        cnt += 1

    builder.do_lbp(init=True, normalize=True)

    print("=== Forward (no evidence) ===")
    for name, marg in sorted(builder.get_marginals(normalize=True).items()):
        print(f"{name}: {marg}")

    is_correct = compare_answers(golden_answer, answer)
    print(f"Is the answer correct? {is_correct}")

    if is_correct:
        observation_state = np.array([1.0 - evidence_strength, evidence_strength], dtype=np.float32)
    else:
        observation_state = np.array([evidence_strength, 1.0 - evidence_strength], dtype=np.float32)
    builder.add_factor([builder.get_answer_state()], f"evid_{builder.get_answer_state()}", potential=observation_state)

    builder.do_lbp(init=True, normalize=True)

    print("=== Posterior (with evidence) ===")
    for name, marg in sorted(builder.get_marginals(normalize=True).items()):
        print(f"{name}: {marg}")

    builder.reset()
    return {"sao": sao, "answer": answer, "graph": builder.graph}
    
def run_demo_with_graph_verbal_confidence(
    question: str,
    golden_answer : str,
    model: Any,
    tokenizer: Any,
    stopping_criteria: Any,
    curr_eos: List[int],
    curr_search_template: str,
    *,
    builder: GraphBuilder,                   
    prior_s0: np.ndarray = np.array([0.5, 0.5], dtype=np.float32),
    obs_eps: float = 1e-6,
    lbp_iters: int = 5,
    do_lbp_each_step: bool = True,
    evidence_strength: float = 0.999
):
    prompt = prompt_with_confidence.format(question=question)
    device = model.device

    if getattr(tokenizer, "chat_template", None):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False
        )

    print('\n\n################# [Start Reasoning + Searching] ##################\n\n')
    
    sao = []
    cnt = 0
    answer = None

    prior_s0 = (prior_s0 / max(prior_s0.sum(), 1e-12)).astype(np.float32)
    prev_o_name: Optional[str] = None  

    while True:
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        attention_mask = torch.ones_like(input_ids)
        
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1024,
            stopping_criteria=stopping_criteria,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.8
        )

        generated_tokens = outputs[0][input_ids.shape[1]:]
        output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        print(f"before fix : {output_text}\n")

        to_parse_output_text = fix_tags_tokenizer(tokenizer, output_text.strip().strip("/n"))
        print(f"after fix: {to_parse_output_text}\n")
        parsed_dict = extract_triples(to_parse_output_text)
        think_content = parsed_dict[0].get("think") if parsed_dict else None
        confidence_content = parsed_dict[0].get("confidence") if parsed_dict else None
        action_content = parsed_dict[0].get("action") if parsed_dict else None
        action_type = parsed_dict[0].get("action_type") if parsed_dict else None
        
        s_text = "think : " + (think_content if think_content else "")
        a_text = action_content if action_content else ""
        if cnt == 0:
            s_text = "question : " + question + "\n" + "think : " + (think_content if think_content else "")
        
        if outputs[0][-1].item() in curr_eos:
            prompt += "\n" + output_text
            o_text = ""  

            temp_sao = {
                "S" : s_text,
                "A" : {
                    "action": a_text,
                    "action_type": action_type,
                    "confidence": confidence_content
                },
                "O" : None
            }
            answer = a_text
            sao.append(temp_sao)

            s_name, a_name, o_name = f"S_{cnt}", f"A_{cnt}", f"O_{cnt}"
            builder.add_rv(s_name, 2, s_text)
            builder.add_rv(a_name, 2, a_text)
            builder.add_rv(o_name, 2, o_text)

            # if cnt == 0:
            #     builder.add_factor([s_name], f"phi_prior_{s_name}", potential=prior_s0)

            # state to action
            builder.add_factor([s_name, a_name], f"act_{cnt}", potential=None)
            
            # action prior
            prior_confidence = _to_float01(confidence_content, default=0.5)
            action_prior = [1.0 - prior_confidence, prior_confidence]
            builder.add_factor([a_name], f"prior_A{cnt}", potential=np.array(action_prior, dtype=np.float32))

            # action to obersvation
            pot_AO = make_identity_obs(eps=obs_eps)
            builder.add_factor([a_name, o_name], f"obs_{cnt}", potential=pot_AO)

            # previous observation to current state
            if prev_o_name is not None:
                builder.add_factor([prev_o_name, s_name], f"trans_{cnt-1}", potential=None)

            if do_lbp_each_step:
                #g.lbp(init=(not lbp_inited), normalize=True, max_iters=lbp_iters)
                builder.do_lbp(init=True, normalize=True, max_iters=lbp_iters)

            break

        # if int(confidence_content, 10) < 0.5:
        #     print(f"Low confidence ({confidence_content}).")
        #     prompt += "My confidence in this search is too low; perhaps I should perform a more fine-grained search.\n"
        #     continue

        tmp_query = get_query(tokenizer.decode(outputs[0], skip_special_tokens=True))
        search_results = search(tmp_query) if tmp_query else ''
        o_text = search_results if search_results else ""

        search_text = curr_search_template.format(
            output_text=output_text,
            search_results=search_results
        )
        
        temp_sao = {
            "S" : s_text,
            "A" : {
                "action": a_text,
                "action_type": action_type,
                "confidence": confidence_content
            },
            "O" : o_text
        }
        sao.append(temp_sao)

        s_name, a_name, o_name = f"S_{cnt}", f"A_{cnt}", f"O_{cnt}"
        builder.add_rv(s_name, 2, s_text)
        builder.add_rv(a_name, 2, a_text)
        builder.add_rv(o_name, 2, o_text)

        if cnt == 0:
            builder.add_factor([s_name], f"phi_prior_{s_name}", potential=prior_s0)

        builder.add_factor([s_name, a_name], f"act_{cnt}", potential=None)

        prior_confidence = _to_float01(confidence_content, default=0.5)
        action_prior = np.array([1.0 - prior_confidence, prior_confidence], dtype=np.float32)
        builder.add_factor([a_name], f"prior_A{cnt}", potential=action_prior)

        pot_AO = make_identity_obs(eps=obs_eps)
        builder.add_factor([a_name, o_name], f"obs_{cnt}", potential=pot_AO)

        if prev_o_name is not None:
            builder.add_factor([prev_o_name, s_name], f"trans_{cnt-1}", potential=None)

        if do_lbp_each_step:
            #g.lbp(init=(not lbp_inited), normalize=True, max_iters=lbp_iters)
            builder.do_lbp(init=True, normalize=True, max_iters=lbp_iters)

        prev_o_name = o_name

        prompt += search_text
        cnt += 1
    
    builder.do_lbp(init=True, normalize=True)
    
    before_marg = builder.get_marginals(normalize=True)
    print("=== Forward (no evidence) ===")
    for name in sorted(before_marg.keys()):
        print(f"{name}: {before_marg[name]}")
    
    is_correct = compare_answers(golden_answer, answer)
    print(f"Is the answer correct? {is_correct}")
    
    #observation_state = np.array([0.05, 0.95], dtype=np.float32) if is_correct else np.array([0.95, 0.05], dtype=np.float32)
    if is_correct:
        observation_state = np.array([1.0 - evidence_strength, evidence_strength], dtype=np.float32)
    else:
        observation_state = np.array([evidence_strength, 1.0 - evidence_strength], dtype=np.float32)
    answer_rv_name = builder.get_answer_state()
    builder.add_factor([answer_rv_name], f"evid_{answer_rv_name}", potential=observation_state)
    
    builder.do_lbp(init=True, normalize=True)
    
    after_marg = builder.get_marginals(normalize=True)  
    print("=== Posterior (with evidence) ===")
    for name in sorted(after_marg.keys()):
        print(f"{name}: {after_marg[name]}")
        
    #builder.print_graph()
    builder.reset()
    return {"sao": sao, "answer": answer, "graph": builder.graph}

def start_demo_verbal_confidence(inference_data_path : str):
    data = open_jsonl(inference_data_path)
    model_id = "Qwen/Qwen2.5-14B-Instruct"
    curr_eos = [151645, 151643] # for Qwen2.5 series models
    curr_search_template = '\n\n{output_text}<information>{search_results}</information>\n\n'
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
    target_sequences = ["</search>", " </search>", "</search>\n", " </search>\n", "</search>\n\n", " </search>\n\n"]
    stopping_criteria = transformers.StoppingCriteriaList([StopOnSequence(target_sequences, tokenizer)])
    
    cpt_model_path = ""
    cpt_graph : GraphBuilder = GraphBuilder(cpt_model_path)
    
    for idx, d in tqdm(enumerate(data)):
        # try:
            question = d['question']
            golden_answers = d['golden_answers']
            output = run_demo_with_graph_verbal_confidence(
                question, golden_answers,
                model, tokenizer, stopping_criteria, curr_eos, curr_search_template,
                builder=cpt_graph,
                prior_s0=np.array([0.5, 0.5], dtype=np.float32),
                obs_eps=1e-6,
                lbp_iters=5,
                do_lbp_each_step=True
            )
        # except Exception as e:
        #     print(f"Error processing entry {idx}: {e}")
        #     continue
        
def start_demo_inner_confidence(inference_data_path : str):
    data = open_jsonl(inference_data_path)
    model_id = "Qwen/Qwen2.5-14B-Instruct"
    curr_eos = [151645, 151643] # for Qwen2.5 series models
    curr_search_template = '\n\n{output_text}<information>{search_results}</information>\n\n'
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
    target_sequences = ["</search>", " </search>", "</search>\n", " </search>\n", "</search>\n\n", " </search>\n\n"]
    stopping_criteria = transformers.StoppingCriteriaList([StopOnSequence(target_sequences, tokenizer)])
    
    cpt_model_path = ""
    cpt_graph : GraphBuilder = GraphBuilder(cpt_model_path)
    
    for idx, d in tqdm(enumerate(data)):
        # try:
            question = d['question']
            golden_answers = d['golden_answers']
            output = run_demo_with_graph_inner_confidence(
                question, golden_answers,
                model, tokenizer, stopping_criteria, curr_eos, curr_search_template,
                builder=cpt_graph,
                obs_eps=1e-6,
                lbp_iters=5,
                do_lbp_each_step=True
            )
        # except Exception as e:
        #     print(f"Error processing entry {idx}: {e}")
        #     continue
           
if __name__ == "__main__":
    inference_data_path = "run_data.jsonl"
    try:
        start_demo_inner_confidence(inference_data_path)
        #start_demo_verbal_confidence(inference_data_path)
    except KeyboardInterrupt as e:
        print("Process interrupted by user. Exiting...")
        sys.exit(0)