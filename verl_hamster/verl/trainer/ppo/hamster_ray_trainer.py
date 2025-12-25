# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pprint import pprint
from typing import Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger

#-------------- Adopt from verl-main/verl/trainer/ppo/ray_trainer.py --------------#
from verl.trainer.ppo.ray_trainer import RayPPOTrainer

from contextlib import contextmanager
from codetiming import Timer
from typing import Type, Dict
import torch
from verl.utils.torch_functional import masked_mean

import sys
from Hamster_Generation_Manager.Hamster_Generation_Manager import Hamster_Generation_Config, Hamster_LLM_Generation_Manager
from verl_hamster.verl.trainer.ppo.ray_trainer import compute_advantage
# ----------------- Modify Ended ----------------------------------------------------#

def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty='kl'):
    responses = data.batch['responses']
    response_length = responses.size(1)
    token_level_scores = data.batch['token_level_scores']
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch['info_mask'] if 'info_mask' in data.batch else data.batch['attention_mask']
    response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    if 'ref_log_prob' in data.batch.keys():
        kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'],
                                    kl_penalty=kl_penalty)  # (batch_size, response_length)
        kld = kld * response_mask
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch['token_level_rewards'] = token_level_rewards

    metrics = {'critic/kl': current_kl, 'critic/kl_coeff': beta}

    return data, metrics


# def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1):
#     # prepare response group
#     # TODO: add other ways to estimate advantages
#     if adv_estimator == 'gae':
#         values = data.batch['values']
#         responses = data.batch['responses']
#         response_length = responses.size(-1)
#         attention_mask = data.batch['attention_mask']
#         response_mask = attention_mask[:, -response_length:]
#         token_level_rewards = data.batch['token_level_rewards']
#         advantages, returns = core_algos.compute_gae_advantage_return(token_level_rewards=token_level_rewards,
#                                                                       values=values,
#                                                                       eos_mask=response_mask,
#                                                                       gamma=gamma,
#                                                                       lam=lam)
#         data.batch['advantages'] = advantages
#         data.batch['returns'] = returns
#     elif adv_estimator == 'grpo':
#         token_level_rewards = data.batch['token_level_rewards']
#         index = data.non_tensor_batch['uid']
#         responses = data.batch['responses']
#         response_length = responses.size(-1)
#         attention_mask = data.batch['attention_mask']
#         response_mask = attention_mask[:, -response_length:]
#         advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
#                                                                         eos_mask=response_mask,
#                                                                         index=index)
#         data.batch['advantages'] = advantages
#         data.batch['returns'] = returns
#     else:
#         raise NotImplementedError
#     return data

def reduce_metrics(metrics: dict):
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics

def _compute_response_info(batch):
    response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-response_length]
    response_mask = batch.batch['attention_mask'][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )

def compute_data_metrics(batch, use_critic=True):
    # TODO: add response length
    sequence_score = batch.batch['token_level_scores'].sum(-1)
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)

    advantages = batch.batch['advantages']
    returns = batch.batch['returns']

    max_response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-max_response_length].bool()
    response_mask = batch.batch['attention_mask'][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info['prompt_length']
    response_length = response_info['response_length']

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch['values']
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # score
        'critic/score/mean':
            torch.mean(sequence_score).detach().item(),
        'critic/score/max':
            torch.max(sequence_score).detach().item(),
        'critic/score/min':
            torch.min(sequence_score).detach().item(),
        # reward
        'critic/rewards/mean':
            torch.mean(sequence_reward).detach().item(),
        'critic/rewards/max':
            torch.max(sequence_reward).detach().item(),
        'critic/rewards/min':
            torch.min(sequence_reward).detach().item(),
        # adv
        'critic/advantages/mean':
            torch.mean(valid_adv).detach().item(),
        'critic/advantages/max':
            torch.max(valid_adv).detach().item(),
        'critic/advantages/min':
            torch.min(valid_adv).detach().item(),
        # returns
        'critic/returns/mean':
            torch.mean(valid_returns).detach().item(),
        'critic/returns/max':
            torch.max(valid_returns).detach().item(),
        'critic/returns/min':
            torch.min(valid_returns).detach().item(),
        **({
            # values
            'critic/values/mean': torch.mean(valid_values).detach().item(),
            'critic/values/max': torch.max(valid_values).detach().item(),
            'critic/values/min': torch.min(valid_values).detach().item(),
            # vf explained var
            'critic/vf_explained_var': (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
        } if use_critic else {}),

        # response length
        'response_length/mean':
            torch.mean(response_length).detach().item(),
        'response_length/max':
            torch.max(response_length).detach().item(),
        'response_length/min':
            torch.min(response_length).detach().item(),
        'response_length/clip_ratio':
            torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        # prompt length
        'prompt_length/mean':
            torch.mean(prompt_length).detach().item(),
        'prompt_length/max':
            torch.max(prompt_length).detach().item(),
        'prompt_length/min':
            torch.min(prompt_length).detach().item(),
        'prompt_length/clip_ratio':
            torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }

    # metrics for actions
    if 'turns_stats' in batch.meta_info:
        metrics['env/number_of_actions/mean'] = float(np.array(batch.meta_info['turns_stats'], dtype=np.int16).mean())
        metrics['env/number_of_actions/max'] = float(np.array(batch.meta_info['turns_stats'], dtype=np.int16).max())
        metrics['env/number_of_actions/min'] = float(np.array(batch.meta_info['turns_stats'], dtype=np.int16).min())
    if 'active_mask' in batch.meta_info:
        metrics['env/finish_ratio'] = 1 - float(np.array(batch.meta_info['active_mask'], dtype=np.int16).mean())
    if 'valid_action_stats' in batch.meta_info:
        metrics['env/number_of_valid_action'] = float(np.array(batch.meta_info['valid_action_stats'], dtype=np.int16).mean())
        metrics['env/ratio_of_valid_action'] = float((np.array(batch.meta_info['valid_action_stats'], dtype=np.int16) / np.array(batch.meta_info['turns_stats'], dtype=np.int16)).mean())
    if 'valid_search_stats' in batch.meta_info:
        metrics['env/number_of_valid_search'] = float(np.array(batch.meta_info['valid_search_stats'], dtype=np.int16).mean())

    return metrics

def compute_timing_metrics(batch, timing_raw):
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info['prompt_length']).item()
    num_response_tokens = torch.sum(response_info['response_length']).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        'gen': num_response_tokens,
        **{
            name: num_overall_tokens for name in ['ref', 'values', 'adv', 'update_critic', 'update_actor', 'rollout']
        },
    }

    return {
        **{
            f'timing_s/{name}': value for name, value in timing_raw.items()
        },
        **{
            f'timing_per_token_ms/{name}': timing_raw[name] * 1000 / num_tokens_of_section[name] for name in set(num_tokens_of_section.keys(
            )) & set(timing_raw.keys())
        },
    }

@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last

class Hamster_RayPPOTrainer(RayPPOTrainer):
    
    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cfg = omega_conf_to_dataclass(self.config.critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role="ref",
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls
            
        # create graph worker for CPT-based reasoning
        cpt_path = OmegaConf.select(self.config, "hamster.cpt_model_path", default=None)
        if cpt_path is not None:
            graph_remote_cls = self.role_worker_mapping.get(Role.GraphWorker, None)
            assert graph_remote_cls is not None, "CPT_Graph worker class not registered in role_worker_mapping"

            graph_kwargs = {
                "cpt_model_path": cpt_path,
                "do_calibration": bool(OmegaConf.select(self.config, "hamster.cpt_do_calibration", default=True)),
                "sharpen_strength": float(OmegaConf.select(self.config, "hamster.graph_sharpen_strength", default=3.0)),
                "min_assoc_strength": float(OmegaConf.select(self.config, "hamster.graph_min_assoc_strength", default=0.15)),
                "graph_obs_eps": float(OmegaConf.select(self.config, "hamster.graph_obs_eps", default=1e-6)),
            }

            resource_pool = self.resource_pool_manager.get_resource_pool(Role.GraphWorker)
            graph_cls = RayClassWithInitArgs(
                cls=graph_remote_cls,
                config=OmegaConf.create(graph_kwargs),
                role="cpt_graph",
            )
            self.resource_pool_to_cls[resource_pool]["cpt_graph"] = graph_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        self.rm_wg = None
        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()
            
        self.graph_wg = None
        if cpt_path is not None and "cpt_graph" in all_wg:
            self.graph_wg = all_wg["cpt_graph"]
            self.graph_wg.init_model()   # -> call GraphServiceWorker.setup(**graph_kwargs)

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.experimental.agent_loop import AgentLoopManager

            self.async_rollout_mode = True
            self.async_rollout_manager = AgentLoopManager(
                config=self.config, worker_group=self.actor_rollout_wg, rm_wg=self.rm_wg
            )
    
    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0
        
        # Follow search-r1
        # Agent config preparation
        from verl_hamster.Hamster_Generation_Manager import Hamster_Graph_Config
        gen_config = Hamster_Generation_Config(
            num_gpus=self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes,
            max_prompt_length=self.config.data.max_prompt_length,
            max_response_length=self.config.data.max_response_length,
            
            
            max_turns=self.config.hamster.max_turns,
            max_start_length=self.config.hamster.max_start_length,
            max_obs_length=self.config.hamster.max_obs_length,
            no_think_rl=self.config.hamster.no_think_rl,
            search_url = self.config.hamster.url,
            topk = self.config.hamster.topk,
            is_verbal= self.config.hamster.is_verbal,
            graph_config = Hamster_Graph_Config(
                cpt_model_path = self.config.hamster.cpt_model_path,
                cpt_rpc_url = self.config.hamster.cpt_rpc_url,
                cpt_do_calibration = self.config.hamster.cpt_do_calibration,
                graph_sharpen_strength = self.config.hamster.graph_sharpen_strength,
                graph_min_assoc_strength = self.config.hamster.graph_min_assoc_strength,
                graph_obs_eps = self.config.hamster.graph_obs_eps,
                graph_lbp_iters = self.config.hamster.graph_lbp_iters,
                graph_evidence_strength = self.config.hamster.graph_evidence_strength,
                graph_do_forward_lbp = self.config.hamster.graph_do_forward_lbp
            ),
            
            
            online_search=self.config.hamster.online_search,
            serper_api_key=self.config.hamster.serper_api_key,
            gemini_api_key=self.config.hamster.gemini_api_key,
            jina_api_key=self.config.hamster.jina_api_key,
            llm_api_key=self.config.hamster.llm_api_key,
            llm_model=self.config.hamster.llm_model,
            llm_base_url=self.config.hamster.llm_base_url
        )

        generation_manager = Hamster_LLM_Generation_Manager(
            tokenizer=self.tokenizer,
            actor_rollout_wg=self.actor_rollout_wg,
            config=gen_config,
        )

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                
                print(f'epoch {epoch}, step {self.global_steps}')
                
                metrics = {}
                timing_raw = {}

                do_profile = (
                    self.global_steps in self.config.trainer.profile_steps
                    if self.config.trainer.profile_steps is not None
                    else False
                )
                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(do_profile)

                #print(f"batch_dict is : {batch_dict}")
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                #print(f"batch keys is : {batch.batch.keys()}, non_tensor_batch keys is : {batch.non_tensor_batch.keys()}, meta_info keys is : {batch.meta_info.keys()}")

                # pop those keys for generation
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "golden_answers"]
                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                if "interaction_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("interaction_kwargs")
                if "index" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("index")
                if "agent_name" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("agent_name")

                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        
                        # ------------------ Modify Here ------------------
                        # if not self.async_rollout_mode:
                        #     gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        # else:
                        #     gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                        
                        first_input_ids = gen_batch.batch["input_ids"][:, -gen_config.max_start_length:].clone().long()
                        final_gen_batch_output = generation_manager.run_llm_loop(
                                gen_batch=gen_batch,
                                initial_input_ids=first_input_ids,
                            )
                        
                        for key in final_gen_batch_output.batch.keys():
                            final_gen_batch_output.batch[key] = final_gen_batch_output.batch[key].long()

                        # with torch.no_grad():
                        #     output = self.actor_rollout_wg.compute_log_prob(final_gen_batch_output)
                        #     final_gen_batch_output = final_gen_batch_output.union(output)

                        # # batch.non_tensor_batch['uid'] = batch.non_tensor_batch['index'].copy()
                        # n_samples = len(batch.batch)  # 当前 batch 大小
                        # if 'index' in gen_batch.non_tensor_batch:
                        #     uid_base = gen_batch.non_tensor_batch['index']
                        # else:
                        #     uid_base = np.array([str(uuid.uuid4()) for _ in range(n_samples)], dtype=object)

                        # batch.non_tensor_batch['uid'] = np.array(uid_base, dtype=object)
                                      
                        #  # 这里是在 gen_batch 已经 repeat 之后、batch 还没 repeat 之前
                        n_samples = len(batch.batch)  # 基批大小，例如 16
                        rollout_n = int(self.config.actor_rollout_ref.rollout.n)

                        if 'index' in gen_batch.non_tensor_batch:
                            idx_rep = np.array(gen_batch.non_tensor_batch['index'])
                            # 期望 idx_rep 长度 = n_samples * rollout_n
                            if len(idx_rep) != n_samples * rollout_n:
                                # 兜底：直接生成新的 uid，避免断言
                                uid_base = np.array([str(uuid.uuid4()) for _ in range(n_samples)], dtype=object)
                            else:
                                # 还原到基批维度：按 [n_samples, rollout_n] reshape，取每组第一个
                                uid_base = idx_rep.reshape(n_samples, rollout_n)[:, 0]
                        else:
                            uid_base = np.array([str(uuid.uuid4()) for _ in range(n_samples)], dtype=object)
                        batch.non_tensor_batch['uid'] = np.array(uid_base, dtype=object)             
                                          
                        # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(final_gen_batch_output)
                        
                        # timing_raw.update(gen_batch_output.meta_info["timing"])
                        # gen_batch_output.meta_info.pop("timing", None)

                    # if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                    #     with marked_timer("gen_max", timing_raw, color="purple"):
                    #         gen_baseline_batch = deepcopy(gen_batch)
                    #         gen_baseline_batch.meta_info["do_sample"] = False
                    #         if not self.async_rollout_mode:
                    #             gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                    #         else:
                    #             gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                    #         batch = batch.union(gen_baseline_output)
                    #         reward_baseline_tensor = self.reward_fn(batch)
                    #         reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                    #         batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                    #         batch.batch["reward_baselines"] = reward_baseline_tensor

                    #         del gen_baseline_batch, gen_baseline_output

                    # batch.non_tensor_batch["uid"] = np.array(
                    #     [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                    # )
                    # # repeat to align with repeated responses in rollout
                    # batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    # batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        from verl.trainer.ppo.ray_trainer import compute_response_mask
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # convert all tensors to long to calculate ece
                    for key in batch.batch.keys():
                        if key != 'old_log_probs':
                            batch.batch[key] = batch.batch[key].long()

                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        # this path will not be traced into the worker processes
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(data=batch, reward_fn=self.reward_fn)
                        else:
                            # run this path
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)
                            #print(f"reward as list: {reward_tensor.cpu().tolist()}")
                            metrics.update(reward_extra_infos_dict)

                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            rollout_old_log_probs = batch.batch["rollout_log_probs"]
                            actor_old_log_probs = batch.batch["old_log_probs"]
                            attention_mask = batch.batch["attention_mask"]
                            responses = batch.batch["responses"]
                            response_length = responses.size(1)
                            response_mask = attention_mask[:, -response_length:]

                            rollout_probs = torch.exp(rollout_old_log_probs)
                            actor_probs = torch.exp(actor_old_log_probs)
                            rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                            rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())
                            rollout_probs_diff_max = torch.max(rollout_probs_diff)
                            rollout_probs_diff_mean = torch.mean(rollout_probs_diff)
                            rollout_probs_diff_std = torch.std(rollout_probs_diff)
                            metrics.update(
                                {
                                    "training/rollout_probs_diff_max": rollout_probs_diff_max.detach().item(),
                                    "training/rollout_probs_diff_mean": rollout_probs_diff_mean.detach().item(),
                                    "training/rollout_probs_diff_std": rollout_probs_diff_std.detach().item(),
                                }
                            )

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)
                    
                    # ------------------ Modify Here ------------------
                    # Here we have training-model's old_log_probs and reference model's ref_log_probs
                    # We can compute Trajectory ece between these two models
                    # TODO
                    need_save_meta_info = defaultdict()
                    if self.config.hamster.compute_ece:
                        from verl.utils.hamster_utils.ece import compute
                        batch_clone = deepcopy(batch)
                        modify_batch = compute(batch_clone)
                        modify_batch_meta_info = modify_batch.meta_info
                        need_save_meta_info.update(modify_batch_meta_info)
                        #metrics.update({"training/trajectory_ece": ece})
                        #metrics.update({"training/trajectory_ece_old": ece_old, "training/trajectory_ece_training": ece_training})
                        metrics.update({"mean_ece_forward" : modify_batch_meta_info["mean_ece_forward"]})
                        metrics.update({"rms_ece_forward" : modify_batch_meta_info["rms_ece_forward"]})
                        metrics.update({"gmean_ece_forward" : modify_batch_meta_info["gmean_ece_forward"]})
                        metrics.update({"max_ece_forward" : modify_batch_meta_info["max_ece_forward"]})
                        metrics.update({"kl_weighted_ece_forward" : modify_batch_meta_info["kl_weighted_ece_forward"]})

                        metrics.update({"mean_ece_backward" : modify_batch_meta_info["mean_ece_backward"]})
                        metrics.update({"rms_ece_backward" : modify_batch_meta_info["rms_ece_backward"]})
                        metrics.update({"gmean_ece_backward" : modify_batch_meta_info["gmean_ece_backward"]})
                        metrics.update({"max_ece_backward" : modify_batch_meta_info["max_ece_backward"]})
                        metrics.update({"kl_weighted_ece_backward" : modify_batch_meta_info["kl_weighted_ece_backward"]})

                        metrics.update({"mean_auc_forward" : modify_batch_meta_info["mean_auc_forward"]})
                        metrics.update({"rms_auc_forward" : modify_batch_meta_info["rms_auc_forward"]})
                        metrics.update({"gmean_auc_forward" : modify_batch_meta_info["gmean_auc_forward"]})
                        metrics.update({"max_auc_forward" : modify_batch_meta_info["max_auc_forward"]})
                        metrics.update({"kl_weighted_auc_forward" : modify_batch_meta_info["kl_weighted_auc_forward"]})

                        metrics.update({"mean_auc_backward" : modify_batch_meta_info["mean_auc_backward"]})
                        metrics.update({"rms_auc_backward" : modify_batch_meta_info["rms_auc_backward"]})
                        metrics.update({"gmean_auc_backward" : modify_batch_meta_info["gmean_auc_backward"]})
                        metrics.update({"max_auc_backward" : modify_batch_meta_info["max_auc_backward"]})
                        metrics.update({"kl_weighted_auc_backward" : modify_batch_meta_info["kl_weighted_auc_backward"]})
                        metrics.update({"acc": modify_batch_meta_info["acc"]})
                        
                        metrics.update({"mean_ece_verbal" : modify_batch_meta_info["mean_ece_verbal"]})
                        metrics.update({"gmean_ece_verbal" : modify_batch_meta_info["gmean_ece_verbal"]})
                        metrics.update({"rms_ece_verbal" : modify_batch_meta_info["rms_ece_verbal"]})
                        metrics.update({"max_ece_verbal" : modify_batch_meta_info["max_ece_verbal"]})
                        
                        metrics.update({"mean_auc_verbal" : modify_batch_meta_info["mean_auc_verbal"]})
                        metrics.update({"gmean_auc_verbal" : modify_batch_meta_info["gmean_auc_verbal"]})
                        metrics.update({"rms_auc_verbal" : modify_batch_meta_info["rms_auc_verbal"]})
                        metrics.update({"max_auc_verbal" : modify_batch_meta_info["max_auc_verbal"]})

                    # ------------------ Modify Here ------------------

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process

                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            
                            # ------ Modify Here -----
                            if self.config.hamster.state_masking:
                                batch, metrics = self._create_loss_mask(batch, metrics)
                            # ------ Modify Here -----
                            
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # # Log rollout generations if enabled
                    # rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    # if rollout_data_dir:
                    #     with marked_timer("dump_rollout_generations", timing_raw, color="green"):
                    #         inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                    #         outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                    #         scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                    #         sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]

                    #         reward_extra_infos_to_dump = reward_extra_infos_dict.copy()
                    #         if "request_id" in batch.non_tensor_batch:
                    #             reward_extra_infos_dict.setdefault(
                    #                 "request_id",
                    #                 batch.non_tensor_batch["request_id"].tolist(),
                    #             )

                    #         self._hamster_dump_generations(
                    #             inputs=inputs,
                    #             outputs=outputs,
                    #             gts=sample_gts,
                    #             scores=scores,
                    #             reward_extra_infos_dict=reward_extra_infos_dict,
                    #             additional_info_dict=need_save_meta_info,
                    #             dump_path=rollout_data_dir,
                    #         )
                    # # Log rollout generations if enabled
                    # rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    # if rollout_data_dir:
                    #     with marked_timer("dump_rollout_generations", timing_raw, color="green"):
                    #         inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                    #         outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                    #         scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                    #         sample_gts = [
                    #             item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None)
                    #             for item in batch
                    #         ]

                    #         # 以 reward_extra_infos_dict 为基础构造一个 dump 用的 dict
                    #         reward_extra_infos_to_dump = (reward_extra_infos_dict or {}).copy()

                    #         # 按样本对齐地附加 request_id（如果有）
                    #         if "request_id" in batch.non_tensor_batch:
                    #             reward_extra_infos_to_dump.setdefault(
                    #                 "request_id",
                    #                 batch.non_tensor_batch["request_id"].tolist(),
                    #             )

                    #         # 按样本对齐地附加 uid
                    #         if "uid" in batch.non_tensor_batch:
                    #             reward_extra_infos_to_dump.setdefault(
                    #                 "uid",
                    #                 batch.non_tensor_batch["uid"].tolist(),
                    #             )

                    #         # 注意这里传的是 reward_extra_infos_to_dump，不是原始的 reward_extra_infos_dict
                    #         self._hamster_dump_generations(
                    #             inputs=inputs,
                    #             outputs=outputs,
                    #             gts=sample_gts,
                    #             scores=scores,
                    #             reward_extra_infos_dict=reward_extra_infos_to_dump,
                    #             additional_info_dict=need_save_meta_info,   # 这里面是 ECE/AUC/acc 等
                    #             dump_path=rollout_data_dir,
                    #         )
                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            sample_gts = [
                                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None)
                                for item in batch
                            ]

                            # 以 reward_extra_infos_dict 为基础构造一个 dump 用的 dict
                            reward_extra_infos_to_dump = (reward_extra_infos_dict or {}).copy()

                            # 先取出 uid 列表（rollout 级别，一般长度 = batch_size_after_rollout）
                            uids = None
                            if "rollout_uid" in batch.non_tensor_batch:
                                # 可能是 np.ndarray，需要转成 list[str]
                                raw_uids = batch.non_tensor_batch["rollout_uid"]
                                try:
                                    uids = list(raw_uids.tolist() if hasattr(raw_uids, "tolist") else list(raw_uids))
                                except TypeError:
                                    uids = [raw_uids]
                                uids = [str(u) for u in uids]

                                # 把 uid 作为一列写进 jsonl
                                reward_extra_infos_to_dump.setdefault("rollout_uid", uids)

                            # 按样本对齐地附加 request_id（如果有）
                            if "request_id" in batch.non_tensor_batch:
                                reward_extra_infos_to_dump.setdefault(
                                    "request_id",
                                    batch.non_tensor_batch["request_id"].tolist(),
                                )

                            # ===== 按 uid 显式对齐 trajectories / graph_calibrations =====
                            # 1. 取出 trajectory / graph_calibrations 的原始列表
                            traj_list = None
                            calib_list = None

                            if "trajectories" in batch.non_tensor_batch:
                                t_raw = batch.non_tensor_batch["trajectories"]
                                traj_list = t_raw.tolist() if hasattr(t_raw, "tolist") else list(t_raw)

                            if "graph_calibrations" in batch.non_tensor_batch:
                                c_raw = batch.non_tensor_batch["graph_calibrations"]
                                calib_list = c_raw.tolist() if hasattr(c_raw, "tolist") else list(c_raw)

                            # 2. 如果有 uid，就通过 uid 做一次 映射对齐（防止顺序万一有偏）
                            if uids is not None and traj_list is not None:
                                # 构造 uid -> trajectory 的映射
                                uid2traj = {}
                                for obj in traj_list:
                                    if isinstance(obj, dict):
                                        u = str(obj.get("rollout_uid", None))
                                        if u is not None:
                                            uid2traj[u] = obj
                                # 按当前 batch 的 uids 顺序重新排列
                                aligned_traj = [uid2traj.get(str(u), None) for u in uids]
                                reward_extra_infos_to_dump.setdefault("trajectory", aligned_traj)

                            if uids is not None and calib_list is not None:
                                uid2calib = {}
                                for obj in calib_list:
                                    if isinstance(obj, dict):
                                        u = str(obj.get("rollout_uid", None))
                                        if u is not None:
                                            uid2calib[u] = obj
                                aligned_calib = [uid2calib.get(str(u), None) for u in uids]
                                reward_extra_infos_to_dump.setdefault("graph_calibrations", aligned_calib)

                            # 注意这里传的是 reward_extra_infos_to_dump，不是原始的 reward_extra_infos_dict
                            self._hamster_dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                gts=sample_gts,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_to_dump,
                                additional_info_dict=need_save_meta_info,   # 这里面是 ECE/AUC/acc 等
                                dump_path=rollout_data_dir,
                            )


                    # validate
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                    ):
                        with marked_timer("testing", timing_raw, color="green"):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                    esi_close_to_expiration = should_save_ckpt_esi(
                        max_steps_duration=self.max_steps_duration,
                        redundant_time=self.config.trainer.esi_redundant_time,
                    )
                    # Check if the conditions for saving a checkpoint are met.
                    # The conditions include a mandatory condition (1) and
                    # one of the following optional conditions (2/3/4):
                    # 1. The save frequency is set to a positive value.
                    # 2. It's the last training step.
                    # 3. The current step number is a multiple of the save frequency.
                    # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                    if self.config.trainer.save_freq > 0 and (
                        is_last_step
                        or self.global_steps % self.config.trainer.save_freq == 0
                        or esi_close_to_expiration
                    ):
                        if esi_close_to_expiration:
                            print("Force saving checkpoint: ESI instance expiration approaching.")
                        with marked_timer("save_checkpoint", timing_raw, color="green"):
                            self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    self._stop_profiling(do_profile)

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)

    # def _hamster_dump_generations(
    #     self, inputs, outputs, gts, scores, reward_extra_infos_dict, additional_info_dict, dump_path
    # ):
    #     # self._dump_generations(
    #     #     inputs=inputs,
    #     #     outputs=outputs,
    #     #     gts=gts,
    #     #     scores=scores,
    #     #     reward_extra_infos_dict=reward_extra_infos_dict,
    #     #     dump_path=dump_path,
    #     # )
    #     """Dump rollout/validation samples as JSONL."""
    #     os.makedirs(dump_path, exist_ok=True)
    #     filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

    #     n = len(inputs)
    #     base_data = {
    #         "input": inputs,
    #         "output": outputs,
    #         "gts": gts,
    #         "score": scores,
    #         "step": [self.global_steps] * n,
    #     }

    #     for k, v in reward_extra_infos_dict.items():
    #         if len(v) == n:
    #             base_data[k] = v
        
    #     for k, v in additional_info_dict.items():
    #         # if len(v) == n:
    #         base_data[k] = v

    #     lines = []
    #     for i in range(n):
    #         entry = {k: v[i] for k, v in base_data.items()}
    #         lines.append(json.dumps(entry, ensure_ascii=False))

    #     with open(filename, "w") as f:
    #         f.write("\n".join(lines) + "\n")

    #     print(f"Dumped generations to {filename}")
    # def _hamster_dump_generations(self, inputs, outputs, gts, scores,
    #                           reward_extra_infos_dict, additional_info_dict, dump_path):
    #     os.makedirs(dump_path, exist_ok=True)
    #     filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

    #     n = len(inputs)

    #     # 基础字段（都是按样本对齐的）
    #     base_data = {
    #         "input": inputs,
    #         "output": outputs,
    #         "gts": gts,
    #         "score": scores,
    #         "step": [self.global_steps] * n,
    #     }

    #     # ---------- 工具函数：转可序列化 ----------
    #     def to_serializable(x):
    #         if isinstance(x, np.generic):
    #             return x.item()
    #         if isinstance(x, np.ndarray):
    #             return x.tolist()
    #         if 'torch' in str(type(x)) and isinstance(x, torch.Tensor):
    #             return x.detach().cpu().tolist()
    #         return x

    #     # ---------- 工具函数：规范化为长度 n 的列表 ----------
    #     def normalize_field(v):
    #         v = to_serializable(v)

    #         # 列表/元组
    #         if isinstance(v, (list, tuple)):
    #             return list(v) if len(v) == n else [v] * n

    #         # 字典：没有 uid 映射就广播整份 dict
    #         if isinstance(v, dict):
    #             return [v] * n

    #         # 其它（标量/字符串/None...）
    #         return [v] * n

    #     # 合并额外信息：先规范化
    #     for k, v in (reward_extra_infos_dict or {}).items():
    #         base_data[k] = normalize_field(v)

    #     for k, v in (additional_info_dict or {}).items():
    #         base_data[k] = normalize_field(v)

    #     # 把基础字段也统一规范化，防止上游改动
    #     for k, v in list(base_data.items()):
    #         base_data[k] = normalize_field(v) if not (isinstance(v, list) and len(v) == n) else v

    #     # 最后，再构建逐行 entry
    #     lines = []
    #     for i in range(n):
    #         entry = {k: base_data[k][i] for k in base_data}
    #         lines.append(json.dumps(entry, ensure_ascii=False))

    #     with open(filename, "w") as f:
    #         f.write("\n".join(lines) + "\n")

    #     print(f"Dumped generations to {filename}")
    
    def _hamster_dump_generations(
        self,
        inputs,
        outputs,
        gts,
        scores,
        reward_extra_infos_dict,
        additional_info_dict,
        dump_path,
    ):
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)

        # ------------------ 基础字段：天然逐样本对齐 ------------------ #
        base_data = {
            "input": inputs,                      # List[str]，len = n
            "output": outputs,                    # List[str]，len = n
            "gts": gts,                           # List[...]/None，len = n
            "score": scores,                      # List[float]，len = n
            "step": [self.global_steps] * n,      # 广播当前 step
        }

        def to_serializable(x):
            if isinstance(x, np.generic):
                return x.item()
            if isinstance(x, np.ndarray):
                return x.tolist()
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().tolist()
            return x

        def normalize_field(v, key_name: str):
            """
            规则：
            - 如果 v 是 list/tuple 且 len == n：按样本对齐使用（比如 uid, mean_forward 等）。
            - 如果 v 是 list/tuple 且 len != n：当作 batch 级摘要，整份广播。
            - 其它类型（标量 / dict / None 等）：同样当 batch 级摘要，整份广播。
            """
            if isinstance(v, (list, tuple)):
                if len(v) == n:
                    return [to_serializable(x) for x in v]
                v_ser = to_serializable(v)
                return [v_ser] * n

            v_ser = to_serializable(v)
            return [v_ser] * n

        # 合并 reward_extra_infos（包含 uid, request_id, 各类 reward 统计等）
        if reward_extra_infos_dict:
            for k, v in reward_extra_infos_dict.items():
                base_data[k] = normalize_field(v, k)

        # 合并 additional_info_dict（通常是 compute_ece 写进来的 mean_ece_* / mean_auc_* / acc 等）
        if additional_info_dict:
            for k, v in additional_info_dict.items():
                base_data[k] = normalize_field(v, k)

        # 再统一保证所有字段都是长度 n 的 list
        for k, v in list(base_data.items()):
            if isinstance(v, list) and len(v) == n:
                continue
            base_data[k] = [to_serializable(v)] * n

        lines = []
        for i in range(n):
            entry = {k: base_data[k][i] for k in base_data}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _create_loss_mask(self, batch, metrics):
        """Create loss mask for state tokens."""
        response_length = batch.batch['responses'].shape[-1]
        response_mask = batch.batch['attention_mask'][:, -response_length:]
        
        loss_mask = batch.batch['info_mask'][:, -response_length:]
        batch.batch['loss_mask'] = loss_mask

        metrics.update({
            'state_tokens/total': loss_mask.sum().item(),
            'state_tokens/coverage': (loss_mask.sum() / response_mask.sum()).item(),
        })
        
        return batch, metrics