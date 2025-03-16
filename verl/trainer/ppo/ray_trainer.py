# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import logging
import os
import time
import re
import traceback
import uuid
from typing import Dict, List, Tuple, Type
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from contextlib import contextmanager

import numpy as np
import torch
import ray
from omegaconf import OmegaConf, open_dict, read_write
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray.base import RayWorkerGroup, RayResourcePool, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.workers.fsdp_workers import CriticWorker, ActorRolloutRefWorker, RewardModelWorker
from verl.utils.torch_functional import masked_mean
from ragen.llm_agent.generation import LLMGenerationManager, GenerationConfig

try:
    from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
except ImportError:
    from verl.utils.dataset import RLHFDataset, collate_fn

WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """
    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_collocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_collocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_collocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes,
                                            use_gpu=True,
                                            max_collocate_count=1,
                                            name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]


def apply_kl_penalty(data, kl_ctrl, kl_penalty='kl'):
    responses = data.batch['responses']
    response_length = responses.size(1)
    token_level_scores = data.batch['token_level_scores']
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch['attention_mask']
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

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # averagen over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch['token_level_rewards'] = token_level_rewards

    metrics = {'critic/kl': current_kl, 'critic/kl_coeff': beta}

    return data, metrics


def compute_advantage(data, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1):
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == 'gae':
        values = data.batch['values']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        token_level_rewards = data.batch['token_level_rewards']
        advantages, returns = core_algos.compute_gae_advantage_return(token_level_rewards=token_level_rewards,
                                                                      values=values,
                                                                      eos_mask=response_mask,
                                                                      gamma=gamma,
                                                                      lam=lam)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator in ['grpo', 'brpo', 'arpo']:
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        raise NotImplementedError
    return data


def reduce_metrics(metrics: dict):
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics


def _compute_response_info(batch):
    """Compute response statistics."""
    metrics = {}
    # If batch is not a list, make it a list for consistent processing
    if not isinstance(batch, list):
        batch = [batch]
        
    token_counts = []
    for item in batch:
        if isinstance(item, dict) and 'response' in item:
            # Count non-padding tokens in the response
            response = item['response']
            if isinstance(response, torch.Tensor):
                # Count non-zero tokens (assuming padding is 0)
                token_count = torch.sum(response != 0).item()
            elif isinstance(response, list):
                token_count = len(response)
            else:
                token_count = 0
            token_counts.append(token_count)
    
    if token_counts:
        metrics['response_token_count/mean'] = sum(token_counts) / len(token_counts)
        metrics['response_token_count/min'] = min(token_counts)
        metrics['response_token_count/max'] = max(token_counts)
    
    return metrics


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
            
        # metrics for actions
        'metric/total_env':
            int(np.array(batch.non_tensor_batch['total_env'], dtype=np.int16).sum()),
        'metric/finished_env':
            int(np.array(batch.non_tensor_batch['finished_env'], dtype=np.int16).sum()),
        'metric/success_env':
            int(np.array(batch.non_tensor_batch['success_env'], dtype=np.int16).sum()),
        'metric/traj_length':
            float(np.array(batch.non_tensor_batch['traj_length'], dtype=np.int16).mean()),
        'metric/valid_action':
            float(np.array(batch.non_tensor_batch['valid_action'], dtype=np.int16).mean()),
        'metric/effective_action':
            float(np.array(batch.non_tensor_batch['effective_action'], dtype=np.int16).mean()),
        'metric/effective_action_ratio':
            float(np.array(batch.non_tensor_batch['effective_action_ratio'], dtype=np.float32).mean()),
    }

    # metric for two-armed bandit
    if batch.non_tensor_batch['data_source'][0] == 'two_armed_bandit':
        batch_action = np.array(batch.non_tensor_batch['bandit_metrics'], dtype=np.int16)
        metrics['metric/n_low_arm'] = int(np.sum(batch_action == 1))
        metrics['metric/n_high_arm'] = int(np.sum(batch_action == 2))
        metrics['metric/n_invalid'] = int(np.sum(batch_action == 0))

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
    start_time = time.time()
    try:
        yield
    finally:
        elapsed_time = time.time() - start_time
        timing_raw[name] = elapsed_time


class RayPPOTrainer(object):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 reward_fn=None,
                 val_reward_fn=None,
                 env=None,
                 val_env=None,
                 env_class=None):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn
        self.env = env
        self.val_env = env if val_env is None else val_env
        self.env_class = env_class
        
        if val_env is not None:
            print("[INFO] val env is different from train env, it means you are evaluating the model's generalization capabilities.")

        self.hybrid_engine = config.get('actor_rollout_ref', {}).get('hybrid_engine', False)
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.use_critic = Role.Critic in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        
        # Initialize rollout_config from actor_rollout_ref config
        self.rollout_config = {}
        if hasattr(self.config, 'actor_rollout_ref') and hasattr(self.config.actor_rollout_ref, 'rollout'):
            self.rollout_config = OmegaConf.to_container(self.config.actor_rollout_ref.rollout, resolve=True)
            print(f"Initialized rollout_config: {self.rollout_config}")
        else:
            print("Warning: No rollout configuration found in actor_rollout_ref")
            
        # Check CUDA availability and handle systems without GPUs
        if not torch.cuda.is_available():
            print("WARNING: CUDA is not available. Running in CPU mode.")
            # Adjust configurations that might depend on GPU availability
            if 'device' in self.rollout_config:
                self.rollout_config['device'] = 'cpu'
            # Set GPU memory utilization to 0 if it exists
            if 'gpu_memory_utilization' in self.rollout_config:
                self.rollout_config['gpu_memory_utilization'] = 0
        
        # Create a ray context if not already initialized
        if not ray.is_initialized():
            ray.init()
        
        self.val_num = 0

        # define KL control
        if self.use_reference_policy:
            if config.algorithm.get('kl_ctrl', {}).get('type') == 'fixed':
                self.kl_ctrl = core_algos.FixedKLController(kl_coef=config.algorithm.kl_ctrl.kl_coef)
            elif config.algorithm.get('kl_ctrl', {}).get('type') == 'adaptive':
                assert config.algorithm.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
                self.kl_ctrl = core_algos.AdaptiveKLController(init_kl_coef=config.algorithm.kl_ctrl.kl_coef,
                                                               target_kl=config.algorithm.kl_ctrl.target_kl,
                                                               horizon=config.algorithm.kl_ctrl.horizon)
            else:
                raise NotImplementedError
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.)

        self._create_dataloader()
        self._init_logger()
    
    def _init_logger(self):
        from verl.utils.tracking import Tracking
        self.logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

    def _create_dataloader(self):
        """Create train dataloader and validation dataloader."""
        
        # Initialize dataloaders for training and validation
        train_config = OmegaConf.to_container(self.config.data, resolve=True)
        
        # Convert to proper Python types
        for k, v in train_config.items():
            if isinstance(v, str) and v.lower() == 'none':
                train_config[k] = None
        
        val_config = train_config.copy()
        
        # Update validation config with specific values
        val_config.update({
            'batch_size': self.config.data.val_batch_size,
            'data_files': self.config.data.val_files,
            'data_num': self.config.data.val_data_num,
        })
        
        train_config.update({
            'batch_size': self.config.data.train_batch_size,
            'data_files': self.config.data.train_files, 
            'data_num': self.config.data.train_data_num,
            'shuffle': self.config.data.shuffle_train_dataloader
        })
        
        # Use PyArrow to read parquet files
        import pyarrow.parquet as pq
        
        original_train_dataset = pq.read_table(train_config['data_files']).to_pandas()
        original_val_dataset = pq.read_table(val_config['data_files']).to_pandas()
        
        print(f"original dataset len: {len(original_train_dataset)}")
        
        # Filter dataset if needed
        if train_config.get('data_num') is not None:
            train_dataset = original_train_dataset.head(train_config['data_num'])
        else:
            train_dataset = original_train_dataset
            
        print(f"filter dataset len: {len(train_dataset)}")
        print(f"filtered training dataset size: {len(train_dataset)}")
        
        # Filter validation dataset
        print(f"original dataset len: {len(original_val_dataset)}")
        if val_config.get('data_num') is not None:
            val_dataset = original_val_dataset.head(val_config['data_num'])
        else:
            val_dataset = original_val_dataset
            
        print(f"filter dataset len: {len(val_dataset)}")
        print(f"filtered validation dataset size: {len(val_dataset)}")
        
        # Create dataloaders
        self.train_dataloader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=train_config['batch_size'],
            shuffle=train_config.get('shuffle', True)
        )
        
        self.val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=val_config['batch_size'],
            shuffle=False
        )
        
        print(f"Size of train dataloader: {len(self.train_dataloader)}")
        print(f"Size of val dataloader: {len(self.val_dataloader)}")
        
        # Calculate total training steps
        self.total_training_steps = self.config.trainer.total_training_steps
        if self.total_training_steps is None:
            self.total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
        
        print(f"Total training steps: {self.total_training_steps}")
        
        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.get('total_epochs', 1)

        if self.config.trainer.get('total_training_steps') is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def init_workers(self):
        """Initialize all Ray workers needed for PPO training."""
        # Init resource pool
        print("Initialize workers...")
        
        # Add missing default roles to config if needed
        if not hasattr(self.config.actor_rollout_ref, 'roles'):
            print("Warning: roles not defined in actor_rollout_ref, setting default roles [actor, rollout]")
            with open_dict(self.config.actor_rollout_ref):
                self.config.actor_rollout_ref.roles = [Role.Actor.name.lower(), Role.Rollout.name.lower()]
        
        # Create a proper RayResourcePool instance
        resource_pool = RayResourcePool(
            process_on_nodes=[1],  # One process per node
            use_gpu=True,
            name_prefix="worker_pool",
            detached=False
        )
        
        # Initialize resource_pool_to_cls dictionary
        self.resource_pool_to_cls = {}
        self.resource_pool_to_cls[resource_pool] = {}
        
        # Register worker classes with proper ray.remote wrapping
        print("Registering ActorRolloutRefWorker")
        ActorRolloutRefWorkerRemote = ray.remote(ActorRolloutRefWorker)
        self.resource_pool_to_cls[resource_pool][Role.ActorRollout.name] = RayClassWithInitArgs(
            ActorRolloutRefWorkerRemote,
            kwargs=dict(
                config=self.config.actor_rollout_ref,
                tokenizer=self.tokenizer,
                rollout_config=self.rollout_config
            )
        )
        
        if self.use_critic:
            print("Registering CriticWorker")
            CriticWorkerRemote = ray.remote(CriticWorker)
            self.resource_pool_to_cls[resource_pool][Role.Critic.name] = RayClassWithInitArgs(
                CriticWorkerRemote,
                kwargs=dict(
                    config=self.config.critic,
                    tokenizer=self.tokenizer
                )
            )
        
        # Add reward model worker if configured
        if 'rm' in self.config and hasattr(self.config, 'rm'):
            rm_cls = ray.remote(RewardModelWorker)
            self.resource_pool_to_cls[resource_pool][Role.RewardModel.name] = RayClassWithInitArgs(
                rm_cls,
                kwargs=dict(
                    config=self.config.rm,
                    tokenizer=self.tokenizer
                )
            )
        
        # Initialize the worker group with the resource pool and ray classes
        print("Initializing worker group")
        
        # We need to convert our dictionary of RayClassWithInitArgs to the format expected by RayWorkerGroup
        # The RayWorkerGroup class expects a dictionary mapping from role names to RayClassWithInitArgs objects
        worker_roles = {}
        
        # Map ActorRolloutRefWorker to ActorRollout role
        if 'actor_rollout_ref' in self.resource_pool_to_cls[resource_pool]:
            worker_roles[Role.ActorRollout.name] = self.resource_pool_to_cls[resource_pool]['actor_rollout_ref']
            
        # Map CriticWorker to Critic role
        if 'critic' in self.resource_pool_to_cls[resource_pool]:
            worker_roles[Role.Critic.name] = self.resource_pool_to_cls[resource_pool]['critic']
            
        # Map RewardModelWorker to RewardModel role
        if 'rm' in self.resource_pool_to_cls[resource_pool]:
            worker_roles[Role.RewardModel.name] = self.resource_pool_to_cls[resource_pool]['rm']
        
        print(f"Worker roles: {list(worker_roles.keys())}")
        
        # Initialize worker groups individually for each worker type
        print("Creating worker groups...")
        self.actor_rollout_wg = None
        self.critic_wg = None
        self.rm_wg = None
        self.wg_dicts = []
        
        # Create the ActorRollout worker group
        if 'actor_rollout_ref' in self.resource_pool_to_cls[resource_pool]:
            print("Creating ActorRollout worker group")
            try:
                self.actor_rollout_wg = self.ray_worker_group_cls(
                    resource_pool=resource_pool,
                    ray_cls_with_init=self.resource_pool_to_cls[resource_pool]['actor_rollout_ref'],  # Pass a single RayClassWithInitArgs object
                    bin_pack=True,
                    name_prefix="actor_rollout_worker",
                    detached=False
                )
                self.wg_dicts.append(self.actor_rollout_wg)
                print("ActorRollout worker group created successfully")
            except Exception as e:
                print(f"Error creating ActorRollout worker group: {e}")
                traceback.print_exc()
                raise
        
        # Create the Critic worker group if configured
        if 'critic' in self.resource_pool_to_cls[resource_pool]:
            print("Creating Critic worker group")
            try:
                self.critic_wg = self.ray_worker_group_cls(
                    resource_pool=resource_pool,
                    ray_cls_with_init=self.resource_pool_to_cls[resource_pool]['critic'],  # Pass a single RayClassWithInitArgs object
                    bin_pack=True,
                    name_prefix="critic_worker",
                    detached=False
                )
                self.wg_dicts.append(self.critic_wg)
                print("Critic worker group created successfully")
            except Exception as e:
                print(f"Error creating Critic worker group: {e}")
                traceback.print_exc()
                raise
        
        # Create the RewardModel worker group if configured
        if 'rm' in self.resource_pool_to_cls[resource_pool]:
            print("Creating RewardModel worker group")
            try:
                self.rm_wg = self.ray_worker_group_cls(
                    resource_pool=resource_pool,
                    ray_cls_with_init=self.resource_pool_to_cls[resource_pool]['rm'],  # Pass a single RayClassWithInitArgs object
                    bin_pack=True,
                    name_prefix="reward_model_worker",
                    detached=False
                )
                self.wg_dicts.append(self.rm_wg)
                print("RewardModel worker group created successfully")
            except Exception as e:
                print(f"Error creating RewardModel worker group: {e}")
                traceback.print_exc()
                raise
                
        print(f"Created {len(self.wg_dicts)} worker groups")
        
        # Direct fix for the worker method issue
        # The issue is that the worker methods aren't properly exposed through Ray
        # This modification explicitly forwards important methods to the actual worker instance
        if hasattr(self, 'actor_rollout_wg') and hasattr(self.actor_rollout_wg, '_workers') and len(self.actor_rollout_wg._workers) > 0:
            import types
            
            # Define a method to generate a wrapper function that calls the remote worker's method
            def create_worker_method_wrapper(worker_group, method_name):
                def wrapper(*args, **kwargs):
                    # Get the first worker
                    worker = worker_group._workers[0]
                    # Call the method remotely
                    try:
                        result_ref = getattr(worker, method_name).remote(*args, **kwargs)
                        # Get and return the result
                        return ray.get(result_ref)
                    except AttributeError:
                        print(f"[ERROR] Worker doesn't have method '{method_name}'")
                        print(f"[DEBUG] Available methods: {dir(worker)}")
                        raise
                return wrapper
            
            # List of important methods that need direct access
            key_methods = [
                'generate_sequences',
                'compute_log_prob',
                'compute_ref_log_prob',
                'compute_values',
                'update_actor',
                'update_critic',
                'compute_rm_score',
                'save_checkpoint',
                'load_model_parameters'
            ]
            
            # Create wrapper methods for each worker group
            worker_groups = []
            if hasattr(self, 'actor_rollout_wg') and hasattr(self.actor_rollout_wg, '_workers') and len(self.actor_rollout_wg._workers) > 0:
                worker_groups.append(('actor_rollout_wg', self.actor_rollout_wg))
            if hasattr(self, 'critic_wg') and hasattr(self.critic_wg, '_workers') and len(self.critic_wg._workers) > 0:
                worker_groups.append(('critic_wg', self.critic_wg))
            if hasattr(self, 'rm_wg') and hasattr(self.rm_wg, '_workers') and len(self.rm_wg._workers) > 0:
                worker_groups.append(('rm_wg', self.rm_wg))
                
            # Add method wrappers to all worker groups
            for group_name, group in worker_groups:
                for method_name in key_methods:
                    # Only add methods if they don't already exist
                    if not hasattr(group, method_name):
                        try:
                            wrapper = create_worker_method_wrapper(group, method_name)
                            setattr(group, method_name, types.MethodType(wrapper, group))
                            print(f"[INFO] Added '{method_name}' method to {group_name}")
                        except Exception as e:
                            print(f"[WARNING] Failed to add method '{method_name}' to {group_name}: {e}")
        else:
            raise NotImplementedError

    def _save_checkpoint(self):
        actor_local_path = os.path.join(self.config.trainer.default_local_dir, 'actor',
                                        f'global_step_{self.global_steps}')
        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
            self.config.trainer.default_hdfs_dir, 'actor')
        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path)

        if self.use_critic:
            critic_local_path = os.path.join(self.config.trainer.default_local_dir, 'critic',
                                             f'global_step_{self.global_steps}')
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
                self.config.trainer.default_hdfs_dir, 'critic')
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        print(f"Starting training loop with n_gpus_per_node={self.config.trainer.n_gpus_per_node}, nnodes={self.config.trainer.nnodes}")
        
        # Initialize global steps
        self.global_steps = 0
        
        if self.config.trainer.get('val_before_train', False):
            print("Validating before training...")
            metrics = self._validate()
            print(f"Validation metrics: {metrics}")
            
            # Return early if val_only is set
            if self.config.trainer.get('val_only', False):
                print("Validation only mode. Exiting.")
                return
                
        # We start from step 1
        self.global_steps += 1
        
        # Create a logger for LLMGenerationManager
        import logging
        logging_logger = logging.getLogger("LLMGenerationManager")
        logging_logger.setLevel(logging.INFO)
        if not logging_logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            logging_logger.addHandler(handler)
        
        gen_config = GenerationConfig(
            max_turns=self.config.get('max_turns', 1),
            max_start_length=self.config.data.max_start_length,
            max_prompt_length=self.config.data.max_prompt_length,
            max_response_length=self.config.data.max_response_length,
            max_obs_length=self.config.data.max_obs_length,
            logging=self.config.logging,
            num_gpus=self.config.trainer.n_gpus_per_node,
            no_think_rl=self.config.algorithm.get('no_think_rl', False),
            state_masking=self.config.actor_rollout_ref.actor.get('state_masking', False),
            start_state_marker=self.config.algorithm.state_masking.get('start_state_marker', ''),
            end_state_marker=self.config.algorithm.state_masking.get('end_state_marker', ''),
        )

        generation_manager = LLMGenerationManager(
            tokenizer=self.tokenizer,
            actor_rollout_wg=self.actor_rollout_wg,
            env_class=self.env_class,
            config=gen_config,
            logger=logging_logger,  # Use the proper logging logger
            is_validation=False,
        )

        envs = [self.env.copy() for _ in range(self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n_agent)] 



        # start training loop
        for epoch in range(self.config.trainer.get('total_epochs', 1)):
            for batch_dict in self.train_dataloader:
                print(f'epoch {epoch}, step {self.global_steps}')

                # update ref_policy_wg
                if self.config.trainer.get('ref_update_steps') is not None and self.global_steps % self.config.trainer.ref_update_steps == 0:
                    self.actor_rollout_wg.save_checkpoint(
                        local_path=f'./log/temp/actor_rollout_wg_global_step_{self.global_steps}',
                        hdfs_path=None
                    )
                    self.ref_policy_wg.load_model_parameters(
                        source_model_path=f'./log/temp/actor_rollout_wg_global_step_{self.global_steps}',
                        strict=True
                    )
                    print(f"load parameters from ./log/temp/actor_rollout_wg_global_step_{self.global_steps} to ref_policy_wg")

                metrics = {}
                timing_raw = {}

                batch = {}
                batch['input_ids'] = batch_dict['input_ids']
                batch['attention_mask'] = batch_dict['attention_mask']
                batch['position_ids'] = batch_dict['position_ids']
                batch['extra_info'] = batch_dict['extra_info']
                batch['data_source'] = batch_dict['data_source']

                env_seeds = [i['index'] for i in batch['extra_info']]
                print("env_seeds:", env_seeds)
                for env, seed in zip(envs, env_seeds):
                    env.reset(seed=seed)


                # pop those keys for generation
                gen_batch = batch.copy()
                gen_batch.pop('extra_info')
                gen_batch.pop('data_source')

                ####################
                # original code here

                # with _timer('gen', timing_raw):
                #     gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                #     batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                #                                              dtype=object)
                #     # repeat to align with repeated responses in rollout
                #     batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                #     batch = batch.union(gen_batch_output)

                #     # output batch to file
                #     self._record_batch(batch, path=f'.log/{self.config.trainer.experiment_name}/gen_batch.txt')

                ####################
                # Below is aLL about agents - the "LLM + forloop"
                ####################

                with _timer('step', timing_raw):
                    """
                    keep rolling to generate K turns of responses.
                    when doing this, update the "original right side" when new responses are generated.
                    finally, concatenate the "original left side" and "original right side" to get the final thing to feed to train the model.

                    Left-pad prompts, right-gen flow, Tensors dance like stardust glow.
                    Errors swarm? Stay calm, don't fret- Code with coffee, debug sunset.
                    """

                    first_input_ids = gen_batch['input_ids'][:, -gen_config.max_start_length:].clone()
                    output_dir = (f"{self.config.logging.log_image_dir}/"
                                 f"{self.config.trainer.experiment_name}/"
                                 f"train/"
                                 f"step_{self.global_steps}")

                    with _timer('gen', timing_raw):
                        generation_manager.timing_raw = timing_raw
                        final_gen_batch_output = generation_manager.run_llm_loop(
                            gen_batch=gen_batch,
                            envs=envs,
                            initial_input_ids=first_input_ids,
                            output_dir=output_dir,
                            global_steps=self.global_steps,
                        )

                    with torch.no_grad():
                        output = self.actor_rollout_wg.compute_log_prob(final_gen_batch_output)
                        final_gen_batch_output = final_gen_batch_output.union(output)
                    
                    if self.config.algorithm.get('adv_estimator') == 'grpo': # NOTE we currently use seed to group, better use prompt (hash) to group
                        batch['uid'] = np.array([str(i) for i in env_seeds], dtype=object)
                    elif self.config.algorithm.get('adv_estimator') == 'brpo':
                        batch['uid'] = np.array(["" for _ in range(len(batch['input_ids']))], dtype=object)
                    elif self.config.algorithm.get('adv_estimator') == 'arpo':
                        batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch['input_ids']))], dtype=object) # No Relative normalization

                    # reward
                    batch['reward'] = np.array([0 for _ in range(len(envs))], dtype=object)
                    for idx, env in enumerate(envs):
                        batch['reward'][idx] = env.reward

                    # metric for two-armed bandit
                    # NOTE here we assume invalid action is 0, low arm is 1, high arm is 2
                    if batch['data_source'][0] == 'two_armed_bandit':
                        batch['bandit_metrics'] = np.array([0 for _ in range(len(envs))], dtype=object)
                        for idx, env in enumerate(envs):
                            batch['bandit_metrics'][idx] = env.get_last_action()
                    # metrics for actions
                    batch['total_env'] = np.array([1 for _ in range(len(envs))], dtype=object)
                    batch['finished_env'] = np.array([0 for _ in range(len(envs))], dtype=object)
                    batch['success_env'] = np.array([0 for _ in range(len(envs))], dtype=object)
                    batch['traj_length'] = np.array([0 for _ in range(len(envs))], dtype=object)
                    batch['valid_action'] = np.array([0 for _ in range(len(envs))], dtype=object)
                    batch['effective_action'] = np.array([0 for _ in range(len(envs))], dtype=object)
                    batch['effective_action_ratio'] = np.array([0 for _ in range(len(envs))], dtype=object)
                    for idx, env in enumerate(envs):
                        batch['finished_env'][idx] = int(env.finished())
                        batch['success_env'][idx] = int(env.success())
                        tracking_vars = env.get_tracking_variables()
                        batch['traj_length'][idx] = len(tracking_vars['actions'])
                        batch['valid_action'][idx] = sum(1 for x in tracking_vars['actions_valid'] if x is not None)
                        batch['effective_action'][idx] = sum(1 for x in tracking_vars['actions_effective'] if x is not None)
                        batch['effective_action_ratio'][idx] = sum(1 for x in tracking_vars['actions_effective'] if x is not None) / len(tracking_vars['actions'])

                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(final_gen_batch_output)

                    ####################
                    ####################


                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch['global_token_num'] = torch.sum(batch['attention_mask'], dim=-1).tolist()

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer('ref', timing_raw):
                            ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer('values', timing_raw):
                            values = self.actor_rollout_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer('adv', timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.actor_rollout_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_tensor = self.reward_fn(batch)
                        batch['token_level_scores'] = reward_tensor

                        # compute rewards. apply_kl_penalty if available, no kl_loss or kl_penalty for GAE
                        if not self.config.actor_rollout_ref.actor.get('use_kl_loss', False) or self.config.algorithm.get('adv_estimator') != 'gae':
                            batch, kl_metrics = apply_kl_penalty(batch,
                                                                 kl_ctrl=self.kl_ctrl,
                                                                 kl_penalty=self.config.algorithm.get('kl_penalty', 'kl'))
                            metrics.update(kl_metrics)
                        else:
                            batch['token_level_rewards'] = batch['token_level_scores']

                        # compute advantages, executed on the driver process
                        batch = compute_advantage(batch,
                                                  adv_estimator=self.config.algorithm.get('adv_estimator', 'gae'),
                                                  gamma=self.config.algorithm.get('gamma', 1.0),
                                                  lam=self.config.algorithm.get('lam', 1.0),
                                                  num_repeat=self.config.actor_rollout_ref.rollout.n)

                    # update critic
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.actor_rollout_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output['metrics'])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.get('critic_warmup') is not None and self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer('update_actor', timing_raw):
                            if self.config.actor_rollout_ref.actor.get('state_masking', False):
                                batch,metrics = self._create_loss_mask(batch, metrics)
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output['metrics'])
                        metrics.update(actor_output_metrics)

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.get('test_freq') > 0 and \
                        self.global_steps % self.config.trainer.test_freq == 0:
                        with _timer('testing', timing_raw):
                            val_metrics: dict = self._validate()

                    if self.config.trainer.get('save_freq') > 0 and \
                            self.global_steps % self.config.trainer.save_freq == 0:
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

                # TODO: make a canonical logger that supports various backend
                print(f"Metrics at step {self.global_steps}: {metrics}")

                self.global_steps += 1

                if self.global_steps >= self.total_training_steps:

                    # perform validation after training
                    if self.val_reward_fn is not None:
                        val_metrics = self._validate()
                    return
    def _create_loss_mask(self, batch, metrics):
        """Create loss mask for action tokens (non-state tokens).
        
        This mask is used to calculate policy gradients only for tokens that are generated by the model
        (actions/reasoning) and NOT for observation tokens that come from the environment.
        
        The mask has 1s for action tokens (include in loss calculation) and 0s for observation tokens 
        (exclude from loss calculation).
        """
        response_length = batch['responses'].shape[-1]
        response_mask = batch['attention_mask'][:, -response_length:]

        # Initialize action mask (all tokens initially considered as actions)
        action_mask = torch.ones_like(response_mask)
        
        responses = [self.tokenizer.decode(resp, skip_special_tokens=False) for resp in batch['responses']]

        for i, response in enumerate(responses):
            # Find all pairs of start and end marker positions
            start_marker = self.config.algorithm.state_masking.get('start_state_marker', '')
            end_marker = self.config.algorithm.state_masking.get('end_state_marker', '')   
            
            # Get all start and end positions
            start_positions = [m.start() for m in re.finditer(re.escape(start_marker), response)]
            end_positions = [m.start() + len(end_marker) for m in re.finditer(re.escape(end_marker), response)]
            
            # Convert character positions to token positions
            for start, end in zip(start_positions, end_positions):
                prefix_to_start = response[:start]
                state_section = response[start:end]
                
                start_tokens = self.tokenizer.encode(prefix_to_start, add_special_tokens=False)
                state_tokens = self.tokenizer.encode(state_section, add_special_tokens=False)
                
                start_token_pos = len(start_tokens)
                end_token_pos = start_token_pos + len(state_tokens)
                
                # Set observation tokens to 0 in the action mask (exclude from loss)
                if start_token_pos < action_mask.shape[1] and end_token_pos <= action_mask.shape[1]:
                    action_mask[i, start_token_pos:end_token_pos] = 0
        
        # Apply response mask to ensure we only consider valid tokens
        loss_mask = action_mask * response_mask
        batch['loss_mask'] = loss_mask
        
        # Debug print
        print("\nRaw batch[0] (before masking):\n", self.tokenizer.decode(batch['responses'][0]))
        response_ids = batch['responses'][0]
        
        # Now unmasked IDs are action tokens (loss_mask = 1)
        unmasked_ids = response_ids[loss_mask[0] == 1]
        print("\nUnmasked batch[0] (action tokens):\n", self.tokenizer.decode(unmasked_ids))
        
        # Now masked IDs are observation tokens (loss_mask = 0)
        masked_ids = response_ids[loss_mask[0] == 0]
        print("\nMasked batch[0] (observation tokens):\n", self.tokenizer.decode(masked_ids))
        
        # Update metrics with meaningful names for actions
        metrics.update({
            'action_tokens/total': loss_mask.sum().item(),
            'action_tokens/coverage': (loss_mask.sum() / response_mask.sum()).item(),
        })
        
        return batch, metrics

    def _validate(self):
        """
        The training loop of PPO with global metric computation.
        Accumulates metrics across all batches before computing final statistics.
        """
        import torch
        # Initialize global metric storage
        global_token_scores = []
        global_metrics = {}
        metrics = defaultdict(list)

        self.val_num += 1

        gen_config = GenerationConfig(
            max_turns=self.config.get('max_turns', 1),
            max_start_length=self.config.data.max_start_length,
            max_prompt_length=self.config.data.max_prompt_length,
            max_response_length=self.config.data.max_response_length,
            max_obs_length=self.config.data.max_obs_length,
            logging=self.config.logging,
            num_gpus=self.config.trainer.n_gpus_per_node,
            no_think_rl=self.config.algorithm.get('no_think_rl', False),
            state_masking=self.config.actor_rollout_ref.actor.get('state_masking', False),
            start_state_marker=self.config.algorithm.state_masking.get('start_state_marker', ''),
            end_state_marker=self.config.algorithm.state_masking.get('end_state_marker', ''),
        )

        # Agent config preparation
        generation_manager = LLMGenerationManager(
            tokenizer=self.tokenizer,
            actor_rollout_wg=self.actor_rollout_wg,
            env_class=self.env_class,
            config=gen_config,
            logger=logging_logger,  # Use the proper logging logger
            is_validation=True,
        )

        envs = [self.val_env.copy() for _ in range(self.config.data.val_batch_size * self.config.actor_rollout_ref.rollout.n_agent)]
        val_global_steps = 1

        for batch_dict in self.val_dataloader:
            timing_raw = {}
            test_batch = {}
            test_batch['input_ids'] = batch_dict['input_ids']
            test_batch['attention_mask'] = batch_dict['attention_mask']
            test_batch['position_ids'] = batch_dict['position_ids']
            test_batch['extra_info'] = batch_dict['extra_info']
            test_batch['data_source'] = batch_dict['data_source']

            env_seeds = [i['index'] for i in test_batch['extra_info']]
            print("env_seeds:", env_seeds)
            for env, seed in zip(envs, env_seeds):
                env.reset(seed=seed)
            
            test_gen_batch = test_batch.copy()
            test_gen_batch.pop('extra_info')
            test_gen_batch.pop('data_source')
            test_gen_batch['meta_info'] = {
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': False,
                'validate': True,
            }
            with _timer('step', timing_raw):
                first_input_ids = test_gen_batch['input_ids'][:, -gen_config.max_start_length:].clone()
                output_dir = (f"{self.config.logging.log_image_dir}/"
                                f"{self.config.trainer.experiment_name}/"
                                f"validation_{self.val_num}/"
                                f"step_{val_global_steps}")
                with _timer('gen', timing_raw):
                    generation_manager.timing_raw = timing_raw
                    final_gen_batch_output = generation_manager.run_llm_loop(
                        gen_batch=test_gen_batch,
                        envs=envs,
                        initial_input_ids=first_input_ids,
                        output_dir=output_dir,
                        global_steps=val_global_steps,
                    )
                with torch.no_grad():
                    output = self.actor_rollout_wg.compute_log_prob(final_gen_batch_output)
                    final_gen_batch_output = final_gen_batch_output.union(output)

                test_batch['reward'] = np.array([0 for _ in range(len(envs))], dtype=object)
                for idx, env in enumerate(envs):
                    test_batch['reward'][idx] = env.reward

                if test_batch['data_source'][0] == 'two_armed_bandit':
                    # metric for two-armed bandit
                    # NOTE here we assume invalid action is 0, low arm is 1, high arm is 2
                    test_batch['bandit_metrics'] = np.array([0 for _ in range(len(envs))], dtype=object)
                    for idx, env in enumerate(envs):
                        test_batch['bandit_metrics'][idx] = env.get_last_action()
                metrics['bandit_metrics'].append(test_batch['bandit_metrics'])
                
                test_batch['total_env'] = np.array([1 for _ in range(len(envs))], dtype=object)
                test_batch['finished_env'] = np.array([0 for _ in range(len(envs))], dtype=object)
                test_batch['success_env'] = np.array([0 for _ in range(len(envs))], dtype=object)
                test_batch['traj_length'] = np.array([0 for _ in range(len(envs))], dtype=object)
                test_batch['valid_action'] = np.array([0 for _ in range(len(envs))], dtype=object)
                test_batch['effective_action'] = np.array([0 for _ in range(len(envs))], dtype=object)
                test_batch['effective_action_ratio'] = np.array([0 for _ in range(len(envs))], dtype=object)
                for idx, env in enumerate(envs):
                    test_batch['finished_env'][idx] = int(env.finished())
                    test_batch['success_env'][idx] = int(env.success())
                    tracking_vars = env.get_tracking_variables()
                    test_batch['traj_length'][idx] = len(tracking_vars['actions'])
                    test_batch['valid_action'][idx] = sum(1 for x in tracking_vars['actions_valid'] if x is not None)
                    test_batch['effective_action'][idx] = sum(1 for x in tracking_vars['actions_effective'] if x is not None)
                    test_batch['effective_action_ratio'][idx] = sum(1 for x in tracking_vars['actions_effective'] if x is not None) / len(tracking_vars['actions'])

                # Accumulate batch metrics into global storage
                global_token_scores.append(test_batch['reward'])

        global_scores = np.concatenate(global_token_scores, axis=0)
        global_metrics = {
            'global_score/mean': float(global_scores.mean()),
            'global_score/max': float(global_scores.max()),
            'global_score/min': float(global_scores.min()),
            'global_score/std': float(global_scores.std()),
            'validate_metric/total_env': int(np.array(metrics['total_env'], dtype=np.int16).sum()),
            'validate_metric/finished_env': int(np.array(metrics['finished_env'], dtype=np.int16).sum()),
            'validate_metric/success_env': int(np.array(metrics['success_env'], dtype=np.int16).sum()),
            'validate_metric/traj_length': float(np.array(metrics['traj_length'], dtype=np.int16).mean()),
            'validate_metric/valid_action': float(np.array(metrics['valid_action'], dtype=np.int16).mean()),
            'validate_metric/effective_action': float(np.array(metrics['effective_action'], dtype=np.int16).mean()),
            'validate_metric/effective_action_ratio': float(np.array(metrics['effective_action_ratio'], dtype=np.float32).mean()),
        }
        if 'bandit_metrics' in metrics: # NOTE hard code for two-armed bandit
            batch_action = np.array(metrics['bandit_metrics'], dtype=np.int16)
            global_metrics['validate_metric/n_low_arm'] = int(np.sum(batch_action == 1))
            global_metrics['validate_metric/n_high_arm'] = int(np.sum(batch_action == 2))
            global_metrics['validate_metric/n_invalid'] = int(np.sum(batch_action == 0))
        print("global_metrics", global_metrics)
        return global_metrics

    def _balance_batch(self, batch, metrics, logging_prefix='global_seqlen'):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        resp_info = _compute_response_info(batch)
        
        # Get sequence lengths (prompt + response lengths)
        sequence_lengths = resp_info['prompt_length'] + resp_info['response_length']
        
        # Get batch size and number of sequences
        batch_size = sequence_lengths.size(0)

        # Check if we are using model parallel
        if self.config.actor_rollout_ref.actor.get('world_size', 1) > 1:
            # Get the number of dp ranks
            dp_world_size = self.config.actor_rollout_ref.actor.get('dp_world_size', 1)

            # Get balanced partitions based on sequence lengths
            reorder_idxs, partition_stats = get_seqlen_balanced_partitions(
                sequence_lengths, dp_world_size
            )

            if not isinstance(reorder_idxs, torch.Tensor):
                reorder_idxs = torch.tensor(reorder_idxs)

            # Log sequence length distribution if needed
            log_seqlen_unbalance(
                partition_stats, batch_size, dp_world_size,
                metrics, logging_prefix
            )

            # Apply reordering to the batch
            batch = batch.reindex(reorder_idxs)
        
        # Ensure loss mask is in the correct precision
        if 'loss_mask' in batch:
            device = batch['responses'].device
            dtype = batch['responses'].dtype
            batch['loss_mask'] = batch['loss_mask'].to(device=device, dtype=dtype)
            
        return batch, metrics
