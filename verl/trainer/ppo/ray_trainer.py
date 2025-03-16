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

import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type, Dict

import re
import json
from collections import defaultdict

import numpy as np
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance


import re
from ragen.llm_agent.generation import LLMGenerationManager, GenerationConfig

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


import torch
from verl.utils.torch_functional import masked_mean


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty='kl'):
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


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1):
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
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last


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
        from torch.utils.data import DataLoader
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
        self.train_dataset = RLHFDataset(parquet_files=self.config.data.train_files,
                                         tokenizer=self.tokenizer,
                                         prompt_key=self.config.data.prompt_key,
                                         max_prompt_length=self.config.data.max_prompt_length,
                                         filter_prompts=True,
                                         return_raw_chat=self.config.data.get('return_raw_chat', False),
                                         truncation='error')
        if self.config.data.get('train_data_num') is not None:
            if self.config.data.train_data_num > len(self.train_dataset.dataframe):
                print(f"[WARNING] training dataset size is smaller than desired size. Using the dataset as the original size {len(self.train_dataset.dataframe)}")
            else:
                self.train_dataset.dataframe = self.train_dataset.dataframe.sample(self.config.data.train_data_num, random_state=42)
        print(f"filtered training dataset size: {len(self.train_dataset.dataframe)}")

        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                           batch_size=self.config.data.train_batch_size,
                                           shuffle=self.config.data.shuffle_train_dataloader,
                                           drop_last=True,
                                           collate_fn=collate_fn)

        self.val_dataset = RLHFDataset(parquet_files=self.config.data.val_files,
                                       tokenizer=self.tokenizer,
                                       prompt_key=self.config.data.prompt_key,
                                       max_prompt_length=self.config.data.max_prompt_length,
                                       filter_prompts=True,
                                       return_raw_chat=self.config.data.get('return_raw_chat', False),
                                       truncation='error')
        if self.config.data.get('val_data_num') is not None:
            if self.config.data.val_data_num > len(self.val_dataset.dataframe):
                print(f"[WARNING] validation dataset size is smaller than desired size. Using the dataset as the original size {len(self.val_dataset.dataframe)}")
            else:
                self.val_dataset.dataframe = self.val_dataset.dataframe.sample(self.config.data.val_data_num, random_state=42)
        print(f"filtered validation dataset size: {len(self.val_dataset.dataframe)}")

        self.val_dataloader = DataLoader(dataset=self.val_dataset,
                                         batch_size=self.config.data.val_batch_size,
                                         shuffle=True,
                                         drop_last=True,
                                         collate_fn=collate_fn)

        print(f'Size of train dataloader: {len(self.train_dataloader)}')
        print(f'Size of val dataloader: {len(self.val_dataloader)}')
        
        assert len(self.train_dataloader) >= 1
        assert len(self.val_dataloader) >= 1

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
        self.logger.info("Initialize workers...")
        self.resource_pool_manager.create_resource_pool()
        
        config = OmegaConf.to_container(self.config, resolve=True)

        # Create the actor, rollout and ref workers
        has_rollout_role = 'rollout' in config['actor_rollout_ref']['roles']
        has_actor_role = 'actor' in config['actor_rollout_ref']['roles']
        has_ref_role = 'ref' in config['actor_rollout_ref']['roles']

        # Initialize worker arguments
        worker_kwargs = {}
        
        # Check for required config sections
        has_ref_config = 'ref' in config
        if has_ref_role and not has_ref_config:
            self.logger.warning("RefPolicy role is defined but config.ref is missing. Skipping reference policy initialization.")
            has_ref_role = False
            
        # Safely access config sections using get() to avoid KeyError
        ref_config = config.get('ref', {}) if has_ref_config else {}
        rm_config = config.get('rm', {}) if 'rm' in config else {}
        critic_config = config.get('critic', {}) if 'critic' in config else {}
        
        # Only add configs that exist
        actor_rollout_ref_kwargs = {'args': (), 'kwargs': {'config': config}}
        worker_kwargs['actor_rollout_ref'] = actor_rollout_ref_kwargs

        # Only initialize critic if config exists
        if 'critic' in config:
            critic_kwargs = {'args': (), 'kwargs': {'config': critic_config}}
            worker_kwargs['critic'] = critic_kwargs

        # Only initialize reward model if config exists
        if 'rm' in config:
            rm_kwargs = {'args': (), 'kwargs': {'config': rm_config}}
            worker_kwargs['rm'] = rm_kwargs

        # Initialize class dictionary with implementation details
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}
        
        ray_classes = {}
        ray_classes['actor_rollout_ref'] = ray.remote(ActorRolloutRefWorker)
        
        if 'critic' in config:
            ray_classes['critic'] = ray.remote(CriticWorker)
        
        if 'rm' in config:
            ray_classes['rm'] = ray.remote(RewardModelWorker)

        worker_group = self.ray_worker_group_cls(
            ray_classes, worker_kwargs, preemptible_node_type="gpu_p4",
            min_replicas=1, max_replicas=1, detached=False)
        
        # Initialize class dictionary with most implementation details
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}
        
        # Initialize worker group list
        self.wg_dicts = []

        # setup rollout worker config
        self.rollout_config = OmegaConf.create()
        with open_dict(self.rollout_config):
            self.rollout_config.training = OmegaConf.create()
            self.rollout_config.training.stop_token = self.tokenizer.eos_token_id
            self.rollout_config.training.pad_token = self.tokenizer.pad_token_id
            
            # Convert torch.dtype to string for OmegaConf compatibility
            # Important: Use strings for torch data types with OmegaConf
            if hasattr(self.config.actor_rollout_ref, 'model') and hasattr(self.config.actor_rollout_ref.model, 'torch_dtype'):
                if self.config.actor_rollout_ref.model.torch_dtype == 'torch.float16':
                    self.rollout_config.training.torch_dtype = 'torch.float16'
                elif self.config.actor_rollout_ref.model.torch_dtype == 'torch.bfloat16':
                    self.rollout_config.training.torch_dtype = 'torch.bfloat16'
                else:
                    self.rollout_config.training.torch_dtype = 'torch.float32'
            else:
                self.rollout_config.training.torch_dtype = 'torch.float16'
                
            # Set attn_implementation if present in config, otherwise default to "flash_attention_2"
            if hasattr(self.config.actor_rollout_ref, 'model') and hasattr(self.config.actor_rollout_ref.model, 'attn_implementation'):
                self.rollout_config.training.attn_implementation = self.config.actor_rollout_ref.model.attn_implementation
            else:
                # Check if CUDA is available before defaulting to flash attention
                if torch.cuda.is_available():
                    self.rollout_config.training.attn_implementation = "flash_attention_2"
                else:
                    self.rollout_config.training.attn_implementation = "eager"
                    
            # Add required training parameters
            min_response_length = 0
            if hasattr(self.config, 'data') and hasattr(self.config.data, 'min_response_length'):
                min_response_length = self.config.data.min_response_length
            self.rollout_config.training.min_length = min_response_length
            
            max_length = 400  # Default value
            if hasattr(self.config, 'data') and hasattr(self.config.data, 'max_response_length'):
                max_length = self.config.data.max_response_length
            self.rollout_config.training.max_length = max_length
            
            self.rollout_config.training.temperature = 0.7
            if hasattr(self.config.actor_rollout_ref, 'rollout') and hasattr(self.config.actor_rollout_ref.rollout, 'temperature'):
                self.rollout_config.training.temperature = self.config.actor_rollout_ref.rollout.temperature
            
            self.rollout_config.training.num_return_sequences = 1
            
            # Access max_obs_length safely
            max_obs_length = 200  # Default value
            if hasattr(self.config, 'data') and hasattr(self.config.data, 'max_obs_length'):
                max_obs_length = self.config.data.max_obs_length
            self.rollout_config.training.max_obs_length = max_obs_length

        if self.hybrid_engine:
            # create actor_rollout class for the hybrid engine
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)

            # Add Flash Attention configuration to actor_rollout
            actor_rollout_config = self.config.actor_rollout_ref.copy()
            
            # Use open_dict to safely modify the config if needed
            with open_dict(actor_rollout_config):
                # Make sure model config exists
                if not hasattr(actor_rollout_config, 'model'):
                    actor_rollout_config.model = OmegaConf.create()
                
                # Add the dtype and attention implementation to the model config
                # These values are already string representations
                actor_rollout_config.model.torch_dtype = self.rollout_config.training.torch_dtype
                actor_rollout_config.model.attn_implementation = self.rollout_config.training.attn_implementation
            
            # Add to resource pool with proper key - use string key matching the role
            self.resource_pool_to_cls[resource_pool][Role.ActorRollout.name] = RayClassWithInitArgs(
                self.role_worker_mapping[Role.ActorRollout],
                kwargs=dict(
                    config=actor_rollout_config,
                    tokenizer=self.tokenizer,
                    rollout_config=self.rollout_config,
                    role=Role.ActorRollout.name
                )
            )
            
            # If critic is needed, set it up similarly
            if self.use_critic:
                # Check if critic config exists
                if not hasattr(self.config, 'critic'):
                    print("[WARNING] Critic role is defined but config.critic is missing. Skipping critic initialization.")
                    self.use_critic = False
                else:
                    resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
                    self.resource_pool_to_cls[resource_pool][Role.Critic.name] = RayClassWithInitArgs(
                        self.role_worker_mapping[Role.Critic],
                        kwargs=dict(
                            config=self.config.critic,
                            tokenizer=self.tokenizer,
                            role=Role.Critic.name
                        )
                    )
                
            # Set up reference policy if needed
            if self.use_reference_policy:
                # Check if reference policy config exists
                if not hasattr(self.config, 'ref'):
                    print("[WARNING] RefPolicy role is defined but config.ref is missing. Skipping reference policy initialization.")
                    self.use_reference_policy = False
                else:
                    resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
                    self.resource_pool_to_cls[resource_pool][Role.RefPolicy.name] = RayClassWithInitArgs(
                        self.role_worker_mapping[Role.RefPolicy],
                        kwargs=dict(
                            config=self.config.ref,
                            tokenizer=self.tokenizer,
                            role=Role.RefPolicy.name
                        )
                    )
                
            # Set up reward model if needed
            if self.use_rm:
                # Check if reward model config exists
                if not hasattr(self.config, 'rm'):
                    print("[WARNING] RewardModel role is defined but config.rm is missing. Skipping reward model initialization.")
                    self.use_rm = False
                else:
                    resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
                    self.resource_pool_to_cls[resource_pool][Role.RewardModel.name] = RayClassWithInitArgs(
                        self.role_worker_mapping[Role.RewardModel],
                        kwargs=dict(
                            config=self.config.rm,
                            tokenizer=self.tokenizer,
                            role=Role.RewardModel.name
                        )
                    )
            
        else:
            raise NotImplementedError

        # Create worker groups
        all_wg = {}
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            # Create colocated worker cls with a dict of role to cls
            colocated_cls_dict = {}
            for role, cls in class_dict.items():
                colocated_cls_dict[role] = cls
            colocated_worker_cls = create_colocated_worker_cls(colocated_cls_dict)

            # create RayWorkerGroup
            worker_group = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=colocated_worker_cls,
                bin_pack=True
            )
            self.wg_dicts.append(worker_group)
            for role in class_dict.keys():
                all_wg[role] = worker_group

        # Create worker shortcuts (only if the corresponding worker exists)
        # Make sure to check both flag and presence in all_wg
        if self.hybrid_engine and Role.ActorRollout.name in all_wg:
            self.actor_rollout_wg = all_wg[Role.ActorRollout.name]
        
        if self.use_critic and Role.Critic.name in all_wg:
            self.critic_wg = all_wg[Role.Critic.name]
            
        if self.use_reference_policy and Role.RefPolicy.name in all_wg:
            self.ref_wg = all_wg[Role.RefPolicy.name]
            
        if self.use_rm and Role.RewardModel.name in all_wg:
            self.rm_wg = all_wg[Role.RewardModel.name]
            
        # Direct fix for the worker method issue
        # The issue is that the worker methods aren't properly exposed through Ray
        # This modification explicitly forwards important methods to the actual worker instance
        if hasattr(self, 'actor_rollout_wg') and hasattr(self.actor_rollout_wg, '_workers') and len(self.actor_rollout_wg._workers) > 0:
            import ray
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
            if hasattr(self, 'ref_wg') and hasattr(self.ref_wg, '_workers') and len(self.ref_wg._workers) > 0:
                worker_groups.append(('ref_wg', self.ref_wg))
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

        
        logger = self.logger
        self.global_steps = 0
        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()
            if self.config.trainer.get('val_only', False):
                return

        # we start from step 1
        self.global_steps += 1



        # Agent config preparation
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
            logger = logger,
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

                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n_agent, interleave=True)

                env_seeds = [i['index'] for i in batch.non_tensor_batch['extra_info']]
                print("env_seeds:", env_seeds)
                for env, seed in zip(envs, env_seeds):
                    env.reset(seed=seed)


                # pop those keys for generation
                gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])

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

                    first_input_ids = gen_batch.batch['input_ids'][:, -gen_config.max_start_length:].clone()
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
                        batch.non_tensor_batch['uid'] = np.array([str(i) for i in env_seeds], dtype=object)
                    elif self.config.algorithm.get('adv_estimator') == 'brpo':
                        batch.non_tensor_batch['uid'] = np.array(["" for _ in range(len(batch.batch))], dtype=object)
                    elif self.config.algorithm.get('adv_estimator') == 'arpo':
                        batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object) # No Relative normalization

                    # reward
                    batch.non_tensor_batch['reward'] = np.array([0 for _ in range(len(envs))], dtype=object)
                    for idx, env in enumerate(envs):
                        batch.non_tensor_batch['reward'][idx] = env.reward

                    # metric for two-armed bandit
                    # NOTE here we assume invalid action is 0, low arm is 1, high arm is 2
                    if batch.non_tensor_batch['data_source'][0] == 'two_armed_bandit':
                        batch.non_tensor_batch['bandit_metrics'] = np.array([0 for _ in range(len(envs))], dtype=object)
                        for idx, env in enumerate(envs):
                            batch.non_tensor_batch['bandit_metrics'][idx] = env.get_last_action()
                    # metrics for actions
                    batch.non_tensor_batch['total_env'] = np.array([1 for _ in range(len(envs))], dtype=object)
                    batch.non_tensor_batch['finished_env'] = np.array([0 for _ in range(len(envs))], dtype=object)
                    batch.non_tensor_batch['success_env'] = np.array([0 for _ in range(len(envs))], dtype=object)
                    batch.non_tensor_batch['traj_length'] = np.array([0 for _ in range(len(envs))], dtype=object)
                    batch.non_tensor_batch['valid_action'] = np.array([0 for _ in range(len(envs))], dtype=object)
                    batch.non_tensor_batch['effective_action'] = np.array([0 for _ in range(len(envs))], dtype=object)
                    batch.non_tensor_batch['effective_action_ratio'] = np.array([0 for _ in range(len(envs))], dtype=object)
                    for idx, env in enumerate(envs):
                        batch.non_tensor_batch['finished_env'][idx] = int(env.finished())
                        batch.non_tensor_batch['success_env'][idx] = int(env.success())
                        tracking_vars = env.get_tracking_variables()
                        batch.non_tensor_batch['traj_length'][idx] = len(tracking_vars['actions'])
                        batch.non_tensor_batch['valid_action'][idx] = sum(1 for x in tracking_vars['actions_valid'] if x is not None)
                        batch.non_tensor_batch['effective_action'][idx] = sum(1 for x in tracking_vars['actions_effective'] if x is not None)
                        batch.non_tensor_batch['effective_action_ratio'][idx] = sum(1 for x in tracking_vars['actions_effective'] if x is not None) / len(tracking_vars['actions'])

                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(final_gen_batch_output)

                    ####################
                    ####################


                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

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
                        batch.batch['token_level_scores'] = reward_tensor

                        # compute rewards. apply_kl_penalty if available, no kl_loss or kl_penalty for GAE
                        if not self.config.actor_rollout_ref.actor.get('use_kl_loss', False) or self.config.algorithm.get('adv_estimator') != 'gae':
                            batch, kl_metrics = apply_kl_penalty(batch,
                                                                 kl_ctrl=self.kl_ctrl,
                                                                 kl_penalty=self.config.algorithm.get('kl_penalty', 'kl'))
                            metrics.update(kl_metrics)
                        else:
                            batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

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
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.get('critic_warmup') is not None and self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer('update_actor', timing_raw):
                            if self.config.actor_rollout_ref.actor.get('state_masking', False):
                                batch,metrics = self._create_loss_mask(batch, metrics)
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
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
                logger.log(data=metrics, step=self.global_steps)

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
        response_length = batch.batch['responses'].shape[-1]
        response_mask = batch.batch['attention_mask'][:, -response_length:]

        # Initialize action mask (all tokens initially considered as actions)
        action_mask = torch.ones_like(response_mask)
        
        responses = [self.tokenizer.decode(resp, skip_special_tokens=False) for resp in batch.batch['responses']]

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
        batch.batch['loss_mask'] = loss_mask
        
        # Debug print
        print("\nRaw batch[0] (before masking):\n", self.tokenizer.decode(batch.batch['responses'][0]))
        response_ids = batch.batch['responses'][0]
        
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
            logger = self.logger,
            is_validation = True,
        )

        envs = [self.val_env.copy() for _ in range(self.config.data.val_batch_size * self.config.actor_rollout_ref.rollout.n_agent)]
        val_global_steps = 1

        for batch_dict in self.val_dataloader:
            timing_raw = {}
            test_batch: DataProto = DataProto.from_single_dict(batch_dict)
            test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n_agent, interleave=True)

            env_seeds = [i['index'] for i in test_batch.non_tensor_batch['extra_info']]
            print("env_seeds:", env_seeds)
            for env, seed in zip(envs, env_seeds):
                env.reset(seed=seed)
            
            test_gen_batch = test_batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
            test_gen_batch.meta_info = {
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': False,
                'validate': True,
            }
            with _timer('step', timing_raw):
                first_input_ids = test_gen_batch.batch['input_ids'][:, -gen_config.max_start_length:].clone()
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

                test_batch.non_tensor_batch['reward'] = np.array([0 for _ in range(len(envs))], dtype=object)
                for idx, env in enumerate(envs):
                    test_batch.non_tensor_batch['reward'][idx] = env.reward

                if test_batch.non_tensor_batch['data_source'][0] == 'two_armed_bandit':
                    # metric for two-armed bandit
                    # NOTE here we assume invalid action is 0, low arm is 1, high arm is 2
                    test_batch.non_tensor_batch['bandit_metrics'] = np.array([0 for _ in range(len(envs))], dtype=object)
                    for idx, env in enumerate(envs):
                        test_batch.non_tensor_batch['bandit_metrics'][idx] = env.get_last_action()
                metrics['bandit_metrics'].append(test_batch.non_tensor_batch['bandit_metrics'])
                
                test_batch.non_tensor_batch['total_env'] = np.array([1 for _ in range(len(envs))], dtype=object)
                test_batch.non_tensor_batch['finished_env'] = np.array([0 for _ in range(len(envs))], dtype=object)
                test_batch.non_tensor_batch['success_env'] = np.array([0 for _ in range(len(envs))], dtype=object)
                test_batch.non_tensor_batch['traj_length'] = np.array([0 for _ in range(len(envs))], dtype=object)
                test_batch.non_tensor_batch['valid_action'] = np.array([0 for _ in range(len(envs))], dtype=object)
                test_batch.non_tensor_batch['effective_action'] = np.array([0 for _ in range(len(envs))], dtype=object)
                test_batch.non_tensor_batch['effective_action_ratio'] = np.array([0 for _ in range(len(envs))], dtype=object)
                for idx, env in enumerate(envs):
                    test_batch.non_tensor_batch['finished_env'][idx] = int(env.finished())
                    test_batch.non_tensor_batch['success_env'][idx] = int(env.success())
                    tracking_vars = env.get_tracking_variables()
                    test_batch.non_tensor_batch['traj_length'][idx] = len(tracking_vars['actions'])
                    test_batch.non_tensor_batch['valid_action'][idx] = sum(1 for x in tracking_vars['actions_valid'] if x is not None)
                    test_batch.non_tensor_batch['effective_action'][idx] = sum(1 for x in tracking_vars['actions_effective'] if x is not None)
                    test_batch.non_tensor_batch['effective_action_ratio'][idx] = sum(1 for x in tracking_vars['actions_effective'] if x is not None) / len(tracking_vars['actions'])

                # Accumulate batch metrics into global storage
                global_token_scores.append(test_batch.non_tensor_batch['reward'])

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
        self.logger.log(data=global_metrics, step=self.val_num)
        return global_metrics

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix='global_seqlen'):
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
        if 'loss_mask' in batch.batch:
            device = batch.batch['responses'].device
            dtype = batch.batch['responses'].dtype
            batch.batch['loss_mask'] = batch.batch['loss_mask'].to(device=device, dtype=dtype)
            
        return batch, metrics
