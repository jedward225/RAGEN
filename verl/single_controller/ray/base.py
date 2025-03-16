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

import time
import ray
import traceback
import socket
import os
from typing import Any, List, Union, Dict, Optional, Tuple, TYPE_CHECKING
from ray.util import list_named_actors
from ray.util.placement_group import placement_group, PlacementGroup, remove_placement_group
from time import sleep
from ray import tune
from ray.runtime_context import get_runtime_context
from ray.experimental.state.api import get_actor
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy, NodeAffinitySchedulingStrategy
from unittest.mock import patch

# First import base classes 
from verl.single_controller.base.worker_group import ResourcePool, ClassWithInitArgs, WorkerGroup
from verl.single_controller.base.worker import Worker
from verl.single_controller.base.decorator import Dispatch, MAGIC_ATTR, dispatch_one_to_all, dispatch_all_to_all, collect_all_to_all

__all__ = ['Worker', 'RayResourcePool', 'RayClassWithInitArgs', 'RayWorkerGroup', 'create_colocated_worker_cls']


def _default_generate_function(worker_group, method_name, dispatch_mode):
    """Default function generator for worker methods"""
    # Define dispatch function based on mode
    if dispatch_mode == Dispatch.RANK_ZERO:
        def dispatch_fn(self, *args, **kwargs):
            return args, kwargs
    elif dispatch_mode == Dispatch.ONE_TO_ALL:
        def dispatch_fn(self, *args, **kwargs):
            return dispatch_one_to_all(self, *args, **kwargs)
    elif dispatch_mode == Dispatch.ALL_TO_ALL:
        def dispatch_fn(self, *args, **kwargs):
            return dispatch_all_to_all(self, *args, **kwargs)
    else:
        raise ValueError(f"Unsupported dispatch mode: {dispatch_mode}")

    # Define execute function based on mode
    if dispatch_mode == Dispatch.RANK_ZERO:
        def execute_fn(method_name, *args, **kwargs):
            return worker_group.execute_rank_zero(method_name, *args, **kwargs)
    else:
        def execute_fn(method_name, *args, **kwargs):
            return worker_group.execute_all(method_name, *args, **kwargs)

    # Define collect function based on mode
    if dispatch_mode == Dispatch.RANK_ZERO:
        def collect_fn(output):
            return output
    else:
        def collect_fn(output):
            return collect_all_to_all(worker_group, output)

    def func(*args, **kwargs):
        args, kwargs = dispatch_fn(worker_group, *args, **kwargs) 
        output = execute_fn(method_name, *args, **kwargs)
        if isinstance(output, ray.ObjectRef) or isinstance(output, list) and all(isinstance(o, ray.ObjectRef) for o in output):
            output = ray.get(output)
        return collect_fn(output)

    return func


def find_free_port():
    """Find a free port on the current node."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def get_random_string(length: int) -> str:
    import random
    import string
    letters_digits = string.ascii_letters + string.digits
    return ''.join(random.choice(letters_digits) for _ in range(length))


class RayResourcePool(ResourcePool):
    """
    Mimic multi-processing resource, but for ray. Maps to a ray placement group
    """

    def __init__(self,
                 process_on_nodes: List[int] = None,
                 use_gpu: bool = True,
                 name_prefix: str = "",
                 max_collocate_count: int = 5,
                 detached=False) -> None:
        super().__init__(process_on_nodes, max_collocate_count)
        self.use_gpu = use_gpu
        # print(f"in RayProcessDispatchConfiguration: name_prefix = {name_prefix}")
        self.name_prefix = name_prefix
        self.pgs = None
        self.detached = detached

    def get_placement_groups(self, strategy="STRICT_PACK", name=None):
        if self.pgs is not None:
            return self.pgs

        pg_name_prefix = name if name else \
            f"{self.name_prefix}verl_group_{'_'.join([str(count) for count in self._store])}:"
        # print(f"pg_name_prefix = {pg_name_prefix}")
        pg_scheme = [[{
            "CPU": self.max_collocate_count,
            "GPU": 1
        } if self.use_gpu else {
            "CPU": self.max_collocate_count
        } for _ in range(process_count)] for process_count in self._store]

        lifetime = 'detached' if self.detached else None

        pgs = [
            placement_group(bundles=bundles, strategy=strategy, name=pg_name_prefix + str(idx), lifetime=lifetime)
            for idx, bundles in enumerate(pg_scheme)
        ]

        ray.get([pg.ready() for pg in pgs])

        self.pgs = pgs
        return pgs


def merge_resource_pool(rp1: RayResourcePool, rp2: RayResourcePool) -> RayResourcePool:
    assert rp1.use_gpu == rp2.use_gpu, 'Both RayResourcePool must either use_gpu or not'
    assert rp1.max_collocate_count == rp2.max_collocate_count, 'Both RayResourcePool must has the same max_collocate_count'
    assert rp1.n_gpus_per_node == rp2.n_gpus_per_node, 'Both RayResourcePool must has the same n_gpus_per_node'
    assert rp1.detached == rp2.detached, 'Detached ResourcePool cannot be merged with non-detached ResourcePool'

    new_store = rp1.store + rp2.store

    merged = RayResourcePool(new_store, rp1.use_gpu, f"{rp1.name_prefix}_{rp2.name_prefix}")
    merged.pgs = rp1.get_placement_groups() + rp2.get_placement_groups()

    return merged


def extract_pg_from_exist(resource_pools: Dict[str, RayResourcePool], src_role_names: List[str],
                          resource_pool: RayResourcePool) -> List:

    src_pgs = [
        pg for role_name, resource_pool in resource_pools.items() for pg in resource_pool.get_placement_groups()
        if role_name in src_role_names
    ]

    sorted_src_pgs = sorted(src_pgs, key=lambda pg: pg.bundle_count, reverse=True)
    sorted_process_on_nodes = sorted([(val, idx) for idx, val in enumerate(resource_pool.store)], reverse=True)

    unsorted_pgs: List[Tuple[int, PlacementGroup]] = []
    searching_idx = 0
    for request_process, original_idx in sorted_process_on_nodes:
        assert searching_idx < len(sorted_src_pgs), f"no enough nodes for request: searching {searching_idx} th node"
        assert request_process <= sorted_src_pgs[searching_idx].bundle_count, \
            f"requesting {request_process} processes, bundle count cannot satisfy"
        unsorted_pgs.append((original_idx, sorted_src_pgs[searching_idx]))
        searching_idx += 1

    return [pg for _, pg in sorted(unsorted_pgs)]


class RayClassWithInitArgs(ClassWithInitArgs):

    def __init__(self, cls, *args, **kwargs) -> None:
        # self._options = kwargs.pop('options', dict())
        super().__init__(cls, *args, **kwargs)
        self._options = {}
        self._additional_resource = {}

    def set_additional_resource(self, additional_resource):
        self._additional_resource = additional_resource

    def update_options(self, options: Dict):
        self._options.update(options)

    def __call__(self,
                 placement_group,
                 placement_group_bundle_idx,
                 use_gpu: bool = True,
                 num_gpus=1,
                 sharing_with=None) -> Any:
        if sharing_with is not None:
            target_node_id = ray.get(sharing_with.get_node_id.remote())
            cuda_visible_devices = ray.get(sharing_with.get_cuda_visible_devices.remote())
            options = {"scheduling_strategy": NodeAffinitySchedulingStrategy(node_id=target_node_id, soft=False)}
            return self.cls.options(**options).remote(*self.args,
                                                      cuda_visible_devices=cuda_visible_devices,
                                                      **self.kwargs)

        options = {
            "scheduling_strategy":
                PlacementGroupSchedulingStrategy(placement_group=placement_group,
                                                 placement_group_bundle_index=placement_group_bundle_idx)
        }
        options.update(self._options)

        if use_gpu:
            options["num_gpus"] = num_gpus

        if len(self._additional_resource) > 1:
            for k, v in self._additional_resource.items():
                options[k] = v

        # print("cls:", self.cls)
        # print("args: ", self.args)
        # print("kwargs: ", self.kwargs)
        return self.cls.options(**options).remote(*self.args, **self.kwargs)


class RayWorkerGroup(WorkerGroup):

    def __init__(self,
                 resource_pool: RayResourcePool = None,
                 ray_cls_with_init: RayClassWithInitArgs = None,
                 bin_pack: bool = True,
                 name_prefix: str = None,
                 detached=False,
                 worker_names=None,
                 **kwargs) -> None:
        super().__init__(resource_pool=resource_pool, **kwargs)
        self.ray_cls_with_init = ray_cls_with_init
        self.name_prefix = get_random_string(length=6) if name_prefix is None else name_prefix

        if worker_names is not None:
            assert self._is_init_with_detached_workers
            self._worker_names = worker_names

        if self._is_init_with_detached_workers:
            self._init_with_detached_workers(worker_names=worker_names)
        else:
            self._init_with_resource_pool(resource_pool=resource_pool,
                                          ray_cls_with_init=ray_cls_with_init,
                                          bin_pack=bin_pack,
                                          detached=detached)

        if ray_cls_with_init is not None:
            self._bind_worker_method(self.ray_cls_with_init.cls, None)

    def _is_worker_alive(self, worker: ray.actor.ActorHandle):
        worker_state_dict = get_actor(worker._actor_id.hex())
        return worker_state_dict.get("state", "undefined") == "ALIVE" if worker_state_dict is not None else False

    def _init_with_detached_workers(self, worker_names):
        workers = [ray.get_actor(name=name) for name in worker_names]
        self._workers = workers
        self._world_size = len(worker_names)

    def _init_with_resource_pool(self, resource_pool, ray_cls_with_init, bin_pack, detached):
        use_gpu = resource_pool.use_gpu

        strategy = "PACK"
        if bin_pack:
            strategy = "STRICT_PACK"
        pgs = resource_pool.get_placement_groups(strategy=strategy)
        world_size = resource_pool.world_size
        self._world_size = world_size
        # cia.add_kwarg("_world_size", world_size)
        num_gpus = 1 / resource_pool.max_collocate_count

        rank = -1
        for pg_idx, local_world_size in enumerate(resource_pool.store):
            pg = pgs[pg_idx]
            assert local_world_size <= pg.bundle_count, \
                f"when generating for {self.name_prefix}, for the "
            for local_rank in range(local_world_size):
                rank += 1

                # we pass in environment variable at option so that Worker can use environment variable to set
                env_vars = {
                    'WORLD_SIZE': str(world_size),
                    'RANK': str(rank),
                    'WG_PREFIX': self.name_prefix,
                    'WG_BACKEND': 'ray',
                    'RAY_LOCAL_WORLD_SIZE': str(local_world_size),
                    'RAY_LOCAL_RANK': str(local_rank),
                }
                if rank != 0:
                    env_vars['MASTER_ADDR'] = self._master_addr
                    env_vars['MASTER_PORT'] = self._master_port

                import re
                cia_name = type(ray_cls_with_init.cls).__name__
                match = re.search(r"ActorClass\(([^)]+)\)", cia_name)  # ray.remote(Obj) -> "ActorClass(Obj)"
                cia_name = match.group(1) if match else cia_name  # "ActorClass(Obj)" -> "Obj"
                name = f"{self.name_prefix}{cia_name}_{pg_idx}:{local_rank}"  # e.g. Worker_2:5

                ray_cls_with_init.update_options({'runtime_env': {'env_vars': env_vars}, 'name': name})

                if detached:
                    ray_cls_with_init.update_options({'lifetime': 'detached'})

                # create a worker
                worker = ray_cls_with_init(placement_group=pg,
                                           placement_group_bundle_idx=local_rank,
                                           use_gpu=use_gpu,
                                           num_gpus=num_gpus)
                self._workers.append(worker)
                self._worker_names.append(name)

                if rank == 0:
                    # Create the register center for the first worker in the group
                    # This needs to be done before other workers try to connect
                    # Get master address and port info
                    master_addr = ray.util.get_node_ip_address()
                    master_port = str(find_free_port())
                    rank_zero_info = {
                        "MASTER_ADDR": master_addr,
                        "MASTER_PORT": master_port,
                    }

                    # Import here to avoid circular imports
                    from verl.single_controller.base.register_center.ray import create_worker_group_register_center
                    
                    # Create the register center actor
                    register_center_name = f"{self.name_prefix}_register_center"
                    register_center = create_worker_group_register_center(
                        name=register_center_name,
                        info=rank_zero_info
                    )
                    print(f"[INFO] Created register center actor: {register_center_name}")
                    
                    # Store master info for later workers
                    self._master_addr = master_addr
                    self._master_port = master_port
                    
                else:
                    # For non-rank-0 workers, wait for the register center to be available
                    register_center_actor = None
                    for _ in range(120):
                        if f"{self.name_prefix}_register_center" not in list_named_actors():
                            time.sleep(1)
                            print(f"[INFO] Waiting for register center actor: {self.name_prefix}_register_center")
                        else:
                            register_center_actor = ray.get_actor(f"{self.name_prefix}_register_center")
                            break
                    
                    # If we failed to find the register center after waiting, we still need the master info
                    # from another source, so we'll use the values from the first worker in the group
                    if register_center_actor is None:
                        print(f"[WARNING] Failed to get register_center_actor: {self.name_prefix}_register_center in {list_named_actors(all_namespaces=True)}")
                        # If we already have master info from rank 0, use it
                        if hasattr(self, '_master_addr') and hasattr(self, '_master_port'):
                            print(f"[INFO] Using previously stored master info: {self._master_addr}:{self._master_port}")
                        else:
                            # Otherwise, try to get the info directly from the first worker
                            if len(self._workers) > 0:
                                first_worker = self._workers[0]
                                try:
                                    # Assume the first worker has the master info
                                    env_vars = ray.get(first_worker.get_env_vars.remote())
                                    self._master_addr = env_vars.get('MASTER_ADDR')
                                    self._master_port = env_vars.get('MASTER_PORT')
                                    print(f"[INFO] Retrieved master info from first worker: {self._master_addr}:{self._master_port}")
                                except Exception as e:
                                    print(f"[ERROR] Failed to get env vars from first worker: {e}")
                                    # Last resort, use the current node's IP and find a free port
                                    self._master_addr = ray.util.get_node_ip_address()
                                    self._master_port = str(find_free_port())
                                    print(f"[INFO] Using fallback master info: {self._master_addr}:{self._master_port}")
                    else:
                        # Successfully found the register center, get the rank zero info
                        try:
                            rank_zero_info = ray.get(register_center_actor.get_rank_zero_info.remote())
                            self._master_addr, self._master_port = rank_zero_info['MASTER_ADDR'], rank_zero_info['MASTER_PORT']
                            print(f"[INFO] Got rank zero info from register center: {self._master_addr}:{self._master_port}")
                        except Exception as e:
                            print(f"[ERROR] Failed to get rank zero info from register center: {e}")
                            # Fallback to using current node's IP and a free port
                            self._master_addr = ray.util.get_node_ip_address()
                            self._master_port = str(find_free_port())
                            print(f"[INFO] Using fallback master info: {self._master_addr}:{self._master_port}")

    @property
    def worker_names(self):
        return self._worker_names

    def spawn(self, prefix_set):
        """
        spawn to a dictionary of worker groups, each with a subset of method with prefix.

        """

        def _rebind_actor_methods(worker_group, actor_name):
            """
            bind the method with actor_prefix to its original name
            """
            prefix: str = actor_name + '_'
            for method_name in dir(worker_group):
                if method_name.startswith(prefix):
                    # only valid when Python >= 3.9
                    original_method_name = method_name.removeprefix(prefix)
                    method = getattr(worker_group, method_name)
                    setattr(worker_group, original_method_name, method)

        new_worker_group_dict = {}
        for prefix in prefix_set:
            new_worker_group = self.from_detached(worker_names=self._worker_names,
                                                  ray_cls_with_init=self.ray_cls_with_init)

            _rebind_actor_methods(new_worker_group, prefix)
            new_worker_group_dict[prefix] = new_worker_group
        return new_worker_group_dict

    def execute_rank_zero_sync(self, method_name: str, *args, **kwargs):
        return ray.get(self.execute_all_async(method_name, **args, **kwargs))

    def execute_rank_zero_async(self, method_name: str, *args, **kwargs):
        remote_call = getattr(self._workers[0], method_name)
        return remote_call.remote(*args, **kwargs)

    def execute_rank_zero(self, method_name: str, *args, **kwargs):
        return self.execute_rank_zero_async(method_name, *args, **kwargs)

    def execute_all(self, method_name: str, *args, **kwargs):
        return self.execute_all_async(method_name, *args, **kwargs)

    def execute_all_sync(self, method_name: str, *args, **kwargs):
        return ray.get(self.execute_all_async(method_name, *args, **kwargs))

    def execute_all_async(self, method_name: str, *args, **kwargs):
        # 这里我们假设，如果 args 和 kwargs 里面所有的参数都是 list，且所有的 list 长度都与 len(self._workers) 一致的话，我们会把
        # list 中的每一个分别发到对应的 worker 上去
        # print(f"execute_all_async: method {method_name}({args}, {kwargs})")
        length = len(self._workers)
        if all(isinstance(arg, list) for arg in args) and all(isinstance(kwarg, list) for kwarg in kwargs.values()):
            if all(len(arg) == length for arg in args) and all(len(kwarg) == length for kwarg in kwargs.values()):
                # print(f"splitting args and kwargs into {length} shards")
                result = []
                for i in range(length):
                    sliced_args = tuple(arg[i] for arg in args)
                    sliced_kwargs = {k: v[i] for k, v in kwargs.items()}
                    remote_call = getattr(self._workers[i], method_name)
                    result.append(remote_call.remote(*sliced_args, **sliced_kwargs))
                return result

        return [getattr(worker, method_name).remote(*args, **kwargs) for worker in self._workers]

    @property
    def master_address(self):
        return self._master_addr

    @property
    def master_port(self):
        return self._master_port

    @property
    def workers(self):
        return self._workers

    @property
    def world_size(self):
        return self._world_size

    def _bind_worker_method(self, user_defined_cls, func_generator):
        for method_name in dir(user_defined_cls):
            method = getattr(user_defined_cls, method_name)

            if hasattr(method, MAGIC_ATTR):
                dispatch_mode = getattr(method, MAGIC_ATTR)

                # Always use _default_generate_function
                func = _default_generate_function(self, method_name, dispatch_mode)
                
                # pass MAGIC_ATTR for outer worker group
                setattr(func, MAGIC_ATTR, dispatch_mode)
                setattr(self.__class__, method_name, func)


"""
Utilities that enables creating workers inside the same ray.Actor, 
with code written in separate ray.Actors.
"""

# Import here instead of the top to avoid circular dependencies
from verl.single_controller.base.decorator import MAGIC_ATTR


def _bind_workers_method_to_parent(cls, key, user_defined_cls):
    """
    Binds the methods of each worker to the WorkerDict. 
    Note that we only bind public methods that are decorated by register
    """
    for method_name in dir(user_defined_cls):
        try:
            method = getattr(user_defined_cls, method_name)
            assert callable(method), f"{method_name} in {user_defined_cls} is not callable"
        except Exception as e:
            # if it is a property, it will fail because Class doesn't have instance property
            continue

        if hasattr(method, MAGIC_ATTR):

            def generate_function(name):

                def func(self, *args, **kwargs):
                    # dispatch to the actual worker
                    return getattr(self.worker_dict[key], name)(*args, **kwargs)

                return func

            func = generate_function(method_name)
            # pass MAGIC_ATTR for outer worker group
            setattr(func, MAGIC_ATTR, getattr(method, MAGIC_ATTR))
            try:
                method_name_with_prefix = key + '_' + method_name
                setattr(cls, method_name_with_prefix, func)
                # print(f'Binding {method_name_with_prefix}')
            except Exception as e:
                raise ValueError(f'Fail to set method_name {method_name}')


def _unwrap_ray_remote(cls):
    if hasattr(cls, '__ray_actor_class__'):
        cls = cls.__ray_actor_class__
    return cls


def create_colocated_worker_cls(class_dict: dict[str, RayClassWithInitArgs]):
    """
    This function should return a class instance that delegates the calls to every 
    cls in cls_dict
    """
    cls_dict = {}
    init_args_dict = {}
    worker_cls = None
    for key, cls in class_dict.items():
        if worker_cls == None:
            worker_cls = cls.cls.__ray_actor_class__.__base__
        else:
            assert worker_cls == cls.cls.__ray_actor_class__.__base__, \
                'the worker class should be the same when share the same process'
        cls_dict[key] = cls.cls
        init_args_dict[key] = {'args': cls.args, 'kwargs': cls.kwargs}

    assert cls_dict.keys() == init_args_dict.keys()

    # TODO: create a class with customizable name
    class WorkerDict(worker_cls):

        def __init__(self):
            super().__init__()
            self.worker_dict = {}
            for key, user_defined_cls in cls_dict.items():
                user_defined_cls = _unwrap_ray_remote(user_defined_cls)
                # directly instantiate the class without remote
                with patch.dict(os.environ, {'DISABLE_WORKER_INIT': '1'}):
                    self.worker_dict[key] = user_defined_cls(*init_args_dict[key].get('args', ()),
                                                             **init_args_dict[key].get('kwargs', {}))

    # now monkey-patch the methods from inner class to WorkerDict
    for key, user_defined_cls in cls_dict.items():
        user_defined_cls = _unwrap_ray_remote(user_defined_cls)
        _bind_workers_method_to_parent(WorkerDict, key, user_defined_cls)

    remote_cls = ray.remote(WorkerDict)
    remote_cls = RayClassWithInitArgs(remote_cls)
    return remote_cls
