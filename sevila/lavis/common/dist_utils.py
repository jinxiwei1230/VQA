"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import datetime
import functools
import os

import torch
import torch.distributed as dist
import timm.models.hub as timm_hub

# 配置和管理分布式训练环境
def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    根据传入的 is_master 参数来配置打印行为，以便在分布式环境中只有主进程才能打印消息，而其他进程不会打印消息。
    """
    # 引入了 Python 内置模块 builtins 的别名 __builtin__。builtins 包含了 Python 内置的函数和对象，包括 print 函数
    import builtins as __builtin__

    # 将当前的 print 函数保存在 builtin_print 变量中，以便后续可以恢复原始的打印行为。
    builtin_print = __builtin__.print

    # 新的 print 函数，接受与原始的 print 函数相同的参数 *args 和 **kwargs
    def print(*args, **kwargs):
        force = kwargs.pop("force", False)  # 在调用 print 时，如果传入了一个名为 "force" 的参数并且其值为 True，那么无论当前进程是否为主进程，都会打印消息
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def init_distributed_mode(args):
    # 检查当前环境中是否设置了一些特定的环境变量，以确定是否应该启用分布式模式
    # RANK：进程的秩（编号） WORLD_SIZE：参与分布式计算的进程总数
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True # 标记，启用了分布式模式

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"  # 指定分布式后端为nccl
    print(
        f"| distributed init (rank {args.rank}, world {args.world_size}): {args.dist_url}",
        flush=True,
    )
    # 初始化分布式训练的进程组
    torch.distributed.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
        timeout=datetime.timedelta(
            days=365
        ),  # allow auto-downloading and de-compressing
    )
    # 确保所有进程在继续执行之前都已经完成初始化
    torch.distributed.barrier()
    # 设置在分布式环境中只有主进程才能打印消息
    setup_for_distributed(args.rank == 0)


def get_dist_info():
    if torch.__version__ < "1.0":
        initialized = dist._initialized
    else:
        initialized = dist.is_initialized()
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:  # non-distributed training
        rank = 0
        world_size = 1
    return rank, world_size


def main_process(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rank, _ = get_dist_info()
        if rank == 0:
            return func(*args, **kwargs)

    return wrapper


def download_cached_file(url, check_hash=True, progress=False):
    """
    Download a file from a URL and cache it locally. If the file already exists, it is not downloaded again.
    If distributed, only the main process downloads the file, and the other processes wait for the file to be downloaded.
    """

    def get_cached_file_path():
        # a hack to sync the file path across processes
        parts = torch.hub.urlparse(url)
        filename = os.path.basename(parts.path)
        cached_file = os.path.join(timm_hub.get_cache_dir(), filename)

        return cached_file

    if is_main_process():
        timm_hub.download_cached_file(url, check_hash, progress)

    if is_dist_avail_and_initialized():
        dist.barrier()

    return get_cached_file_path()
