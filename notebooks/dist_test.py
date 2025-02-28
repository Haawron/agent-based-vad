import traceback
from datetime import timedelta

import torch
import torch.distributed as dist


def init_distributed(
    rank: int = 0,
    world_size: int = 1,
    host: str = 'localhost',
    port: int = 50001,
    backend: str = 'nccl',
):
    try:
        print(f'Initializing rank {rank} with world size {world_size}')
        dist.init_process_group(
            backend,
            init_method=f'tcp://{host}:{port}',
            # init_method='file:///code/tmp/tmpfile',
            timeout=timedelta(seconds=60),
            rank=rank,
            world_size=world_size,
            device_id=torch.device(0),
        )
        print(f'Initialized rank {dist.get_rank()} with world size {dist.get_world_size()}')
    except Exception as e:
        print(f'Failed to initialize rank {rank} with world size {world_size}')
        # print(e)
        traceback.print_exc()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    from fire import Fire
    Fire(init_distributed)


#################### TCP ####################
# node1
# docker run -e NCCL_SOCKET_IFNAME=nebula1 --network host --mount type=bind,src=/projects3/home/hglee/prjs/agent-based-vad/,dst=/code --rm -it --gpus all torch python notebooks/dist_test.py --rank 0 --world_size 2 --host '0.0.0.0'

# node2
# docker run -e NCCL_SOCKET_IFNAME=nebula1 --network host --mount type=bind,src=/projects3/home/hglee/prjs/agent-based-vad/,dst=/code --rm -it --gpus all torch python notebooks/dist_test.py --rank 1 --world_size 2 --host '10.90.21.21'



# docker run --sysctl net.ipv4.ip_local_port_range="50000 51000" --mount type=bind,src=/projects3/home/hglee/prjs/agent-based-vad/,dst=/code --rm -it --gpus all -p 50000-50100 torch python notebooks/dist_test.py --rank 0 --world_size 2 --host '0.0.0.0'
# docker run -e NCCL_DEBUG=info --mount type=bind,src=/projects3/home/hglee/prjs/agent-based-vad/,dst=/code --rm -it --gpus all torch python notebooks/dist_test.py --rank 1 --world_size 2 --host '10.90.21.21'


#################### FILE SYSTEM ####################
# node1
# docker run --mount type=bind,src=/projects3/home/hglee/prjs/agent-based-vad/,dst=/code --rm -it --gpus all --ipc host torch python notebooks/dist_test.py --rank 0 --world_size 2

# node2
# docker run --mount type=bind,src=/projects3/home/hglee/prjs/agent-based-vad/,dst=/code --rm -it --gpus all --ipc host torch python notebooks/dist_test.py --rank 1 --world_size 2
