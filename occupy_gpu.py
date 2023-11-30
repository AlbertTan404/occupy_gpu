import torch
import time
import subprocess
import multiprocessing


def get_gpu_utilization(gpu_id):
    _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]

    COMMAND = "nvidia-smi --query-gpu=utilization.gpu --format=csv"
    gpu_util_info = _output_to_list(subprocess.check_output(COMMAND.split()))[1:]
    gpu_util_value = int(gpu_util_info[gpu_id].split()[0])
    
    return gpu_util_value
 

def matrix_multiplication(gpu_id, size, interval, gpu_util_range=(20, 50)):
    a = torch.rand(size, size, device=gpu_id)
    b = torch.rand(size, size, device=gpu_id)

    while True:
        # adjust the computations
        c = a * b
        c = a * b * c * c * b * a
        gpu_util = get_gpu_utilization(gpu_id)
        if gpu_util > max(gpu_util_range):  # if you are running other gpu programs, slow the script down
            interval = min(10, interval * 2)
            # print(f'id-{gpu_id}: {interval}')
        elif gpu_util < min(gpu_util_range):  # if low util
            interval = max(0.0001, interval / 2)
            # print(f'id-{gpu_id}: {interval}')
        time.sleep(interval)


if __name__ == "__main__":
    size = 1024 * 16  # matrix size for computation, larger size occupies more GPU memory
    interval = 0.01  # seconds
    num_gpus = torch.cuda.device_count()

    # matrix_multiplication(0, size, interval)

    processes = []
    for gpu_id in range(num_gpus):
        p = multiprocessing.Process(target=matrix_multiplication, args=(gpu_id, size, interval))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
