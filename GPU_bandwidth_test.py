import torch
import time

def test_gpu_bandwidth(src_device, dst_device, size_gb=1, num_runs=10):
    """
    测试两个GPU之间的数据传输带宽
    参数：
        src_device: 源GPU设备ID
        dst_device: 目标GPU设备ID
        size_gb: 测试数据大小（GB）
        num_runs: 测试次数
    返回：
        平均带宽（GB/s）
    """
    # 计算张量元素数量（float32占4字节）
    num_elements = int(size_gb * (1024 ** 3) // 4)
    
    try:
        # 在源设备创建随机张量
        tensor_src = torch.randn(num_elements, device=f'cuda:{src_device}', dtype=torch.float32)
        # 在目标设备创建空张量
        tensor_dst = torch.empty_like(tensor_src, device=f'cuda:{dst_device}')
    except RuntimeError as e:
        print(f"显存分配失败，请尝试减小测试数据大小（当前大小：{size_gb}GB）")
        raise e

    # 预热：执行一次无效传输
    for _ in range(3):
        tensor_dst.copy_(tensor_src)
    torch.cuda.synchronize()

    # 正式测试
    start_time = time.time()
    for _ in range(num_runs):
        tensor_dst.copy_(tensor_src)
        torch.cuda.synchronize()  # 确保每次复制操作完成
    end_time = time.time()

    # 计算带宽
    total_data = size_gb * num_runs  # 总传输数据量（GB）
    elapsed = end_time - start_time  # 总耗时（秒）
    bandwidth = total_data / elapsed

    return bandwidth

if __name__ == "__main__":
    # 检查可用GPU数量
    if torch.cuda.device_count() < 2:
        print("需要至少两个GPU设备")
        exit(1)

    # 测试参数配置
    TEST_SIZE_GB = 1    # 测试数据大小（GB）
    NUM_RUNS = 10       # 测试次数
    DEVICE_A = 0        # 第一个GPU ID
    DEVICE_B = 1        # 第二个GPU ID

    try:
        # 测试A到B的带宽
        print(f"测试 GPU{DEVICE_A} → GPU{DEVICE_B} 带宽...")
        bw_ab = test_gpu_bandwidth(DEVICE_A, DEVICE_B, TEST_SIZE_GB, NUM_RUNS)
        print(f"带宽：{bw_ab:.2f} GB/s\n")

        # 测试B到A的带宽
        print(f"测试 GPU{DEVICE_B} → GPU{DEVICE_A} 带宽...")
        bw_ba = test_gpu_bandwidth(DEVICE_B, DEVICE_A, TEST_SIZE_GB, NUM_RUNS)
        print(f"带宽：{bw_ba:.2f} GB/s")

        # 打印最终结果
        print("\n最终结果：")
        print(f"GPU{DEVICE_A} → GPU{DEVICE_B}: {bw_ab:.2f} GB/s")
        print(f"GPU{DEVICE_B} → GPU{DEVICE_A}: {bw_ba:.2f} GB/s")
        
    except Exception as e:
        print(f"测试失败：{str(e)}")