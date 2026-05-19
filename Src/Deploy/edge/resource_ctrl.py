# 模拟算力控制，实际可使用 cgroups 或线程池限制
def get_max_cpu():
    return 4.0  # 边缘可用 CPU 核心数
