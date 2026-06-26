import psutil


def get_cpu_available_cores():
    """返回当前可用CPU核心数（粗略估算）"""
    usage = psutil.cpu_percent(interval=0.1)
    total = psutil.cpu_count(logical=True)
    if total is None:
        return 0
    return total * (1 - usage / 100.0)


def get_memory_percent():
    return psutil.virtual_memory().percent
