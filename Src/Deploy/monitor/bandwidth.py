# deploy/monitor/bandwidth.py
import subprocess
import json
import os

# 请确保这个路径指向你的 iperf3.exe（根据实际情况调整）
IPERF_EXE = r"C:\\Program Files\\iperf-3.21-win64-dynamic-auth\\iperf3.exe"
# 如果你放在了其他位置，例如 C:\iperf3\iperf3.exe，就改成那个路径。


def measure_bandwidth_iperf(server_ip, port=5001, duration=2):
    """使用 iperf3 客户端测量到指定服务端的上行带宽 (Mbps)"""
    cmd = [IPERF_EXE, "-c", server_ip, "-p",
           str(port), "-t", str(duration), "-J"]
    try:
        result = subprocess.run(cmd, capture_output=True,
                                text=True, timeout=duration + 5)
        if result.returncode == 0 and result.stdout:
            data = json.loads(result.stdout)
            # 提取发送速率 (bits_per_second) 并转换为 Mbps
            bw_bps = data['end']['sum_sent']['bits_per_second']
            return bw_bps / 1e6
        else:
            print("iperf3 测量失败，返回码:", result.returncode)
            if result.stderr:
                print("错误详情:", result.stderr)
            return 0.0
    except subprocess.TimeoutExpired:
        print("iperf3 测量超时")
        return 0.0
    except FileNotFoundError:
        print(f"找不到 iperf3 程序，请检查路径: {IPERF_EXE}")
        return 0.0
    except Exception as e:
        print(f"iperf3 测量异常: {e}")
        return 0.0
