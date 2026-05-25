import requests

def report_status(algo_url, status_dict):
    try:
        resp = requests.post(algo_url, json=status_dict, timeout=5)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.HTTPError as e:
        print(f"HTTP错误: {e}")
        resp = getattr(e, 'response', None)
        if resp is not None:
            try:
                error_detail = resp.json()
                print("服务器返回错误详情:", error_detail)
            except Exception:
                print("响应内容:", resp.text)
        else:
            print("无响应内容")
        return None
    except Exception as e:
        print(f"状态上报或获取决策失败: {e}")
        return None
