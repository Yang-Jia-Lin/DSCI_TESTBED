import requests
import time

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


class RoundClient:
    """HTTP client for fixed-size v2 scheduling rounds."""

    def __init__(self, base_url: str, round_id: str, user_id: int):
        self.base_url = base_url.rstrip("/")
        self.round_id = str(round_id)
        self.user_id = int(user_id)

    def _url(self, suffix: str) -> str:
        return f"{self.base_url}/api/v2/rounds/{self.round_id}/{suffix}"

    def register(self, payload: dict) -> dict:
        response = requests.post(
            self._url("devices/register"), json=payload, timeout=15
        )
        response.raise_for_status()
        return response.json()

    def heartbeat(self) -> dict:
        response = requests.post(
            self._url(f"devices/{self.user_id}/heartbeat"), timeout=10
        )
        response.raise_for_status()
        return response.json()

    def wait_for_decision(
        self, *, poll_interval_s: float = 1.0, timeout_s: float = 90.0
    ) -> dict:
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            response = requests.get(
                self._url(f"decisions/{self.user_id}"), timeout=10
            )
            if response.status_code == 200:
                return response.json()
            if response.status_code != 202:
                response.raise_for_status()
            time.sleep(float(poll_interval_s))
        raise TimeoutError(f"Timed out waiting for round {self.round_id!r} decision")

    def submit_measurements(self, payload: dict) -> dict:
        response = requests.post(
            self._url(f"measurements/{self.user_id}"), json=payload, timeout=30
        )
        response.raise_for_status()
        return response.json()
