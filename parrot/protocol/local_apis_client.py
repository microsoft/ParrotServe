import requests
from requests.exceptions import RequestException

from .local_apis_response import *


def check_heartbeat(addr: str) -> HeartbeatResponse:
    url = addr + "/heartbeat"
    timeout = 3
    try:
        resp = requests.post(url, json={}, timeout=timeout)
        assert resp.status_code == 200
        resp = make_response(HeartbeatResponse, resp)
        return resp
    except (RequestException, KeyError, ValueError, TypeError, AssertionError) as e:
        # forward to caller side
        raise e
