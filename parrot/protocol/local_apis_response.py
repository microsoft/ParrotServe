import dataclasses
from dataclasses import dataclass
from requests import Response


@dataclass
class HeartbeatResponse:
    model_ready: bool
    cached_tokens: int
    running_jobs: int


def make_response(resp_cls, resp: Response):
    resp_data = resp.json()
    init_data = [
        (field.name, resp_data[field.name]) for field in dataclasses.fields(resp_cls)
    ]
    return resp_cls(**dict(init_data))
