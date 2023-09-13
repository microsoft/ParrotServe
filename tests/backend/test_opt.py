from parrot.backend.runner import Runner
from parrot.backend.entity import FillJob, GenerationJob
import numpy as np


def test_single_fill():
    runner = Runner("facebook/opt-125m")

    job = FillJob(
        session_id=0,
        context_id=0,
        parent_context_id=-1,
        tokens_id=np.random.randint(50, 50000, size=10).tolist(),
    )

    runner.run([job])


if __name__ == "__main__":
    test_single_fill()
