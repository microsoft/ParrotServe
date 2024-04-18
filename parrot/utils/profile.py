# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


import cProfile, pstats, io
import contextlib


@contextlib.contextmanager
def cprofile(profile_title: str):
    # global cprofile_stream

    pr = cProfile.Profile()
    pr.enable()

    yield

    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats(2)
    ps.print_stats()

    print(
        "\n\n\n" + f"*** {profile_title} ***" + "\n" + s.getvalue() + "\n\n\n",
        flush=True,
    )


@contextlib.contextmanager
def torch_profile(profile_title: str):
    import torch.profiler as profiler

    with profiler.profile(
        activities=[
            profiler.ProfilerActivity.CPU,
            profiler.ProfilerActivity.CUDA,
        ]
    ) as prof:
        yield

    print(
        "\n\n\n"
        + f"*** {profile_title} ***"
        + "\n"
        + prof.key_averages().table(sort_by="cuda_time_total")
        + "\n\n\n",
        flush=True,
    )
