# This is a policy simulator to discover the best policy for Sharrot

from typing import List, Tuple, Generator
import asyncio
import numpy as np


cur_time = 0  # ms
total_mem = 4096 * 4  # tokens


# ---------- Requests ----------


class Request:
    def __init__(self, request_no: int) -> None:
        self.request_no = request_no
        self.prefix_type = 0
        self.output_len = 0
        self.cur_len = 0
        self.total_latency = 0
        self.finish_event: asyncio.Event = asyncio.Event()
        self.arrival_time = 0
        self.filled = False
        self.swapped = False


def generate_requests(
    request_num: int,
    prefix_type_num: int,
    output_length_upper_bound: int,
    output_length_mean: int,
    request_rate: float,
) -> List[Request]:
    requests = []
    arrival_time = 0

    for i in range(request_num):
        request = Request(request_no=i + 1)

        request.prefix_type = np.random.randint(prefix_type_num)
        output_length = output_length_upper_bound + 1
        while output_length > output_length_upper_bound:
            output_length = round(np.random.exponential(scale=output_length_mean)) + 1
        request.output_len = output_length

        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue

        request.arrival_time = arrival_time

        # Sample the request interval from the exponential distribution.
        interval = np.random.exponential(1.0 / request_rate)
        arrival_time += interval

        requests.append(request)
    return requests


# ---------- Schedulers ----------


class Scheduler:
    def __init__(self, max_batch_size: int) -> None:
        self.waiting_queue: List[Request] = []
        self.running_queue: List[Request] = []
        self.max_batch_size = max_batch_size

    def add_request(self, request: Request) -> None:
        self.waiting_queue.append(request)

    def schedule(self) -> List[Request]:
        pass

    def is_empty(self) -> bool:
        return len(self.waiting_queue) == 0 and len(self.running_queue) == 0

    def finish_batch(self) -> None:
        global total_mem
        for request in self.running_queue:
            if request.cur_len >= request.output_len:
                self.running_queue.remove(request)
                request.finish_event.set()
                total_mem -= request.output_len

    def preempt(self) -> None:
        # recompute
        global total_mem
        mem_upper_bound = 4000 * 16
        while total_mem > mem_upper_bound:
            for request in self.waiting_queue[::-1]:
                if not request.filled:
                    continue
                total_mem -= request.output_len
                request.filled = False
                request.cur_len = 0


class FIFOScheduler(Scheduler):
    def schedule(self) -> List[Request]:
        while (
            len(self.running_queue) < self.max_batch_size
            and len(self.waiting_queue) > 0
        ):
            request = self.waiting_queue.pop(0)
            self.running_queue.append(request)

        return self.running_queue.copy()


class GGScheduler(Scheduler):
    def schedule(self) -> List[Request]:
        while (
            len(self.running_queue) < self.max_batch_size
            and len(self.waiting_queue) > 0
        ):
            prefix_counter = {}
            for request in self.running_queue + self.waiting_queue:
                if request.prefix_type not in prefix_counter:
                    prefix_counter[request.prefix_type] = 0
                prefix_counter[request.prefix_type] += 1
            max_prefix = max(prefix_counter, key=prefix_counter.get)

            found = False

            for request in self.waiting_queue:
                if request.prefix_type == max_prefix:
                    self.running_queue.append(request)
                    self.waiting_queue.remove(request)
                    found = True
                    break

            if not found:
                request = self.waiting_queue.pop(0)
                self.running_queue.append(request)

        return self.running_queue.copy()


class LGScheduler(Scheduler):
    def schedule(self) -> List[Request]:
        while (
            len(self.running_queue) < self.max_batch_size
            and len(self.waiting_queue) > 0
        ):
            if len(self.running_queue) == 0:
                request = self.waiting_queue.pop(0)
                self.running_queue.append(request)
                continue

            prefix_counter = {}
            for request in self.running_queue:
                if request.prefix_type not in prefix_counter:
                    prefix_counter[request.prefix_type] = 0
                prefix_counter[request.prefix_type] += 1
            max_prefix = max(prefix_counter, key=prefix_counter.get)

            found = False

            for request in self.waiting_queue:
                if request.prefix_type == max_prefix:
                    self.running_queue.append(request)
                    self.waiting_queue.remove(request)
                    found = True
                    break

            if not found:
                request = self.waiting_queue.pop(0)
                self.running_queue.append(request)

        return self.running_queue.copy()


class MGScheduler(Scheduler):
    def schedule(self) -> List[Request]:
        empty_slots = self.max_batch_size - len(self.running_queue)

        while (
            len(self.running_queue) < self.max_batch_size
            and len(self.waiting_queue) > 0
        ):
            if len(self.running_queue) == 0:
                request = self.waiting_queue.pop(0)
                self.running_queue.append(request)
                continue

            prefix_counter1 = {}
            for request in self.running_queue:
                if request.prefix_type not in prefix_counter1:
                    prefix_counter1[request.prefix_type] = 0
                prefix_counter1[request.prefix_type] += 1
            max_prefix1 = max(prefix_counter1, key=prefix_counter1.get)

            prefix_counter2 = {}
            for request in self.running_queue + self.waiting_queue:
                if request.prefix_type not in prefix_counter2:
                    prefix_counter2[request.prefix_type] = 0
                prefix_counter2[request.prefix_type] += 1
            max_prefix2 = max(prefix_counter2, key=prefix_counter2.get)

            if prefix_counter1[max_prefix1] < len(self.running_queue) // 2:
                # use GG
                max_prefix = max_prefix2
            else:
                # use LG
                max_prefix = max_prefix1

            found = False

            for request in self.waiting_queue:
                if request.prefix_type == max_prefix:
                    self.running_queue.append(request)
                    self.waiting_queue.remove(request)
                    found = True
                    break

            if not found:
                request = self.waiting_queue.pop(0)
                self.running_queue.append(request)

        return self.running_queue.copy()


class GGSwapScheduler(Scheduler):
    def preempt(self) -> float:
        # swap
        global total_mem
        mem_upper_bound = 4000 * 16
        move_out_tokens = 0
        while total_mem > mem_upper_bound:
            for request in self.waiting_queue[::-1]:
                if not request.filled:
                    continue
                total_mem -= request.output_len
                move_out_tokens += request.output_len
                request.swapped = True

        swap_time = (
            2359296 // 16 * move_out_tokens * 2 / (24 * 1024 * 1024 * 1024)
        )  # PCIe 24 GB/s

        for request in self.running_queue:
            request.total_latency += swap_time

        return

    def swap_back(self) -> float:
        global total_mem
        move_in_tokens = 0
        for request in self.running_queue:
            if request.swapped:
                total_mem += request.output_len
                request.swapped = False
                move_in_tokens += request.output_len

        swap_time = (
            2359296 // 16 * move_in_tokens * 2 / (24 * 1024 * 1024 * 1024)
        )  # PCIe 24 GB/s

        for request in self.running_queue:
            request.total_latency += swap_time

        return swap_time

    def schedule(self) -> List[Request]:
        self.waiting_queue.extend(self.running_queue)
        self.running_queue = []

        while (
            len(self.running_queue) < self.max_batch_size
            and len(self.waiting_queue) > 0
        ):
            prefix_counter = {}
            for request in self.running_queue + self.waiting_queue:
                if request.prefix_type not in prefix_counter:
                    prefix_counter[request.prefix_type] = 0
                prefix_counter[request.prefix_type] += 1
            max_prefix = max(prefix_counter, key=prefix_counter.get)

            found = False

            for request in self.waiting_queue:
                if request.prefix_type == max_prefix:
                    self.running_queue.append(request)
                    self.waiting_queue.remove(request)
                    found = True
                    break

            if not found:
                request = self.waiting_queue.pop(0)
                self.running_queue.append(request)

        return self.running_queue.copy()


class GGRecomputeScheduler(Scheduler):
    def schedule(self) -> List[Request]:
        self.waiting_queue.extend(self.running_queue)
        self.running_queue = []

        while (
            len(self.running_queue) < self.max_batch_size
            and len(self.waiting_queue) > 0
        ):
            prefix_counter = {}
            for request in self.running_queue + self.waiting_queue:
                if request.prefix_type not in prefix_counter:
                    prefix_counter[request.prefix_type] = 0
                prefix_counter[request.prefix_type] += 1
            max_prefix = max(prefix_counter, key=prefix_counter.get)

            found = False

            for request in self.waiting_queue:
                if request.prefix_type == max_prefix:
                    self.running_queue.append(request)
                    self.waiting_queue.remove(request)
                    found = True
                    break

            if not found:
                request = self.waiting_queue.pop(0)
                self.running_queue.append(request)

        return self.running_queue.copy()


# ---------- Tools ----------


def request_client(
    request_rate: float,
    scheduler: Scheduler,
    request_num: int,
    prefix_type_num: int,
    output_length_upper_bound: int,
    output_length_mean: int,
    seed: int = 0,
) -> None:
    # Init
    global cur_time
    global total_mem
    cur_time = 0
    total_mem = 0
    np.random.seed(seed)

    # Generate requests
    requests = generate_requests(
        request_num,
        prefix_type_num,
        output_length_upper_bound,
        output_length_mean,
        request_rate,
    )

    # Run
    request_index = 0
    while request_index < request_num or not scheduler.is_empty():
        if request_index < request_num:
            while requests[request_index].arrival_time <= cur_time:
                scheduler.add_request(requests[request_index])
                request_index += 1
                if request_index >= request_num:
                    break

        batch = scheduler.schedule()
        if len(batch) == 0:
            cur_time += 1  # 1ms pass
        else:
            latency = run_batch(batch)
            if isinstance(scheduler, GGSwapScheduler):
                scheduler.swap_back()
            scheduler.finish_batch()
            scheduler.preempt()
            cur_time += latency

    # Requests
    mean_latency = (
        sum([request.total_latency / request.output_len for request in requests])
        / request_num
    )
    # print(f"[Log] Mean per token latency: {mean_latency} ms", flush=True)
    print(mean_latency, ",", flush=True)


def cost_model(batch_size, bs_threshold, alpha) -> float:
    standard_per_latency = 20  # ms

    return (
        standard_per_latency
        + standard_per_latency * max(0, batch_size - bs_threshold) * alpha
    )


def run_batch(requests: List[Request]) -> float:
    prefix_counter = {}

    assert len(requests) > 0

    for request in requests:
        if request.prefix_type not in prefix_counter:
            prefix_counter[request.prefix_type] = 0
        prefix_counter[request.prefix_type] += 1

    main_prefix = max(prefix_counter, key=prefix_counter.get)
    main_prefix_counter = prefix_counter[main_prefix]
    batch_size = len(requests)

    total_latency = 0  # ms

    # For shared request
    if main_prefix_counter >= 1:
        total_latency += cost_model(main_prefix_counter, 32, 1 / 64)

    # For each remaining request
    total_latency += cost_model(batch_size - main_prefix_counter, 8, 1 / 16)

    request_nos = [request.request_no for request in requests]

    global total_mem
    # print(
    #     f"[Log] Total latency: {total_latency} ms; Run requests: {request_nos}; Main Prefix: {main_prefix}; Cur memory: {total_mem}",
    #     flush=True,
    # )

    # await asyncio.sleep(total_latency / 1000)

    for request in requests:
        request.cur_len += 1
        total_mem += 1
        request.total_latency += total_latency
        if not request.filled:
            request.filled = True

    return total_latency


def main():
    scheduler = GGSwapScheduler(max_batch_size=64)
    request_num = 1000
    seed = 0
    prefix_type_num = 4
    output_length_upper_bound = 512
    output_length_mean = 256

    for request_rate in [0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0]:
        request_client(
            request_rate,
            scheduler,
            request_num,
            prefix_type_num,
            output_length_upper_bound,
            output_length_mean,
            seed,
        )


if __name__ == "__main__":
    main()
