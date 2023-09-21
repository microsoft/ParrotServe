from parrot.utils import RecyclePool


def test_recycle_pool():
    pool = RecyclePool(4)
    for i in range(4):
        assert pool.allocate() in [0, 1, 2, 3]

    for i in range(32):
        pool.free(i % 4)
        assert pool.allocate() in [0, 1, 2, 3]

    for i in range(4):
        pool.free(i)


def test_recycle_pool_error():
    pool = RecyclePool(1)
    pool.allocate()

    try:
        pool.allocate()
    except ValueError:
        pass

    pool.free(0)
    try:
        pool.free(0)
    except ValueError:
        pass


if __name__ == "__main__":
    test_recycle_pool()
    test_recycle_pool_error()
