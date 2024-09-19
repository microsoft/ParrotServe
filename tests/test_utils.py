from parrot.utils import RecyclePool, bytes_to_encoded_b64str, encoded_b64str_to_bytes


def test_recycle_pool():
    pool = RecyclePool()
    for i in range(4):
        assert pool.allocate() in [0, 1, 2, 3]

    for i in range(32):
        pool.free(i % 4)
        assert pool.allocate() in [0, 1, 2, 3]

    for i in range(4):
        pool.free(i)


def test_recycle_pool_error():
    pool = RecyclePool()
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


def test_serialize_tools():
    data = b"hello world"
    encoded = bytes_to_encoded_b64str(data)
    decoded = encoded_b64str_to_bytes(encoded)
    assert data == decoded



if __name__ == "__main__":
    test_recycle_pool()
    test_recycle_pool_error()
    test_serialize_tools()