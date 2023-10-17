# Constants for Parrot

# ---------- Interval ----------
HEARTBEAT_INTERVAL = 5  # seconds
ENGINE_LOOP_INTERVAL = 0.01

# NOTE(chaofan): HEARTBEAT_INTERVAL + GC_INTERVAL < CONTEXT_EXPIRE_TIME
GC_INTERVAL = 10  # seconds
CONTEXT_EXPIRE_TIME = 25  # seconds

# ---------- Chunk Related ----------
FILL_NO_CHUNK = -1
PIPELINE_SEND_CHUNK_NUM = 128
DETOKENIZE_CHUNK_NUM = 8
STREAMING_END_TOKEN_ID = -1


# ---------- Recycle Pool ----------
PROCESS_POOL_SIZE = 4096
THREAD_POOL_SIZE = 4096
CONTEXT_POOL_SIZE = 4096

# ---------- None ID ----------
NONE_THREAD_ID = -1
NONE_CONTEXT_ID = -1
