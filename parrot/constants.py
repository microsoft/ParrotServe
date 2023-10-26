# Constants for Parrot

# NOTE(chaofan): All time constansts (with the name of *_TIME, *_INTERVAL) are in seconds.

# ---------- HTTP Server ----------
DEFAULT_SERVER_HOST = "localhost"
DEFAULT_OS_SERVER_PORT = 9000
DEFAULT_ENGINE_SERVER_PORT = 9001

# ---------- Interval ----------
VM_HEARTBEAT_INTERVAL = 5
ENGINE_HEARTBEAT_INTERVAL = 3
OS_LOOP_INTERVAL = 1
ENGINE_LOOP_INTERVAL = 0.01

# NOTE(chaofan): HEARTBEAT_INTERVAL + LOOP_INTERVAL < EXPIRE_TIME
VM_EXPIRE_TIME = 7
ENGINE_EXPIRE_TIME = 7

# ---------- Chunk Related ----------
FILL_NO_CHUNK = -1
PIPELINE_SEND_CHUNK_NUM = 128
DETOKENIZE_CHUNK_NUM = 8
STREAMING_END_TOKEN_ID = -1


# ---------- Recycle Pool ----------
PROCESS_POOL_SIZE = 4096
THREAD_POOL_SIZE = 4096
CONTEXT_POOL_SIZE = 4096
ENGINE_POOL_SIZE = 4096

# ---------- None ID ----------
NONE_THREAD_ID = -1
NONE_CONTEXT_ID = -1
NONE_PROCESS_ID = -1
