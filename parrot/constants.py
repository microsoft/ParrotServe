# Constants for Parrot

# ---------- Recycle Pool ----------
RECYCLE_POOL_SIZE = 4096

# ---------- Program ----------
HEARTBEAT_INTERVAL = 5  # seconds
FUTURE_MAGIC_HEADER = "__future_magic_header__"

# ---------- OS ----------
FILL_NO_CHUNK = -1
PIPELINE_SEND_CHUNK_NUM = 128
DETOKENIZE_CHUNK_NUM = 8
STREAMING_END_TOKEN_ID = -1

# ---------- Protocol ----------
NONE_THREAD_ID = -1
NONE_CONTEXT_ID = -1

# ---------- Engine ----------
ENGINE_LOOP_INTERVAL = 0.01

# NOTE(chaofan): HEARTBEAT_INTERVAL + GC_INTERVAL < CONTEXT_EXPIRE_TIME
GC_INTERVAL = 10  # seconds
CONTEXT_EXPIRE_TIME = 25  # seconds
