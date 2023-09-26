# Constants for Parrot

# ---------- Recycle Pool ----------
RECYCLE_POOL_SIZE = 4096

# ---------- Controller ----------
FILL_NO_CHUNK = -1
HEARTBEAT_INTERVAL = 5  # seconds

# ---------- Executor ----------
PIPELINE_SEND_CHUNK_NUM = 128
DETOKENIZE_CHUNK_NUM = 8
STREAMING_END_TOKEN_ID = -1

# ---------- Protocol ----------
NONE_SESSION_ID = -1
NONE_CONTEXT_ID = -1

# ---------- Backend ----------
ENGINE_LOOP_INTERVAL = 0.01

# NOTE(chaofan): HEARTBEAT_INTERVAL + GC_INTERVAL < CONTEXT_EXPIRE_TIME
GC_INTERVAL = 10  # seconds
CONTEXT_EXPIRE_TIME = 25  # seconds
