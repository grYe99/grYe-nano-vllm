import os

_AR_ASYNC_CHUNKED: bool | None = None
_AR_FUSED_NORM: bool | None = None
_AR_NUM_CHUNKS: int | None = None
_AR_MIN_TOKENS: int | None = None


def configure(
    ar_async_chunked: bool = False,
    ar_fused_norm: bool = True,
    ar_num_chunks: int = 2,
    ar_min_tokens: int = 1024,
) -> None:
    global _AR_ASYNC_CHUNKED, _AR_FUSED_NORM, _AR_NUM_CHUNKS, _AR_MIN_TOKENS
    _AR_ASYNC_CHUNKED = ar_async_chunked
    _AR_FUSED_NORM = ar_fused_norm
    _AR_NUM_CHUNKS = ar_num_chunks
    _AR_MIN_TOKENS = ar_min_tokens


def _resolve_from_env() -> tuple[bool, bool] | None:
    mode = os.environ.get("NANOVLLM_AR_MODE")
    if mode is not None:
        mode = int(mode)
        if mode == 0:
            return False, False
        elif mode == 1:
            return False, True
        elif mode == 2:
            return True, True
    return None


def _ensure_env_resolved() -> None:
    global _AR_ASYNC_CHUNKED, _AR_FUSED_NORM
    if _AR_ASYNC_CHUNKED is not None and _AR_FUSED_NORM is not None:
        return
    env = _resolve_from_env()
    if env is not None:
        if _AR_ASYNC_CHUNKED is None:
            _AR_ASYNC_CHUNKED = env[0]
        if _AR_FUSED_NORM is None:
            _AR_FUSED_NORM = env[1]
    else:
        if _AR_ASYNC_CHUNKED is None:
            _AR_ASYNC_CHUNKED = False
        if _AR_FUSED_NORM is None:
            _AR_FUSED_NORM = True


def get_ar_async_chunked() -> bool:
    _ensure_env_resolved()
    return _AR_ASYNC_CHUNKED


def get_ar_fused_norm() -> bool:
    _ensure_env_resolved()
    return _AR_FUSED_NORM


def get_ar_num_chunks() -> int:
    global _AR_NUM_CHUNKS
    if _AR_NUM_CHUNKS is None:
        _AR_NUM_CHUNKS = int(os.environ.get("NANOVLLM_AR_NUM_CHUNKS", "2"))
    return _AR_NUM_CHUNKS


def get_ar_min_tokens() -> int:
    global _AR_MIN_TOKENS
    if _AR_MIN_TOKENS is None:
        _AR_MIN_TOKENS = int(os.environ.get("NANOVLLM_AR_MIN_TOKENS", "1024"))
    return _AR_MIN_TOKENS
