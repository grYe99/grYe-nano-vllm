import torch


def test_dtype_mapping():
    from nanovllm.utils.pynccl import NcclCommunicator

    nccl = NcclCommunicator.__new__(NcclCommunicator)
    nccl._init_nccl_types()
    assert torch.bfloat16 in nccl._dtype_map
    assert torch.float32 in nccl._dtype_map
    assert torch.float16 in nccl._dtype_map
    assert torch.int32 in nccl._dtype_map


def test_singleton_guard():
    from nanovllm.utils.pynccl import get_communicator

    import pytest

    with pytest.raises(AssertionError, match="not initialized"):
        get_communicator()


def test_unique_id_length_validation():
    from nanovllm.utils.pynccl import NcclCommunicator

    import pytest

    with pytest.raises(ValueError, match="unique_id must be exactly 128 bytes"):
        NcclCommunicator.__new__(NcclCommunicator).__init__(1, 0, b"too-short")


def test_destroy_singleton():
    from nanovllm.utils.pynccl import (
        NcclCommunicator,
        destroy_communicator,
        get_communicator,
    )

    import pytest

    # Simulate an initialized singleton
    nccl = NcclCommunicator.__new__(NcclCommunicator)
    nccl._lib = None  # prevent actual NCCL calls
    nccl._comm = None

    import nanovllm.utils.pynccl as pynccl_mod
    pynccl_mod._NCCL_COMM = nccl

    # Singleton should be accessible
    assert get_communicator() is nccl

    # Destroy should reset the singleton
    destroy_communicator()
    with pytest.raises(AssertionError, match="not initialized"):
        get_communicator()
