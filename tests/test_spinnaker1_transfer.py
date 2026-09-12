import struct

import pytest

from pines.adapters.spinnaker1_transfer import bounded_memory_writer


class Memory:
    def __init__(self):
        self.calls = []

    def write(self, x, y, address, data, *, n_bytes=None, offset=0, cpu=0, get_sum=False):
        self.calls.append((x,y,address,data,n_bytes,offset,cpu,get_sum))
        if isinstance(data, int):
            return 4, data if get_sum else 0
        if len(data) > 256*1024:
            raise ValueError("request payload exceeds service limit")
        total = sum(value[0] for value in struct.iter_unpack("<I", data)) & 0xFFFFFFFF if get_sum else 0
        return len(data), total


@pytest.mark.parametrize("get_sum", [False, True])
def test_large_region_preserves_bytes_addresses_and_native_return_value(get_sum):
    memory = Memory()
    payload = bytes(range(256)) * 4097
    write = bounded_memory_writer(Memory.write)
    count, total = write(memory, 2, 3, 0x70000000, payload, get_sum=get_sum)
    assert count == len(payload)
    assert b"".join(call[3] for call in memory.calls) == payload
    assert [call[2] for call in memory.calls] == [0x70000000+i*256*1024 for i in range(5)]
    assert all(len(call[3]) <= 256*1024 for call in memory.calls)
    expected = sum(value[0] for value in struct.iter_unpack("<I", payload)) & 0xFFFFFFFF if get_sum else 0
    assert total == expected


def test_small_and_non_byte_requests_use_the_original_api():
    memory = Memory()
    write = bounded_memory_writer(Memory.write)
    assert write(memory, 0, 0, 100, b"1234", cpu=2) == (4,0)
    assert len(memory.calls) == 1
    assert memory.calls[0][2:] == (100,b"1234",None,0,2,False)
    assert write(memory, 0, 0, 104, 37, get_sum=True) == (4,37)


def test_chunk_boundary_respects_device_word_size():
    with pytest.raises(ValueError, match="32-bit"):
        bounded_memory_writer(Memory.write, 7)
