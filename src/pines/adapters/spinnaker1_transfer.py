"""Bound the payload size of SpiNNaker proxy memory writes.

This host-side compatibility layer changes only upload granularity. Every byte
is written once at its original device address through the existing transceiver.
"""
from __future__ import annotations

from functools import wraps


def bounded_memory_writer(original, chunk_bytes=256*1024):
    if chunk_bytes < 4 or chunk_bytes % 4:
        raise ValueError("Memory transfer chunks must contain complete 32-bit words")

    @wraps(original)
    def write(self, x, y, base_address, data, *, n_bytes=None, offset=0,
              cpu=0, get_sum=False):
        # The data-specification loader supplies complete byte regions. Other
        # public API forms retain the installed transceiver's behavior.
        if (not isinstance(data, (bytes, bytearray)) or len(data) <= chunk_bytes
                or offset != 0 or n_bytes not in (None, len(data))):
            return original(self, x, y, base_address, data, n_bytes=n_bytes,
                            offset=offset, cpu=cpu, get_sum=get_sum)
        written, total = 0, 0
        for start in range(0, len(data), chunk_bytes):
            chunk = data[start:start+chunk_bytes]
            count, subtotal = original(self, x, y, base_address+start, chunk,
                n_bytes=len(chunk), offset=0, cpu=cpu, get_sum=get_sum)
            written += count
            total = (total+subtotal) & 0xFFFFFFFF
        # Preserve the native API's optional additive protocol sum.
        return written, total

    return write


def install_bounded_memory_transfer(chunk_bytes=256*1024):
    from spinnman.spalloc.spalloc_transceiver import SpallocTransceiver

    SpallocTransceiver.write_memory = bounded_memory_writer(
        SpallocTransceiver.write_memory, chunk_bytes)
    return dict(name="bounded_proxy_memory_writes", max_payload_bytes=chunk_bytes,
                effect="Contiguous byte-identical writes through the installed transceiver")
