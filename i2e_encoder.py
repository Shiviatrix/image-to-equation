import struct
from typing import Dict, Any

def encode_i2e(genes: Dict[str, Any], width: int, height: int, output_path: str):
    """
    Encodes DCT frequency genes into a highly compressed .i2e binary file
    using delta encoding on the frequencies.
    """
def encode_i2e_to_bytes(genes: Dict[str, Any], width: int, height: int) -> bytes:
    """
    Encode genes to a compressed .i2e payload using zlib.
    Magic header: I2EZ, followed by version, width, height, then zlib‑compressed channel data.
    """
    import io, zlib
    # --- Inner payload (channel data) ---
    inner = io.BytesIO()
    # 2. Channels
    channels = ['R', 'G', 'B']
    for channel in channels:
        num_terms = genes.get(f'dct_num_terms_{channel}', 0)
        inner.write(struct.pack('<I', num_terms))
        # Extract terms
        terms = []
        for i in range(num_terms):
            u = genes.get(f'dct_{channel}_u_{i}', 0)
            v = genes.get(f'dct_{channel}_v_{i}', 0)
            C = genes.get(f'dct_{channel}_C_{i}', 0.0)
            terms.append((u, v, C))
        # Sort terms for better delta encoding
        terms.sort(key=lambda x: (x[0], x[1]))
        prev_u = prev_v = 0
        for u, v, C in terms:
            delta_u = u - prev_u
            delta_v = v - prev_v
            inner.write(struct.pack('<hhf', delta_u, delta_v, C))
            prev_u, prev_v = u, v
    # Compress the inner payload
    compressed = zlib.compress(inner.getvalue())
    # --- Final payload with header ---
    out = io.BytesIO()
    out.write(b'I2EZ')
    # version 1, width, height
    out.write(struct.pack('<BHH', 1, width, height))
    out.write(compressed)
    return out.getvalue()


def encode_i2e(genes: Dict[str, Any], width: int, height: int, output_path: str):
    """
    Encodes DCT frequency genes into a highly compressed .i2e binary file
    using delta encoding on the frequencies, then compresses the payload with zlib.
    """
    # Generate raw payload (header + delta encoded data)
    raw_payload = encode_i2e_to_bytes(genes, width, height)
    # Compress using zlib (lossless)
    import zlib
    compressed_payload = zlib.compress(raw_payload)
    # Write to file with new magic header indicating compression
    with open(output_path, 'wb') as f:
        f.write(b'I2EZ')  # new magic for compressed .i2e
        # version, width, height are already part of raw_payload first bytes after magic
        # Re‑use the packed version/size from raw payload (bytes 4-8)
        f.write(raw_payload[4:9])  # version (1) + width (2) + height (2)
        f.write(compressed_payload)

