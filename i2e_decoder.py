import struct

def decode_i2e_from_bytes(file_bytes: bytes) -> tuple[dict, int, int]:
    """
    Decodes an .i2e binary payload directly into frequency genes for rendering.
    Returns (genes, width, height)
    """
    import io
    f = io.BytesIO(file_bytes)
    
    magic = f.read(4)
    if magic == b'I2EQ':
        # uncompressed payload – continue as before
        version, width, height = struct.unpack('<BHH', f.read(5))
    elif magic == b'I2EZ':
        # compressed payload – after reading version, width, height we will decompress the rest
        import zlib
        version, width, height = struct.unpack('<BHH', f.read(5))
        f = io.BytesIO(zlib.decompress(f.read()))
    else:
        raise ValueError("Not a valid .i2e file")
    
    channels = ['R', 'G', 'B']
    genes = {}
    genes['dct_num_terms'] = 0
    
    for channel in channels:
        num_terms_bytes = f.read(4)
        if not num_terms_bytes:
            break
        num_terms = struct.unpack('<I', num_terms_bytes)[0]
        genes[f'dct_num_terms_{channel}'] = num_terms
        genes['dct_num_terms'] = max(genes['dct_num_terms'], num_terms)
        
        prev_u = 0
        prev_v = 0
        
        for i in range(num_terms):
            delta_u, delta_v, C = struct.unpack('<hhf', f.read(8))
            u = prev_u + delta_u
            v = prev_v + delta_v
            
            genes[f'dct_{channel}_u_{i}'] = u
            genes[f'dct_{channel}_v_{i}'] = v
            genes[f'dct_{channel}_C_{i}'] = C
            
            prev_u = u
            prev_v = v
            
    return genes, width, height

def decode_i2e_to_string(file_path: str, str_x: str = "x", str_y: str = "y") -> str:
    """
    Decodes an .i2e binary file back into the massive 5MB mathematical equation string.
    """
    with open(file_path, 'rb') as f:
        magic = f.read(4)
        if magic != b'I2EQ':
            raise ValueError("Not a valid .i2e file")
            
        version, width, height = struct.unpack('<BHH', f.read(5))
        
        channels = ['R', 'G', 'B']
        eqs = {}
        
        for channel in channels:
            num_terms_bytes = f.read(4)
            if not num_terms_bytes:
                break
            num_terms = struct.unpack('<I', num_terms_bytes)[0]
            
            prev_u = 0
            prev_v = 0
            
            terms = []
            for _ in range(num_terms):
                delta_u, delta_v, C = struct.unpack('<hhf', f.read(8))
                u = prev_u + delta_u
                v = prev_v + delta_v
                
                terms.append(f"({C:.5f} * cos({v} * pi * {str_x}) * cos({u} * pi * {str_y}))")
                
                prev_u = u
                prev_v = v
                
            if not terms:
                terms.append("0")
                
            def balanced_sum(t_list):
                if len(t_list) == 1:
                    return t_list[0]
                mid = len(t_list) // 2
                left = balanced_sum(t_list[:mid])
                right = balanced_sum(t_list[mid:])
                return f"({left} + {right})"
                
            raw_eq = balanced_sum(terms)
            max_0 = f"(0.5 * ({raw_eq} + abs({raw_eq})))"
            clamp_1 = f"(0.5 * ({max_0} + 1.0 - abs({max_0} - 1.0)))"
            eqs[channel] = clamp_1
            
        R_eq = f"({eqs.get('R', '0')} * (0.5 * v^2 - 1.5 * v + 1.0))"
        G_eq = f"({eqs.get('G', '0')} * (-v^2 + 2.0 * v))"
        B_eq = f"({eqs.get('B', '0')} * (0.5 * v^2 - 0.5 * v))"
        
        return f"({R_eq} + {G_eq} + {B_eq})"

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        eq = decode_i2e_to_string(sys.argv[1])
        print(f"Decoded {len(eq)} character mathematical equation successfully.")
