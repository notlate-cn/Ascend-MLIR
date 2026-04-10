import json, struct
schema=json.load(open('/tmp/fc_leakyrelu_tiling.json'))
out=bytearray()
for p in schema['tiling_params']:
    out += struct.pack('<i', p['value'])
open('/tmp/fc_leakyrelu_tiling.bin','wb').write(out)
print(len(out))
