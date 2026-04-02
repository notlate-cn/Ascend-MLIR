import json, struct
schema=json.load(open('/Volumes/GM9/code/Ascend-MLIR/examples/matmul-add-relu-sum/tiling_space_mix.json'))
vals={
"usedCoreNum":1,"M":128,"N":128,"Ka":64,"Kb":64,"singleCoreM":128,"singleCoreN":128,"singleCoreK":64,
"baseM":64,"baseN":128,"baseK":64,"depthA1":1,"depthB1":1,"stepM":1,"stepN":1,"isBias":0,"transLength":0,
"iterateOrder":0,"shareMode":0,"shareL1Size":0,"shareL0CSize":0,"shareUbSize":0,"batchM":0,"batchN":0,
"singleBatchM":0,"singleBatchN":0,"stepKa":0,"stepKb":0,"depthAL1CacheUB":0,"depthBL1CacheUB":0,"dbL0A":0,
"dbL0B":0,"dbL0C":0,"ALayoutInfoB":0,"ALayoutInfoS":0,"ALayoutInfoN":0,"ALayoutInfoG":0,"ALayoutInfoD":0,
"BLayoutInfoB":0,"BLayoutInfoS":0,"BLayoutInfoN":0,"BLayoutInfoG":0,"BLayoutInfoD":0,"CLayoutInfoB":0,
"CLayoutInfoS1":0,"CLayoutInfoN":0,"CLayoutInfoG":0,"CLayoutInfoS2":0,"BatchNum":0,"mxTypePara":0}
out=bytearray()
for p in schema['tiling_params']:
    out += struct.pack('<i', vals[p['name']])
open('/Volumes/GM9/code/Ascend-MLIR/examples/matmul-add-relu-sum/proto_packed_mix_v3/t.bin','wb').write(out)
print(len(out))
