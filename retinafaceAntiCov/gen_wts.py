import struct
from retinaface_cov import RetinaFaceCoV

gpuid = 0
model = RetinaFaceCoV('./cov2/mnet_cov2', 0, gpuid, 'net3l')

f = open('retinafaceAntiCov.wts', 'w')
f.write('{}\n'.format(len(model.model.get_params()[0].keys()) + len(model.model.get_params()[1].keys())))
for k, v in model.model.get_params()[0].items():
    vr = v.reshape(-1).asnumpy()
    f.write('{} {} '.format(k, len(vr)))
    for vv in vr:
        f.write(' ')
        f.write(struct.pack('>f',float(vv)).hex())
    f.write('\n')
for k, v in model.model.get_params()[1].items():
    vr = v.reshape(-1).asnumpy()
    f.write('{} {} '.format(k, len(vr)))
    for vv in vr:
        f.write(' ')
        f.write(struct.pack('>f',float(vv)).hex())
    f.write('\n')

