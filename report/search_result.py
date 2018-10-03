import os
import operator

mapper = {}
path = '/home/zhaos5/projs/wsd/wsd_perf'
for root, dirs, files in os.walk(path):
    for file in files:
        if file.startswith('step') and '0930' in root:
            acc_sidx = file.index('_acc')
            acc_eidx = file.rindex('_acc2', acc_sidx)
            try:
                acc = float(file[acc_sidx+len('-acc'):acc_eidx])
            except:
                print('error:%s%s' % (root, file))
                sari = 0.0
            mapper[root + '/' + file] = acc
mapper = sorted(mapper.items(), key=operator.itemgetter(1), reverse=True)
cnt = 10
for k,v in mapper:
    if cnt == 0:
        break
    cnt -= 1
    print(k)
    print(v)
