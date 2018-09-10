import numpy as np

cands = [[2,3,4,5,6], [4,5,6,7,8]]
res = []
for cands_tmp in cands:
    r = np.random.choice(cands_tmp, 2, False)
    res.append(r)

print(res)