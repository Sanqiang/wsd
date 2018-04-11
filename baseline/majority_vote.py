from model.model_config import BaseConfig
from data_generator.data import EvalData
from model.model_config import get_path
from operator import itemgetter


mapper = {}
for line in open(get_path('../wsd_data/medline/abbr.txt')):
    items = line.strip().split('\t')
    cnt = int(items[1])
    abbr_items = items[0].split('|')
    if abbr_items[1] not in mapper:
        mapper[abbr_items[1]] = []
    mapper[abbr_items[1]].append((abbr_items[2], cnt))
for abbr in mapper:
    mapper[abbr].sort(key=itemgetter(-1), reverse=True)


model_config = BaseConfig()
data = EvalData(model_config)

sample = data.get_sample()
correct_cnt, correct_cnt2, correct_cnt3, correct_cnt4, correct_cnt5 = 0.0, 0.0, 0.0, 0.0, 0.0
total_cnt = 0.0
while sample is not None:
    ts = sample['targets']

    for t in ts:
        if t[1] == 0 or t[2] == 0:
            continue
        cascade_add = False
        try:
            pred = data.sense2id[data.id2abbr[t[1]] + '|' + mapper[data.id2abbr[t[1]]][0][0]]
            if pred == t[2]:
                correct_cnt += 1
                cascade_add = True
            else:
                cascade_add = False
        except:
            continue

        try:
            pred2 = [data.sense2id[data.id2abbr[t[1]] + '|' + mapper[data.id2abbr[t[1]]][0][0]],
                     data.sense2id[data.id2abbr[t[1]] + '|' + mapper[data.id2abbr[t[1]]][1][0]]]
            if t[2] in pred2:
                correct_cnt2 += 1
                cascade_add = True
            else:
                cascade_add = False
        except IndexError:
            if cascade_add:
                correct_cnt2 += 1

        try:
            pred3 = [data.sense2id[data.id2abbr[t[1]] + '|' + mapper[data.id2abbr[t[1]]][0][0]],
                     data.sense2id[data.id2abbr[t[1]] + '|' + mapper[data.id2abbr[t[1]]][1][0]],
                     data.sense2id[data.id2abbr[t[1]] + '|' + mapper[data.id2abbr[t[1]]][2][0]]]
            if t[2] in pred3:
                correct_cnt3 += 1
                cascade_add = True
            else:
                cascade_add = False
        except IndexError:
            if cascade_add:
                correct_cnt3 += 1

        try:
            pred4 = [data.sense2id[data.id2abbr[t[1]] + '|' + mapper[data.id2abbr[t[1]]][0][0]],
                     data.sense2id[data.id2abbr[t[1]] + '|' + mapper[data.id2abbr[t[1]]][1][0]],
                     data.sense2id[data.id2abbr[t[1]] + '|' + mapper[data.id2abbr[t[1]]][2][0]],
                     data.sense2id[data.id2abbr[t[1]] + '|' + mapper[data.id2abbr[t[1]]][3][0]]]
            if t[2] in pred4:
                correct_cnt4 += 1
                cascade_add = True
            else:
                cascade_add = False
        except IndexError:
            if cascade_add:
                correct_cnt4 += 1


        try:
            pred5 = [data.sense2id[data.id2abbr[t[1]] + '|' + mapper[data.id2abbr[t[1]]][0][0]],
                     data.sense2id[data.id2abbr[t[1]] + '|' + mapper[data.id2abbr[t[1]]][1][0]],
                     data.sense2id[data.id2abbr[t[1]] + '|' + mapper[data.id2abbr[t[1]]][2][0]],
                     data.sense2id[data.id2abbr[t[1]] + '|' + mapper[data.id2abbr[t[1]]][3][0]],
                     data.sense2id[data.id2abbr[t[1]] + '|' + mapper[data.id2abbr[t[1]]][4][0]]]
            if t[2] in pred5:
                correct_cnt5 += 1
                cascade_add = True
            else:
                cascade_add = False
        except IndexError:
            if cascade_add:
                correct_cnt5 += 1

        total_cnt += 1

    sample = data.get_sample()

acc = correct_cnt / total_cnt
acc2 = correct_cnt2 / total_cnt
acc3 = correct_cnt3 / total_cnt
acc4 = correct_cnt4 / total_cnt
acc5 = correct_cnt5 / total_cnt
print('acc_%s_acc2_%s_acc3_%s_acc4_%s_acc5_%s' % (str(acc), str(acc2), str(acc3), str(acc4), str(acc5)))
