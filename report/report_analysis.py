import argparse
import operator
from collections import defaultdict, Counter

parser = argparse.ArgumentParser(description='Report Parameter')
parser.add_argument('-r', '--result', default=None, help='The path of result')
parser.add_argument('-m', '--mode', default=None, help='The mode of analysis')


args = parser.parse_args()

result = args.result


if args.mode == 'abbr':
    abbr_correct_stat, abbr_incorrect_stat = defaultdict(float), defaultdict(float)
    for line in open(result):
        try:
            items = line.split('\t')
            if len(items) < 3:
                continue
            abbr = items[0][5:]
            gt = items[2][3:]
            preds = items[1][5:].split(';')
            pred = preds[0]
            if pred != gt:
                print('incorrect classification\tpred:%s\tgt:%s for abbr:%s' % (pred, gt, abbr))
                abbr_incorrect_stat[abbr] += 1
            else:
                abbr_correct_stat[abbr] += 1
        except:
            print(line)

    c = Counter()
    for abbr in abbr_incorrect_stat:
        if abbr in abbr_correct_stat:
            c[abbr] = abbr_incorrect_stat[abbr] / (abbr_correct_stat[abbr] + abbr_incorrect_stat[abbr])
        # else:
        #     c[abbr] = 1.0
    for abbr, ratio in c.most_common():
        print('abbr:%s\t\terr_rate:%s\t\terr_cnt:%s\t\tcor_cnt:%s' %
              (abbr, ratio, abbr_incorrect_stat[abbr], abbr_correct_stat[abbr]))

if args.mode == 'pair':
    total_cnt = 0
    pair_stat = defaultdict(float)
    for line in open(result):
        items = line.split('\t')
        if len(items) < 3:
            continue
        abbr = items[0][5:]
        gt = items[2][3:]
        preds = items[1][5:].split(';')
        pred = preds[0]
        if pred != gt:
            # print('incorrect classification\tpred:%s\tgt:%s for abbr:%s' % (pred, gt, abbr))
            if hash(pred) > hash(gt):
                pred, gt = gt, pred
            pair_stat[abbr + '|' + pred + '|' + gt] += 1
        total_cnt += 1
    pair_stat = sorted(pair_stat.items(), key=operator.itemgetter(1), reverse=True)

    for k, v in pair_stat:
        print('%s\t%s\t%s' % (k, v, float(v)/total_cnt))