"""Split the Zhimeng Luo's data into training/eval with preprocess."""
import random as rd


def splitter():
    buffer_eval = []
    buffer_train = []

    path = '/Users/zhaosanqiang916/git/wsd_data/zhimeng/medline_procs.processed.txt'
    for line in open(path):
        if rd.random() < 0.1:
            buffer_eval.append(line)
        else:
            buffer_train.append(line)

    eval_path = '/Users/zhaosanqiang916/git/wsd_data/medline/eval.txt'
    train_path = '/Users/zhaosanqiang916/git/wsd_data/medline/train.txt'
    f_eval = open(eval_path, "w")
    f_train = open(train_path, "w")

    f_train.write(''.join(buffer_train))
    f_eval.write(''.join(buffer_eval))
    f_train.close()
    f_eval.close()


if __name__ == '__main__':
    splitter()