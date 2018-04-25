from __future__ import division
from __future__ import print_function

from collections import defaultdict

from phrase_tree import PhraseTree, FScore


def eval_tree(gold, test):
    def fields(brackets):
        count = defaultdict(int)
        for item, nums in brackets.items():
            count[(item[1], item[2])] += nums
        return count

    def compare(goldbracks, predbracks):
        correct = 0
        for gb in goldbracks:
            if gb in predbracks:
                correct += min(goldbracks[gb], predbracks[gb])
        pred_total = sum(predbracks.values())
        gold_total = sum(goldbracks.values())
        return correct, pred_total, gold_total

    gold_brackets = gold.brackets(advp_prt=True)
    pred_brackets = test.brackets(advp_prt=True)
    correct, pred_total, gold_total = compare(gold_brackets, pred_brackets)
    label_fscore = FScore(correct, pred_total, gold_total)
    correct, pred_total, gold_total = compare(fields(gold_brackets), fields(pred_brackets))
    struct_fscore = FScore(correct, pred_total, gold_total)
    return {
        'gold_length': len(gold),
        'pred_length': len(test),
        'struct': struct_fscore,
        'label': label_fscore,
        'gold_span': gold_total,
        'pred_span': pred_total
    }


def eval_trees(gold_trees, test_trees, verbose=False):
    gold_words = 0
    gold_spans = 0
    pred_words = 0
    pred_spans = 0
    struct_score = FScore()
    label_score = FScore()
    for gold, test in zip(gold_trees, test_trees):
        eval_res = eval_tree(gold, test)
        gold_words += eval_res['gold_length']
        pred_words += eval_res['pred_length']
        gold_spans += eval_res['gold_span']
        pred_spans += eval_res['pred_span']
        struct_score += eval_res['struct']
        label_score += eval_res['label']
    if verbose:
        count = len(gold_trees)
        print("count=", count)
        print("gold_avg_len={},pred_avg_len={},count={}".format(gold_words, pred_words, count))
        print("gold_avg_span={},pred_avg_span={},count={}".format(gold_spans, pred_spans, count))
        print("struct_score={},label_score={}".format(struct_score, label_score))

    return struct_score, label_score


def eval_files(gold_file, test_file):
    gold_trees = PhraseTree.load_treefile(gold_file)
    test_trees = PhraseTree.load_treefile(test_file)
    accuracy = FScore()

    match_gold_trees = []
    match_pred_trees = []
    umatch_gold_trees = []
    umatch_pred_trees = []
    for gold, test in zip(gold_trees, test_trees):
        if len(gold) == len(test):
            match_gold_trees.append(gold)
            match_pred_trees.append(test)
        else:
            umatch_gold_trees.append(gold)
            umatch_pred_trees.append(test)

    print("***eval matched pair***")
    struct, label = eval_trees(match_gold_trees, match_pred_trees, verbose=True)
    accuracy += label
    print("***eval unmatched pair***")
    _, label = eval_trees(umatch_gold_trees, umatch_pred_trees, verbose=True)
    accuracy += label
    print("sum struct score:", struct + _)
    return accuracy
