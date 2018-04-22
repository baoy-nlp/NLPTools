from __future__ import division
from __future__ import print_function

from parser_utils import write_docs
from phrase_tree import PhraseTree, FScore


def len_match_error(gold_tree, pred_tree):
    if len(gold_tree) != len(pred_tree):
        return 1
    else:
        return 0


def eval_parse(gold_file, test_file):
    gold_trees = PhraseTree.load_treefile(gold_file)
    test_trees = PhraseTree.load_treefile(test_file)
    cumulative = FScore()
    len_match_analysis = 0
    for gold, test in zip(gold_trees, test_trees):
        if len(test.sentence) == 0:
            acc = FScore()
        else:
            acc = test.compare(gold, advp_prt=True)
        len_match_analysis += len_match_error(gold, test)
        cumulative += acc

    # write_docs("test.gold", docs=gold_trees)
    # write_docs("test.pred", docs=test_trees)
    print("length match error ratio:", len_match_analysis / len(gold_trees))
    return cumulative
