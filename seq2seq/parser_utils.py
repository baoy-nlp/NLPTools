from __future__ import division
from __future__ import print_function

import os


def write_docs(fname, docs):
    f = open(fname, 'w')
    for doc in docs:
        f.write(str(doc))
        f.write('\n')
    f.close()


def data_to_ref(origin_data_file, ref_file):
    res = []
    with open(origin_data_file, 'r') as data_file:
        for line in data_file:
            line_res = line.strip("\n").split("\t")[1]
            res.append(line_res)
    write_docs(ref_file, res)


def seq2tree(translate):
    """
    Args:
        words: the input of translation method
        translate: the output of translation method

    Return:
        PTB Format Plain Text
    Raise:
        Do not Match Exception
    """
    if not translate[-1].endswith("/" + translate[0]):
        translate.append("/" + translate[0])

    translate = pre_valid_process(translate)

    stack = []
    WORD = " XX)"
    for item in translate:
        if not item.startswith("/"):
            stack.append((item, False))
        else:
            key = item[1:]
            span_length = 1
            for span_length in range(1, len(stack) + 1):
                if stack[-span_length][0] == key:
                    break
                else:
                    span_length += 1
            if 1 < span_length <= len(stack):
                new_node = ")"
                for _ in range(span_length - 1):
                    p = stack.pop()
                    if p[1]:
                        new_node = " " + p[0] + new_node
                    else:
                        word = "(" + p[0] + WORD
                        new_node = " " + word + new_node
                p = stack.pop()
                new_p = ("(" + p[0] + new_node, True)
                stack.append(new_p)
    res_list = [s[0] for s in stack]
    res = "(S " + " ".join(res_list) + ")" if len(res_list) > 0 else "(S (XX XX))"

    res = post_valid_process(res)
    # tree = PhraseTree.parse(res)

    return res


def convert_to_ptb(translate_file):
    res = []
    with open(translate_file, 'r') as tree_file:
        for line in tree_file:
            line_tree = seq2tree(line.strip("\n").split(" "))
            res.append(line_tree)

    write_docs(translate_file + ".convert", res)
    return translate_file + ".convert"


def sentence_replace(words, sentence):
    assert len(sentence) <= len(words)
    return [(words[i], sentence[i][1]) for i in range(len(sentence))]


def pre_valid_process(seq):
    left_key = seq[0]
    right_key = "/" + seq[0]
    left_count = 1
    right_count = 0
    for item in seq:
        if item == left_key:
            left_count += 1
        if item == right_key:
            right_count += 1

    if right_count > left_count:
        base = [left_key] * (right_count - left_count)
        base.extend(seq)
        seq = base
    else:
        seq.extend([right_key] * (left_count - right_count))
    return seq


def post_valid_process(seq):
    seqs = seq.split(" ")
    res = []
    for item in seqs:
        if item.startswith("(") or item.endswith(")"):
            res.append(item)
    return " ".join(res)


def eval_f1score(predict_file, target_file):
    from error_analysis import eval_parse
    pred_file = convert_to_ptb(predict_file)
    gold_file = convert_to_ptb(target_file)
    f1_score = eval_parse(gold_file, pred_file)
    os.remove(pred_file)
    os.remove(gold_file)
    return f1_score


if __name__ == "__main__":
    # translate_file = "../parser_data/dev.pred.11800"
    # gold_file = "../parser_data/dev.pred.15000"
    # print(eval_f1score(translate_file, gold_file))
    pred = "S NP NP DT NNP /NP /NP NP NP NP NN /NP /NP VP NP NP NP NN /NP /NP VP /VP /VP " \
           "/S /S /VP /S /S /S /S /VP /S /S /S /S /S /VP /S /S /S /S /S /S /VP /S /S /S /S " \
           "/S /S /VP /S /S /S /S /S /S /VP /S /S /S /S /S /S /VP /S /S /S /S /S /S /VP /S /S " \
           "/S /S /S /S /VP /S /S /S /S /S /S /VP /S /S /S /S /S /S /VP /S /S /S /S /S /S /VP " \
           "/S /S /S /S /S /S /VP /S /S /S /S /S /S /VP /S /S /S /S /S /S /VP /S /S /S /S /S /S " \
           "/VP /S /S /S /S /S /S /VP /S /S /S /S /S /S /VP /S /S /S /S /S /S /VP /S /S /S /S /S " \
           "/S /VP /S /S /S /S /S /S /VP /S /S /S /S /S /S /VP /S /S /S /S /S /S /VP /S /S /S /S /S " \
           "/S /VP /S /S /S /S /S /S /VP /S /S /S /S /S /S /VP /S /S /S /S /S /S /VP /S /S /S /S /S " \
           "/S /VP /S /S /S /S /S /S /VP /S /S /S /S /S /S /VP /S /S /S /S /S /S /VP /S /S /S /S /S " \
           "/S /VP /S /S /S /S /S /S /VP /S /S /S /S /S /S /VP /S /S /S /S /S /S /VP /S /S /S /S /S /S /VP /S"
    tree = seq2tree(pred.split(" "))
    print(tree)
