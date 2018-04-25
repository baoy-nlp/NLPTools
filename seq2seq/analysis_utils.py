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


def analysis_seq(translate, fm):
    pad_brackets = 0
    fix_brackets = 0
    rm_brackets = 0

    result = ""
    stack = []
    for item in translate:
        if not item.startswith("/"):
            result += "("
            result += item
            if fm.is_tag(item):
                result += " XX)"
            else:
                stack.append(item)
        else:
            result += ")"
            if len(stack) == 0:
                pad_brackets += 1
            else:
                top = stack.pop()
                if item[1:] != top:
                    fix_brackets += 1
    if not result.startswith("("):
        result = "(" + result
    pad = result.count("(") - result.count(")")
    pad_brackets += pad
    if pad > 0:
        result += ")" * pad
    result = result.replace("(", " ( ")
    result = result.replace(")", " )")
    result = result.replace(")  (", ") (")

    stack = []
    res = ""
    tokens = result.strip(" ").split(" ")
    for index, item in enumerate(tokens):
        if item != ")":
            stack.append(item)
        else:
            if len(stack) == 0:
                pass
            else:
                str = []
                while stack[-1] != "(":
                    top = stack.pop()
                    str.append(top)

                if len(stack) == 1:
                    if index < len(tokens) - 1:
                        stack.append(str.pop())
                        sec = ' '.join(list(reversed(str)))
                        stack.append(sec)
                        rm_brackets += 1
                    else:
                        sec = ' '.join(list(reversed(str)))
                        res = "(" + sec + ")"
                else:
                    stack.pop()
                    if len(str) == 1:
                        rm_brackets += 1
                    else:
                        sec = ' '.join(list(reversed(str)))
                        str = "(" + sec + ")"
                        stack.append(str)

    return pad_brackets, fix_brackets, rm_brackets


def analysis_file(tgt_file, fm):
    res = []
    with open(tgt_file, 'r') as tree_file:
        for line in tree_file:
            line_tree = analysis_seq(line.strip("\n").split(" "), fm)
            res.append(line_tree)

    pad_brackets = [item[0] for item in res]
    fix_brackets = [item[1] for item in res]
    rm_brackets=[item[2] for item in res]

    print("avg pad:", sum(pad_brackets) / len(res))
    print("avg_fix:", sum(fix_brackets) / len(res))
    print("avg_rm:", sum(rm_brackets) / len(res))

def match_construct(translate, fm):
    result = ""
    for item in translate:
        if not item.startswith("/"):
            result += "("
            result += item
            if fm.is_tag(item):
                result += " XX)"
        else:
            result += ")"

    if not result.startswith("("):
        result = "(" + result

    pad = result.count("(") - result.count(")")
    if pad > 0:
        result += ")" * pad
    result = result.replace("(", " ( ")
    result = result.replace(")", " )")
    result = result.replace(")  (", ") (")

    stack = []
    res = ""
    tokens = result.strip(" ").split(" ")
    for index, item in enumerate(tokens):
        if item != ")":
            stack.append(item)
        else:
            if len(stack) == 0:
                pass
            else:
                str = []
                while stack[-1] != "(":
                    top = stack.pop()
                    str.append(top)

                if len(stack) == 1:
                    if index < len(tokens) - 1:
                        stack.append(str.pop())
                        sec = ' '.join(list(reversed(str)))
                        stack.append(sec)

                    else:
                        sec = ' '.join(list(reversed(str)))
                        res = "(" + sec + ")"
                else:
                    stack.pop()
                    if len(str) == 1:
                        pass
                    else:
                        sec = ' '.join(list(reversed(str)))
                        str = "(" + sec + ")"
                        stack.append(str)
    return res


def seq2tree(translate):
    """
    Args:
        words: the input of translation method
        translate: the output of translation method

    Return:
        PTB Format Plain Text
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


def convert_to_ptb2(translate_file, fm):
    res = []
    with open(translate_file, 'r') as tree_file:
        for line in tree_file:
            line_tree = match_construct(line.strip("\n").split(" "), fm)
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
    from analysis import eval_files
    pred_file = convert_to_ptb(predict_file)
    gold_file = convert_to_ptb(target_file)
    f1_score = eval_files(gold_file, pred_file)
    os.remove(pred_file)
    os.remove(gold_file)
    return f1_score


def eval_f1score2(predict_file, target_file, fm=None):
    if fm is None:
        from global_names import GlobalNames
        if GlobalNames.fm is None:
            from features import FeatureMapper
            GlobalNames.fm = FeatureMapper.load_json(GlobalNames.fm_file)
        fm = GlobalNames.fm
    from analysis import eval_files
    pred_file = convert_to_ptb2(predict_file, fm)
    gold_file = convert_to_ptb2(target_file, fm)
    f1_score = eval_files(gold_file, pred_file)
    os.remove(pred_file)
    os.remove(gold_file)
    return f1_score


if __name__ == "__main__":
    from features import FeatureMapper
    from global_names import GlobalNames

    fm = FeatureMapper.load_json("../data/vocab.json")
    # pred = "FRAG NP INTJ RB PRP$ NN /NP : NP NNP NNP POS /NP . /FRAG"
    # tree = match_construct(pred.split(), fm)
    # print(tree)
    # pred = "NP NP RB PRP$ VB /NP : NP JJ NNP POS /NP . /NP"
    # tree = match_construct(pred.split(), fm)
    # print(tree)

    # origin = seq2tree(pred.split())
    # print(origin)
    #
    # gold_file = "../s2t-3lstm-bidirectional/.mid_dev.ref"
    # pred_file = "../s2t-3lstm-bidirectional/.mid_dev.pred"
    #
    # print(eval_f1score(pred_file, gold_file))
    # print(eval_f1score2(pred_file, gold_file, fm))

    file = ".mid_dev.pred"
    analysis_file(file, fm)
