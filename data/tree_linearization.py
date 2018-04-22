def write_docs(fname, docs):
    f = open(fname, 'w')
    for doc in docs:
        f.write(str(doc))
        f.write('\n')
    f.close()


def rep_TD_linearization(tree_string):
    stack, tags, words = [], [], []
    for id, tok in enumerate(tree_string.strip().split()):
        if tok[0] == "(":
            symbol = tok[1:]
            tags.append(symbol)
            stack.append(symbol)
        else:
            assert tok[-1] == ")"
            stack.pop()
            tags.pop()
            tags.append("XX")
            while tok[-2] == ")":
                tags.append("/" + stack.pop())
                tok = tok[:-1]
            words.append(tok[:-1])
    return str.join(" ", words), str.join(" ", tags[1:-1])  # Strip "TOP" tag.


def top_down_linearization(tree_string):
    stack, tags, words = [], [], []
    for tok in tree_string.strip().split():
        if tok[0] == "(":
            symbol = tok[1:]
            tags.append(symbol)
            stack.append(symbol)
        else:
            assert tok[-1] == ")"
            stack.pop()  # Pop the POS-tag.
            while tok[-2] == ")":
                tags.append("/" + stack.pop())
                tok = tok[:-1]
            words.append(tok[:-1])
    return str.join(" ", words), str.join(" ", tags[1:-1])  # Strip "TOP" tag.


def process_tree_bank(input_file, output_file, func=rep_TD_linearization):
    res = []
    with open(input_file, 'r') as tree_file:
        for line in tree_file:
            src, tar = func(line)
            res.append(src + "\t" + tar)

    write_docs(output_file, docs=res)


if __name__ == "__main__":
    import sys

    tree_file = sys.argv[1]
    out_file = sys.argv[2] if len(sys.argv) > 2 else sys.argv[1] + ".s2s"
    process_tree_bank(tree_file, out_file, func=top_down_linearization)
