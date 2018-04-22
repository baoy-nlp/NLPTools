from seq2seq.parser_utils import write_docs
from seq2seq.parser_utils import eval_f1score

fname = '.mid_dev.pred'


def line_to_list(line=str()):
    line = line.replace(',', '')
    line = line.replace("[", "")
    line = line.replace("]", "")
    line = line.replace("u", '')
    line = line.replace("'", "")
    return line.split(" ")


res = []
with open(fname, 'r') as tree_file:
    for line in tree_file:
        res.append(' '.join(line_to_list(line.strip("\n"))))

write_docs(fname[1:], res)

print eval_f1score(".mid_dev.ref", fname[1:])
