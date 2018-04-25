from __future__ import division
from __future__ import print_function

import json
from collections import defaultdict, OrderedDict

from phrase_tree import PhraseTree


class FeatureMapper(object):
    """
    Maps words, tags, and label actions to indices.
    """

    @staticmethod
    def vocab_init(fname, verbose=True):
        """
        Learn vocabulary from file of strings.
        """
        tag_freq = defaultdict(int)

        trees = PhraseTree.load_treefile(fname)

        for i, tree in enumerate(trees):
            for (word, tag) in tree.sentence:
                tag_freq[tag] += 1

            if verbose:
                print('\rTree {}'.format(i), end='')
                sys.stdout.flush()

        if verbose:
            print('\r', end='')

        tags = ['XX'] + sorted(tag_freq)
        tdict = OrderedDict((t, i) for (i, t) in enumerate(tags))

        if verbose:
            print('Loading features from {}'.format(fname))
            print('( {} tags)'.format(
                len(tdict),
            ))

        return {
            'tdict': tdict,
        }

    def __init__(self, vocabfile, verbose=True):

        if vocabfile is not None:
            data = FeatureMapper.vocab_init(
                fname=vocabfile,
                verbose=verbose,
            )
            self.tdict = data['tdict']

            self.init_tag()

    def init_tag(self):
        self.tag_list = [0 for _ in range(len(self.tdict))]
        for k, v in self.tdict.iteritems():
            self.tag_list[v] = k

    @staticmethod
    def from_dict(data):
        new = FeatureMapper(None)
        new.tdict = data['tdict']
        new.init_tag()
        return new

    def as_dict(self):
        return {
            'tdict': self.tdict,
        }

    def save_json(self, filename):
        with open(filename, 'w') as fh:
            json.dump(self.as_dict(), fh)

    @staticmethod
    def load_json(filename):
        print('load vocab ...')
        with open(filename) as fh:
            data = json.load(fh, object_pairs_hook=OrderedDict)
        return FeatureMapper.from_dict(data)


    def tag_id(self, tag_str):
        return self.tdict[tag_str] if tag_str in self.tdict else self.tdict[FeatureMapper.UNK]

    def tag_str(self, id):
        return self.tag_list[id]

    def is_tag(self, tag_str):
        return (tag_str in self.tdict)


if __name__ == "__main__":
    outfile = "../data/vocab.json"
    import sys

    infile = sys.argv[1]

    fm = FeatureMapper(infile)
    fm.save_json(outfile)

    print("init the vocab finish")
