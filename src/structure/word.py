class Word(object):
    def __init__(self, word, tag=None):
        self.word = word
        self.tag = tag

    def __str__(self):
        return "{}, {}".format(self.word, self.tag)
