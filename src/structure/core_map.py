class CoreMap(object):
    def __init__(self, sentence, structure=None):
        self.sentence = sentence
        self.structure = structure

    def __str__(self):
        return "sentence:{}\ttree:{}".format(self.sentence, self.structure)
