from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optimize

from core_nlp.data.phrase_tree import PhraseTree
from core_nlp.inference.parse import Parser
from core_nlp.models.parser.span_parser_nn import SpanParserNN
from core_nlp.utils.measures import FScore


def train(fm, args):
    train_data_file = args.train
    dev_data_file = args.dev
    epochs = args.epochs
    batch_size = args.batch_size
    unk_param = args.unk_param
    alpha = args.alpha
    beta = args.beta
    model_save_file = args.model

    print("this is train mode")
    start_time = time.time()

    network = SpanParserNN(fm, args)
    optimizer = optimize.Adadelta(network.parameters(), eps=1e-7, rho=0.99)

    # network.cuda()

    training_data = fm.gold_data_from_file(train_data_file)
    num_batches = -(-len(training_data) // batch_size)
    print('Loaded {} training sentences ({} batches of size {})!'.format(
        len(training_data),
        num_batches,
        batch_size,
    ))
    parse_every = -(-num_batches // 4)

    dev_trees = PhraseTree.load_trees(dev_data_file)
    print('Loaded {} validation trees!'.format(len(dev_trees)))

    best_acc = FScore()

    for epoch in xrange(1, epochs + 1):
        print('........... epoch {} ...........'.format(epoch))

        total_cost = 0.0
        total_states = 0
        training_acc = FScore()

        np.random.shuffle(training_data)

        for b in xrange(num_batches):
            batch = training_data[(b * batch_size): ((b + 1) * batch_size)]

            explore = [
                Parser.exploration(
                    example,
                    fm,
                    network,
                    alpha=alpha,
                    beta=beta,
                ) for example in batch
            ]
            for (_, acc) in explore:
                training_acc += acc

            batch = [example for (example, _) in explore]
            sum_loss = np.zeros(1)

            for example in batch:

                ## random UNKing ##
                for (i, w) in enumerate(example['w']):
                    if w <= 2:
                        continue

                    freq = fm.word_freq_list[w]
                    drop_prob = unk_param / (unk_param + freq)
                    r = np.random.random()
                    if r < drop_prob:
                        example['w'][i] = 0

                fwd, back = network.evaluate_word(
                    example['w'],
                    example['t'],
                )

                for (left, right), correct in example['struct_data'].items():
                    scores = network(fwd, back, left, right, 'struct')
                    probs = F.softmax(scores, dim=0)
                    loss = -torch.log(probs[correct])
                    sum_loss += loss.data.numpy()
                    loss.backward(retain_graph=True)

                total_states += len(example['struct_data'])

                for (left, right), correct in example['label_data'].items():
                    scores = network(fwd, back, left, right, 'label')
                    probs = F.softmax(scores, dim=0)
                    loss = -torch.log(probs[correct])
                    sum_loss += loss.data.numpy()
                    loss.backward(retain_graph=True)
                total_states += len(example['label_data'])

            total_cost += sum_loss
            optimizer.step()
            network.zero_grad()
            mean_cost = total_cost / total_states

            print(
                '\rBatch {}  Mean Cost {:.4f} [Train: {}]'.format(
                    b,
                    mean_cost,
                    training_acc,
                ),
                end='',
            )
            sys.stdout.flush()

            if ((b + 1) % parse_every) == 0 or b == (num_batches - 1):
                dev_acc = Parser.evaluate_corpus(
                    dev_trees,
                    fm,
                    network,
                )
                print('  [Val: {}]'.format(dev_acc))

                if dev_acc > best_acc:
                    best_acc = dev_acc
                    torch.save(network, model_save_file)
                    print('    [saved model: {}]'.format(model_save_file))

        current_time = time.time()
        runmins = (current_time - start_time) / 60.
        print('  Elapsed time: {:.2f}m'.format(runmins))
