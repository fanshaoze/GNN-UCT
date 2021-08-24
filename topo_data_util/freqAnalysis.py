import argparse
import collections
import json
import math

from utils.graphUtils import copy_graph_files
import criteria


def frequency_analysis(docs, bag_of_words):
    """
    :return: {path: ratio of pos docs that contain path - ratio of neg docs that contain path}
    """
    doc_num = len(docs)
    return collections.Counter({word: 1. * sum(word in doc for doc in docs) / doc_num for word in bag_of_words})


def tf(doc):
    result = collections.defaultdict(float)
    doc_len = len(doc)
    for word in doc:
        result[word] += 1. / doc_len
    return result

def idf(docs, bag_of_words):
    result = {}
    for word in bag_of_words:
        occur_num = sum(word in doc for doc in docs)
        result[word] = math.log(1. * len(docs) / occur_num)
    return result

def tf_idf_analysis(docs, bag_of_words):
    """
    :return: {word: tf-idf in docs}
    """
    doc_num = len(docs)

    idf_counter = idf(docs, bag_of_words)

    results = collections.defaultdict(float)
    for doc in docs:
        tf_counter = tf(doc)
        for word in doc:
            results[word] += tf_counter[word] * idf_counter[word]

    # normalize
    for word in results.keys():
        results[word] /= 1. * doc_num
        #results[word] /= 1. * doc_occurrence_nums[word]

    return results


def print_results(results:list):
    for result in results:
        print('path: %s' % result[0])
        print('freq: %.4f' % result[1])

def print_stats(results):
    print('Most occurrences in positive cases')
    print_results(results[:20])
    print()
    print('Most occurrences in negative cases')
    print_results(list(reversed(results[-20:])))


def copy_positive_figs(pos_names):
    from config import EXP_DIR, RANDOM_DATA
    copy_graph_files(pos_names, EXP_DIR + RANDOM_DATA, 'positive_cases/')

def find_most_freq_paths(data, is_positive, is_negative, metric_name='freq'):
    """
    :param data: {name: {paths:, eff:, vout:}}
    :return: [(path, freq), ...] in descent order in freq
    """
    names = data.keys()

    pos_names = []
    pos_docs = []
    neg_docs = []
    bag_of_words = set()

    for name in names:
        words = data[name]['paths']
        eff = data[name]['eff']
        vout = data[name]['vout']

        if is_positive(eff, vout):
            pos_docs.append(words)
            pos_names.append(name)
        elif is_negative(eff, vout):
            neg_docs.append(words)
        else:
            # not gooing to deal with moderate topos
            continue

        bag_of_words.update(words)

    print('positive topos:', pos_names)
    copy_positive_figs(pos_names)

    if metric_name == 'freq':
        metric = frequency_analysis
    elif metric_name == 'tfidf':
        metric = tf_idf_analysis
    else:
        raise Exception('unknown metric ' + args.metric)

    pos_results = metric(pos_docs, bag_of_words)
    neg_results = metric(neg_docs, bag_of_words)
    print('positive cases', len(pos_docs))
    print('negative cases', len(neg_docs))

    results = pos_results
    results.subtract(neg_results)
    # [(path, freq), ...] in descent order in freq
    results = list(sorted(results.items(), key=lambda _: _[1], reverse=True))

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--metric', type=str, default='freq')
    args = parser.parse_args()

    data = json.load(open('words.json'))
    results = find_most_freq_paths(data, criteria.high_reward, criteria.low_reward, args.metric)
    print_stats(results)
