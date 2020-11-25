import numpy as np, random as rnd, pickle
from collections import Counter, defaultdict


def argmax(dictionary):
    max_key = next(iter(dictionary.keys()))
    for k, v in dictionary.items():
        if v > dictionary[max_key]:
            max_key = k
    return max_key

if __name__ == "__main__":
    with open("poems.txt") as f:
        data = [
            (int(pid), int(stmt), list(doc))
            for pid, stmt, *_, doc in (line.split("\t")
                                       for line in f.read().splitlines())
        ]

    print(data[0])

    rnd.seed(471)
    rnd.shuffle(data)
    print(data[0])

    bnd = int(len(data) * 0.9)
    train = data[:bnd]
    test = data[bnd:]

    Nc = Counter(stmt for _, stmt, *_ in train)
    Ndoc = len(train)
    logprior = {k: np.log(v / Ndoc) for k, v in Nc.items()}
    print(Nc, Ndoc, logprior)

    vocabulary = list(set(ch for *_, doc in train for ch in doc))
    print(len(vocabulary), vocabulary[:14])

    bigdoc = defaultdict(list)
    for _, stmt, doc in train:
        bigdoc[stmt].extend(doc)

    print(*(bigdoc[stmt][:14] for stmt in bigdoc), sep="\n")
    print(*(len(bigdoc[stmt]) for stmt in bigdoc))

    counts = {k: Counter(v) for k, v in bigdoc.items()}
    print(*(counts[stmt].most_common(5) for stmt in counts), sep="\n")
    print(*(len(counts[stmt]) for stmt in counts))

    for stmt in counts:
        counts[stmt].update(vocabulary)

    print(*(counts[stmt].most_common(5) for stmt in counts), sep="\n")
    print(*(len(counts[stmt]) for stmt in counts))

    logsum = {stmt: np.log(len(doc) + len(vocabulary))
              for stmt, doc in bigdoc.items()}
    loglikelihood = {
        ch: {stmt: np.log(cnt[ch]) - logsum[stmt]
             for stmt, cnt in counts.items()}
        for ch in vocabulary
    }

    with open("loglike.pkl", "wb") as fb:
        pickle.dump(loglikelihood, fb)

    print(loglikelihood['愁'], loglikelihood['秀'])

    vocabset = set(vocabulary)
    logp_bayes = defaultdict(lambda: {stmt: lp for stmt, lp in logprior.items()})
    for *meta, doc in test:
        for ch in doc:
            if ch in vocabset:
                for stmt in counts:
                    logp_bayes[tuple(meta)][stmt] += loglikelihood[ch][stmt]

    results = {pid: (ans, argmax(res))
               for (pid, ans), res in logp_bayes.items()}
    print(results)

    accuracy = sum(v[0] == v[1] for v in results.values()) / len(results)
    print(accuracy)

    precision = dict()
    recall = dict()
    for stmt in logprior:
        all_eq = ans_eq = res_eq = 0

        for ans, res in results.values():
            if ans == stmt:
                ans_eq += 1
                if ans == res:
                    all_eq += 1
            if res == stmt:
                res_eq += 1

        precision[stmt] = all_eq / res_eq
        recall[stmt] = all_eq / ans_eq

    print(precision, recall, sep="\n")
