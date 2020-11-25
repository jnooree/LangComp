import numpy as np

N = 1410000000

def smooth(corpus:dict, k=1):
    result = dict()
    for n in corpus:
        logN = np.log(N + k * len(corpus[n]))
        result[n] = {key: np.log(val + k) - logN
                     for key, val in corpus[n].items()}
    return result


def prob_ngram(n:int, sent:str, corpus:dict):
    words = sent.split()
    if not words:
        return 0.

    if n == 1:
        logp = np.sum([corpus[n][w] for w in words])
    else:
        logp = corpus[1][words[0]]

        for end in range(2, n):
            k_nm1 = " ".join(words[:end-1])
            k_n   = " ".join((k_nm1, words[end-1]))
            logp += corpus[end][k_n] - corpus[end-1][k_nm1]

        for end in range(n, len(words)):
            k_nm1 = " ".join(words[end-n:end-1])
            k_n   = " ".join((k_nm1, words[end-1]))
            logp += corpus[n][k_n] - corpus[n-1][k_nm1]

    return np.exp(logp)


def perplx(p, k):
    return np.power(p, -1/k)


def recursive_get(n, start, words, corpus, default=0.):
    if n <= 0:
        return default

    key = ' '.join(words[start:start+n])
    return corpus[n][key] if key in corpus[n] else \
        recursive_get(n-1, start+1, words, corpus, default=default)


def interploate(sent:str, lbds:tuple, corpus:dict):
    words = sent.split()
    if not words:
        return 0.

    calc_mat = [
        [recursive_get(n, end-n, words, corpus) /
         recursive_get(n-1, end-n, words, corpus, default=1)
         for n in range(1, len(lbds) + 1)]
        for end in range(1, len(words) + 1)
    ]

    interpolated_probs = np.dot(np.asarray(calc_mat),
                                np.flip(np.asarray(lbds)))

    return np.exp(np.sum(np.log(interpolated_probs)))


if __name__ == "__main__":
    occur = {
        1: {'하늘은': 3520000, '파랗고': 392000, '단풍잎은': 34600,
            '빨갛고': 339000, '은행잎은': 24300, '노랗고': 359000},
        2: {'하늘은 파랗고': 56100, '파랗고 단풍잎은': 23,
            '단풍잎은 빨갛고': 160, '빨갛고 은행잎은': 85,
            '은행잎은 노랗고': 198},
        3: {'하늘은 파랗고 단풍잎은': 34, '파랗고 단풍잎은 빨갛고': 0,
            '단풍잎은 빨갛고 은행잎은': 3, '빨갛고 은행잎은 노랗고': 85}
    }

    test = '하늘은 파랗고 단풍잎은 빨갛고 은행잎은 노랗고'

    # By smoothing
    smoothed = smooth(occur)
    p_all = [(n, prob_ngram(n, test, smoothed))
             for n in smoothed]

    print("Smoothed probability: ")
    for res in p_all:
        print(*res)

    # Perplx
    split = test.split()
    p_all_np = np.asarray([p for _, p in p_all])
    pplx = perplx(p_all_np, len(split))

    print("\nPerplexity: ")
    for n, res in enumerate(pplx, 1):
        print(n, res)

    # By interpolation
    no_smooth = {n: {k: v / N for k, v in corpus.items()}
                 for n, corpus in occur.items()}

    print("\nInterpolated probability: ")
    print(interploate(test, (0.5, 0.3, 0.2), no_smooth))
    print(interploate(test, (0.7, 0.2, 0.1), no_smooth))
