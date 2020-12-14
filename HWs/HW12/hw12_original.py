#!/usr/bin/env python3
# coding: utf-8

import os, sys, re
from functools import partial
import nltk, numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns

from sklearn import metrics
from textstat import textstat

from nltk.stem.porter import PorterStemmer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec, FastText
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VS

stemmer = PorterStemmer()
sentiment_analyzer = VS()

space_pattern = re.compile(r"\s+")
url_pattern = re.compile(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|"
                         r"[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+")
mention_pattern = re.compile(r"@[\w\-]+")
hashtag_pattern = re.compile(r"#([\w\-]+)")
emoji_pattern = re.compile(r'&#([0-9A-Fa-f]+);')

word_pattern = re.compile("[a-z0-9_]+")
word_pattern_puncts = re.compile("[a-z.,!?0-9_]+")


def preprocess(text_string, url_repl="", mention_repl="",
               hashtag_repl=r"\1", emoji_repl=r"\1"):
    """
    Accepts a text string and replaces:
    1) lots of whitespace with one space
    2) urls with url_repl
    3) mentions with mention_repl
    4) hashtags with hashtag_repl

    This allows us to get standardized counts of urls and mentions
    Without caring about specific people mentioned
    """
    parsed_text = space_pattern.sub(" ", text_string)
    parsed_text = url_pattern.sub(url_repl, parsed_text)
    parsed_text = mention_pattern.sub(mention_repl, parsed_text)
    parsed_text = hashtag_pattern.sub(hashtag_repl, parsed_text)
    parsed_text = emoji_pattern.sub(emoji_repl, parsed_text)

    return parsed_text

def tokenize(tweet):
    """Removes punctuation & excess whitespace, sets to lowercase,
    and stems tweets. Returns a list of stemmed tokens."""
    tokens = [stemmer.stem(t)
              for t in word_pattern.findall(tweet.lower())]
    return tokens

def basic_tokenize(tweet):
    """Same as tokenize but without the stemming"""
    return word_pattern_puncts.findall(tweet.lower())

def count_twitter_objs(text_string):
    """
    Accepts a text string and replaces:
    1) urls with URLHERE
    2) lots of whitespace with one instance
    3) mentions with MENTIONHERE
    4) hashtags with HASHTAGHERE
    4) emojis with EMOJIHERE

    This allows us to get standardized counts of urls and mentions
    Without caring about specific people mentioned.

    Returns counts of urls, mentions, and hashtags.
    """
    parsed_text = preprocess(text_string, "URLHERE", "MENTIONHERE",
                             "HASHTAGHERE", "EMOJIHERE")
    return (parsed_text.count("URLHERE"),
            parsed_text.count("MENTIONHERE"),
            parsed_text.count("HASHTAGHERE"),
            parsed_text.count("EMOJIHERE"))

def other_features(tweet):
    """This function takes a string and returns a list of features.
    These include Sentiment scores, Text and Readability scores,
    as well as Twitter specific features"""
    sentiment = sentiment_analyzer.polarity_scores(tweet)

    text = preprocess(tweet)  # Get text only
    words = text.split()

    syllables = textstat.syllable_count(text)
    num_chars = len(text)
    num_chars_total = len(tweet)
    num_terms = len(tweet.split())
    num_words = len(words)
    avg_syl = round(float((syllables+0.001)) / float(num_words+0.001), 4)
    num_unique_terms = len(set(words))

    ###Modified FK grade, where avg words per sentence is just num words/1
    FKRA = round(float(0.39 * float(num_words)/1.0) +
                 float(11.8 * avg_syl) - 15.59, 1)
    ##Modified FRE score, where sentence fixed to 1
    FRE = round(206.835 - 1.015*(float(num_words)/1.0) -
                (84.6*float(avg_syl)), 2)

    twitter_objs = count_twitter_objs(tweet)
    retweet = 0
    if "rt" in text:
        retweet = 1

    features = [FKRA, FRE, syllables, avg_syl, num_chars, num_chars_total,
                num_terms, num_words, num_unique_terms, sentiment["neg"],
                sentiment["pos"], sentiment["neu"], sentiment["compound"],
                *twitter_objs, retweet]
    return features

def get_feature_array(tweets):
    feats = []
    for t in tweets:
        feats.append(other_features(t))
    return np.array(feats)

def embed_tfidf(tweets, ngram_range=(1, 3), norm=None,
                decode_error="replace", min_df=5, max_df=0.75, **kwargs):
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        norm=norm,
        decode_error=decode_error,
        min_df=min_df,
        max_df=max_df,
        **kwargs
    )

    #Construct tfidf matrix and get relevant scores
    # vocab = {v: i for i, v in enumerate(vectorizer.get_feature_names())}
    return vectorizer.fit_transform(tweets).toarray()

def main(embedby=embed_tfidf, max_features=(10000, 5000), nsplits=5):
    df = pd.read_csv("train.txt", "\t", header=0, names=["class", "tweet"])
    print(df.describe())
    print(df.columns)
    # df["class"].hist()
    df["class"] = df["class"].apply(lambda x: 0 if x == "hate" else 1
                                    if x == "offensive" else 2)

    tweets = df.tweet
    stopwords = nltk.corpus.stopwords.words("english")
    other_exclusions = ["#ff", "ff", "rt"]
    stopwords.extend(other_exclusions)

    # np.ndarray of shape(len(tweets), embedding_dimension)
    matrix = embedby(
        tweets,
        tokenizer=basic_tokenize,
        preprocessor=preprocess,
        stop_words=stopwords,
        use_idf=True,
        smooth_idf=False,
        lowercase=False,
        max_features=max_features[0]
    )
    # matrix = embedby(
    #     tweets,
    #     preprocessor=preprocess,
    #     use_idf=True,
    #     smooth_idf=False,
    #     analyzer="char",
    #     lowercase=False,
    #     ngram_range=(2, 20),
    #     max_features=max_features[0]
    # )

    #Get POS tags for tweets and save as a string
    tweet_tags = []
    for t in tweets:
        tokens = basic_tokenize(preprocess(t))
        tags = nltk.pos_tag(tokens)
        tag_list = [x[1] for x in tags]
        tag_str = " ".join(tag_list)
        tweet_tags.append(tag_str)

    # np.ndarray of shape(len(tweets), embedding_dimension)
    pos_matrix = embedby(
        pd.Series(tweet_tags),
        tokenizer=None,
        lowercase=False,
        preprocessor=None,
        stop_words=None,
        use_idf=False,
        smooth_idf=False,
        max_features=max_features[1]
    )

    # other_features_names = ["FKRA", "FRE", "num_syllables", "avg_syl_per_word", "num_chars",
    #                         "num_chars_total", "num_terms", "num_words", "num_unique_words",
    #                         "vader neg", "vader pos", "vader neu", "vader compound",
    #                         "num_hashtags", "num_mentions", "num_urls", "is_retweet"]
    feats = get_feature_array(tweets)

    #Now join them all up
    M = np.concatenate([matrix, pos_matrix, feats], axis=1)
    print(M.shape)

    #Finally get a list of variable names
    # variables = [None] * len(vocab)
    # for k, v in vocab.items():
    #     variables[v] = k

    # pos_variables = [None] * len(pos_vocab)
    # for k, v in pos_vocab.items():
    #     pos_variables[v] = k

    # feature_names = variables + pos_variables + other_features_names

    X = pd.DataFrame(M)
    y = df["class"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    pipe = Pipeline(
        [("select", SelectFromModel(
            LogisticRegression(class_weight="balanced", penalty="l1", C=0.01,
                               solver="liblinear"),
            threshold="median")),
         ("model", LogisticRegression(class_weight="balanced", penalty="l2",
                                      solver="liblinear"))]
    )
    grid_search = GridSearchCV(
        pipe,
        {},
        n_jobs=min(nsplits, os.cpu_count()),
        cv=StratifiedKFold(n_splits=nsplits, random_state=42).split(X_train, y_train),
        verbose=2
    )
    model = grid_search.fit(X_train, y_train)

    y_preds = model.predict(X_test)
    report = metrics.classification_report(y_test, y_preds)
    print(report)

    confusion_matrix = metrics.confusion_matrix(y_test, y_preds)
    matrix_proportions = np.zeros((3, 3))
    for i in range(0, 3):
        matrix_proportions[i, :] = confusion_matrix[i, :] / \
            float(confusion_matrix[i, :].sum())
    names = ["Hate", "Offensive", "Neither"]
    confusion_df = pd.DataFrame(matrix_proportions, index=names, columns=names)

    plt.rc("pdf", fonttype=42)
    plt.rcParams["ps.useafm"] = True
    plt.rcParams["pdf.use14corefonts"] = True
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.serif"] = "Times"
    plt.rcParams["font.family"] = "serif"

    fig = plt.figure(figsize=(5, 5))
    sns.heatmap(confusion_df, annot=True, annot_kws={"size": 12},
                cmap="gist_gray_r", cbar=False, square=True, fmt=".2f")
    plt.ylabel(r"\textbf{True categories}", fontsize=14)
    plt.xlabel(r"\textbf{Predicted categories}", fontsize=14)
    plt.tick_params(labelsize=12)
    plt.show()

    fig.savefig("confusion_matrix.png", dpi=150)

    # y.hist()
    # plt.show()
    # pd.Series(y_preds).hist()
    # plt.show()


if __name__ == "__main__":
    main(embed_tfidf, nsplits=int(sys.argv[-1]))

#### UNUSED ####

# def embed_gensim(embedder, tweets, lowercase=True, tokenizer=None, preprocessor=None,
#                  analyzer="word", stop_words=None, ngram_range=(1, 3),
#                  max_features=100, min_df=5, seed=42, sg=1, hs=0, **_):
#     processed_tweets = list()
#     for tweet in tweets:
#         if preprocessor:
#             tweet = preprocessor(tweet)

#         if lowercase:
#             tweet = tweet.lower()

#         if tokenizer:
#             tweet = tokenizer(tweet)
#         elif analyzer == "word":
#             tweet = tweet.split()
#         elif analyzer == "char":
#             tweet = list(tweet)

#         if stop_words and analyzer == "word":
#             stopwords_set = set(stop_words)
#             tweet = [term for term in tweet if term not in stopwords_set]

#         processed_tweets.append(tweet)

#     vectorizer = embedder(processed_tweets, size=max_features, window=ngram_range[-1],
#                           min_count=min_df, seed=seed, workers=os.cpu_count(),
#                           sg=sg, hs=hs)

#     matrix = list()
#     for tweet in processed_tweets:
#         embedding = np.empty((len(tweet), max_features))
#         for idx, word in enumerate(tweet):
#             embedding[idx] = vectorizer.wv.get_vector(word) \
#                 if word in vectorizer.wv.vocab else 0.0
#         matrix.append(embedding.sum(axis=0))

#     matrix = np.asarray(matrix)
#     norms = np.linalg.norm(matrix, axis=1)
#     norms[norms == 0] = 1.0
#     return matrix / norms.reshape(matrix.shape[0], 1)

# embed_word2vec = partial(embed_gensim, Word2Vec)
# embed_fasttext = partial(embed_gensim, FastText)
