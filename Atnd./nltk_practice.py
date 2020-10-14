import nltk, re
from typing import Iterable
# nltk.download("punkt")

sentence = "The quick brown fox jumps over the lazy dog"
nltk.word_tokenize(sentence)

def maxmatch(sentence:str, dictionary:Iterable):
    if not sentence:
        return []

    for i in range(len(sentence), 0, -1):
        word, rem = sentence[:i], sentence[i:]
        if word in dictionary:
            matches = maxmatch(rem, dictionary)
            matches.append(word)
            return matches

# print(maxmatch("아버지가방에들어가신다", ["아버지", "가방", "에", "가", "들어가신다", "방"]))

pattern = re.compile(r"(?:[가-힣]+대학 )?([가-힣]+학?[과부](?: [가-힣]+전공)?)")
# string = "서울대학교 자연과학대학 화학부 정누리"
string = "서울대학교 자연과학대학 물리천문학부 천문학전공 정누리"
# print(pattern.match(string))
# print(pattern.search(string).group(1))

string = "2018년 10월 31일 11시"
print(re.sub(r"[0-9]", "#", string))

string = "031-8800-2206"
print(re.sub(r"(0[02-9][0-9]?)-([0-9]{3,4}-[0-9]{4})", r"(\1) \2", string))
