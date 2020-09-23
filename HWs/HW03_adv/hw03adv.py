import re

days = "월화수목금토일"
onlychar = re.compile(r"[a-zㄱ-ㅎㅏ-ㅣ가-힣]+")

if __name__ == "__main__":
    # 1. 요일 계산 프로그램
    today = days.index(input("오늘의 요일을 입력하세요: "))
    N = int(input("며칠 후의 요일을 계산할까요? "))
    day = days[(today + N) % 7]
    print(f"{N}일 후는 {day}요일입니다.")

    # 2. 단어 빈도 계산 프로그램
    sent = input("문장을 입력하세요: ")

    freq = dict()
    for match in onlychar.finditer(sent.lower()):
        word = match.group()
        freq[word] = freq.get(word, 0) + 1

    print(freq)
