from unicodedata import name

# 모음으로 끝나는지 판정하기 위한 변수
VOWELS = set("AEIOU")

def ishangul(string):
    # 모두 'HANGUL'로 시작하는 경우만 한글로 된 string이다
    return True if all(name(char).startswith("HANGUL") for char in string) else False

def hasbatchim(syllable):
    # 한글이 아닐 경우 오류 발생
    if not ishangul(syllable):
        raise ValueError("한글 1문자를 입력하세요.")

    # 나머지 경우는 한글이다.
    # 이때 name() 함수가 돌려주는 string의 마지막에 해당 음절의 이름이 포함되어 있고,
    # 한글은 받침이 없는 경우 반드시 모음으로 끝나므로 마지막 문자가 모음인지만 확인하면 된다.
    return name(syllable)[-1] not in VOWELS

def get_ga(hangul):
    try: # hasbatchim() 함수는 오류를 낼 수 있으므로 try block 안에서 실행
        # 마지막 음절이 받침을 가지고 있으면 "이", 가지고 있지 않으면 "가"를 반환
        return "이" if hasbatchim(hangul[-1]) else "가"
    except ValueError:
        # ValueError가 발생했다면 한글이 아니므로 아무것도 반환하지 않음
        pass

# 다양하게 이용할 수 있는데, 대표적으로 계정 생성 등에서 이름을 입력받는 데 사용할 수 있겠다.

user_name = input("사용자의 이름을 입력하세요: ")
while not ishangul(user_name):
    user_name = input("이름은 모두 한글로 구성되어 있어야 합니다. 다시 입력하세요: ")

print(f"{user_name}{get_ga(user_name)} 가입하였습니다. 안녕하세요!")

# 좀 더 다양하게 사용하고 싶다면 get_ga() 함수를 확장하여 get_josa() 함수를 만들 수도 있다.
# get_josa() 함수는 받침이 있을 때 / 없을 때 사용할 조사를 순서대로 원소로 가지는
# tuple인 candidates를 keyword argument로 받는다. 사용해 보자.

def get_josa(hangul, candidates=("이", "가")):
    try:
        return candidates[0] if hasbatchim(hangul[-1]) else candidates[1]
    except ValueError:
        pass


address = input(f"{user_name}{get_josa(user_name, ('은', '는'))} 어디에 사시나요? ")

# 주소가 영문일 경우 조사를 붙이지 않음
print(f"{user_name}의 주소 {address}{get_josa(address, ('을', '를')) or ''} 저장했습니다.")
