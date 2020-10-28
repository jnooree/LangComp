# -*- coding: utf-8 -*-
# title: eliza.py
# course: Language and Computer
# author(s): Suzi Park, 5조: 김호정 박지희 오재원 정누리
# date created: 2020-10-14
# description: ELIZA

import re, random


def no_dup(pool, last):
    if len(pool) > 1:
        candids = random.sample(pool, 2)
        return candids[0] if candids[0] != last else candids[1]
    else:
        return pool[0]


def me_to_you(sent):
    sent = re.sub(r'\bME\b', 'YOU', sent)
    return re.sub(r'\bMY(SELF)?\b', r'YOUR\1', sent)


def respond(message, last_response=None):
    try:
        # preprocessing
        message = re.sub(r'\s+', ' ', message)
        message = message.rstrip(".!?")
        message = message.upper()

        # "I'm sad"
        match = re.match(r".*\bI(?: AM|'M) (DEPRESSED|SAD|SICK|UNHAPPY)\b.*",
                         message)
        if match:
            sad = match.group(1)
            responses = [
                f"I AM SORRY TO HEAR THAT YOU ARE {sad}",
                f"WHY DO YOU THINK YOU ARE {sad}",
                f"DO YOU THINK COMING HERE WILL HELP YOU NOT TO BE {sad}"
            ]
            return no_dup(responses, last_response)

        # "need"
        match = re.match(r".*\b([^ ]+) NEED(S?) ([A-Z ']+)",
                         message)
        if match:
            subj = match.group(1)
            third = match.group(2)
            trailing = match.group(3)

            you_and_I = ["YOU", "I"]
            for i, test in enumerate(you_and_I):
                if subj == test:
                    subj = you_and_I[i-1]
                    break

            pronouns_diff = {"I": "ME", "HE": "HIM", "SHE": "HER",
                             "WE": "US", "THEY": "THEM"}
            obj = pronouns_diff.get(subj, subj)

            responses = [
                f"WHAT WOULD IT MEAN TO YOU IF {subj} GOT {trailing}",
                f"WHY DO{'ES' if third else ''} {subj} NEED {trailing}",
                f"WHAT IF {subj} NEVER GOT {trailing}",
                f"WHAT WOULD GETTING {trailing} MEAN TO {obj}",
                f"WHAT DOES NEEDING {trailing} "
                "HAVE TO DO WITH THIS DISCUSSION"
            ]
            return no_dup(responses, last_response)

        # "be like"
        match = re.match(r".*\b(BE(?:EN)?|AM|IS|ARE|WAS|WERE"
                         r"|(?:[A-Z]+'(?:M|S|RE))) LIKE\b.*",
                         message)
        if match:
            return "WHAT RESEMBLANCE DO YOU SEE"

        # you (are|do)( not|n't)? (.*)
        match = re.match(r".*?\bYOU (ARE|WERE|DO|DID)( NOT|N'T)? "
                         r"(.+?)(?:(?:AND|BUT|OR|SO|FOR|YET)\b|$)",
                         message)

        if match:
            trailing = me_to_you(match.group(3))

            if match.group(1).startswith("D"):
                trailing = f"{match.group(1)}{match.group(2)} " \
                           f"{trailing}"
            else:
                tense = {"ARE": "AM", "WERE": "WAS"}
                trailing = f"{tense[match.group(1)]}" \
                           f"{' NOT' if match.group(2) else ''} {trailing}"

            responses = [
                f"WHY DO YOU THINK I {trailing}",
                f"DOES IT PLEASE YOU TO BELIEVE I {trailing}",
                f"WHAT MAKES YOU THINK I {trailing}"
            ]
            return no_dup(responses, last_response)

        # "my"
        match = re.match(r".*\bMY (.*)", message)
        if match:
            trailing = me_to_you(match.group(1))

            responses = [f'YOUR {trailing}']

            family_match = re.match(r"^((?:FATHER|MOTHER|MOM|DAD"
                                    r"|SISTER|BROTHER|WIFE|HUSBAND"
                                    r"|SON|DAUGHTER|CHILD(?:REN)?)S?)\b(.*)",
                                    trailing)
            if family_match:
                responses.append("TELL ME MORE ABOUT YOUR FAMILY")
                if family_match.group(2):
                    responses.extend([
                        f"WHO ELSE IN YOUR FAMILY{family_match.group(2)}",
                        f"WHAT ELSE COMES TO MIND WHEN YOU "
                        f"THINK OF YOUR {family_match.group(1)}"
                    ])

            return no_dup(responses, last_response)

        # generalization
        if re.match(r".*\bALL\b.*", message):
            return 'IN WHAT WAY'
        elif re.match(r".*\bALWAYS\b.*", message):
            return 'CAN YOU THINK OF A SPECIFIC EXAMPLE'

    except Exception:
        pass

    finally:
        # no keyword
        responses = [
            "PLEASE GO ON",
            "WHAT DOES THAT SUGGEST TO YOU",
            "I'M NOT SURE I UNDERSTAND YOU FULLY"
        ]
        return no_dup(responses, last_response)


if __name__ == '__main__':
    with open("eliza.txt") as f:
        data = f.read().splitlines()

    last_response = None
    for line in data[::2]:
        print(line)
        last_response = respond(line.replace("’", "'"), last_response)
        print(last_response)
