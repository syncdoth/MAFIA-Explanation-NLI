import pandas as pd
from utils.data_utils import get_token_rationales, get_contiguous_phrases


def test_get_token_rationales():
    df = pd.DataFrame({
        'Sentence1_marked_1': [
            'test sent without highlight!',
            '*highlight* with *punctuation.*',
            '',
            '*multiple* *highlights!*',
        ],
        'Sentence1_marked_2': [
            'test sent without highlight!',
            'highlight with *punctuation.* *highlight*',
            '',
            '*multiple,* *highlights!*',
        ],
        'Sentence1_marked_3': [
            'test sent *without* highlight!',
            'highlight with *punctuation.*',
            'empty',
            'multiple *highlights!*',
        ],
        'Sentence2_marked_1': [
            'test sent without highlight!',
            '*highlight* with *punctuation.*',
            '',
            '*multiple* *highlights!*',
        ],
        'Sentence2_marked_2': [
            'test sent without highlight!',
            'highlight with *punctuation.* *highlight*',
            '',
            '*multiple* *highlights!*',
        ],
        'Sentence2_marked_3': [
            'test sent *without* highlight!',
            'highlight with *punctuation.*',
            'empty',
            'multiple *highlights!*',
        ],
    })

    answers_df = pd.DataFrame({
        'Sentence1_vote': [[], ['punctuation'], [], ['multiple', 'highlights']],
        'Sentence2_vote': [[], ['punctuation'], [], ['multiple', 'highlights']],
        'Sentence1_union': [['without'], ['highlight', 'punctuation', 'highlight'], [],
                            ['multiple', 'highlights']],
        'Sentence2_union': [['without'], ['highlight', 'punctuation', 'highlight'], [],
                            ['multiple', 'highlights']],
    })
    get_token_rationales(df, 'union')
    get_token_rationales(df, 'vote')

    for i in range(len(df)):
        print(f'testing task {i+1}...', end='')
        out = df.iloc[i][[
            'Sentence1_vote', 'Sentence2_vote', 'Sentence1_union', 'Sentence2_union'
        ]]
        if (out != answers_df.iloc[i]).all():
            print('failed')
            print(f'the output was \n{out} \n while answer was \n{answers_df.iloc[i]}')
        else:
            print('passed!')


def test_get_contiguous_phrases():
    texts = [
        "I love eating food.",
        "I love eating food.",
        "I love eating food.",
        "I love eating food.",
        "I love eating food in the evening.",
        "I love eating food in the evening.",
    ]
    tokens = [
        ["I", "eating"],
        ["I", "food"],
        ["I", "love", "food"],
        ["I", "love", "eating"],
        ["I", "eating", "food", "the"],
        ["I", "the", "evening"],
    ]
    answers = [[['I'], ['eating']], [['I'], ['food']], [['I', 'love'], ['food']],
               [['I', 'love', 'eating']], [['I'], ['eating', 'food'], ['the']],
               [['I'], ['the', 'evening']]]

    for i, (text, token, answer) in enumerate(zip(texts, tokens, answers)):
        print(f"test #{i}:",
              "passed" if get_contiguous_phrases(text, token) == answer else "failed")


if __name__ == '__main__':
    test_get_token_rationales()
    test_get_contiguous_phrases()
