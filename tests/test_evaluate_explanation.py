from ground_truth_eval.evaluate_explanation import find_common_tokens


def test_find_common_tokens():
    pred_list = [
        ['man', 'walks', 'man'],
        ['man', 'man'],
        [],
    ]
    gt_list = [
        ['the', 'man', 'walks'],
        ['the', 'man', 'walks', 'man'],
        ['none'],
    ]
    answers = [
        ['man', 'walks'],
        ['man', 'man1'],
        [],
    ]

    for i, (p, g, a) in enumerate(zip(pred_list, gt_list, answers)):
        common = find_common_tokens(p, g)
        print(f'test #{i}:', 'passed' if set(a) == common else 'failed')


if __name__ == '__main__':
    test_find_common_tokens()