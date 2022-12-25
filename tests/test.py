'''
@Time    : 2022/11/11 9:53
@Author  : leeguandon@gmail.com
'''


import io
import json
import argparse
# from VideoClipSeo import apply


def parse_args():
    parser = argparse.ArgumentParser(description='inert and text_resize')
    parser.add_argument('--user', default=r'E:\comprehensive_library\mmaction2_add\tests\input.json')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # profile.run("main()")
    args = parse_args()

    with io.open(args.user, 'r', encoding='UTF-8') as psd_data:
        input_dict = json.load(psd_data)

    pass





