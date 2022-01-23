import numpy as np
from os import listdir
from os.path import isfile, join


def compare(gold, test):
    dif = test - gold
    dif_sum = np.sum(dif)
    dif_abs_sum = np.sum(np.abs(dif))
    return dif, dif_sum, dif_abs_sum


def compare_existing_cpp_output_with_gold():
    BASE_DIR_CPP = '../dumps/inference_outputs_cpp/'
    BASE_DIR_GOLD = '../dumps/inference_outputs/'

    cpp_outputs = [f for f in listdir('../dumps/inference_outputs_cpp') if
                   isfile(join('../dumps/inference_outputs_cpp', f))]

    for cpp in cpp_outputs:
        gold = np.load(BASE_DIR_GOLD + cpp)
        test = np.load(BASE_DIR_CPP + cpp)
        dif, dif_sum, dif_sum_abs = compare(gold, test)
        print(cpp, ": ")
        print("dif sum: ", dif_sum)
        print("dif sum abs: ", dif_sum_abs)

        print(gold[0,0,0:32,0])
        print(test[0,0,0:32,0])


compare_existing_cpp_output_with_gold()
