# -*- coding: cp1251 -*-

import sys, getopt
from predict import predict

def main(argv):
    inputfile = "D:/Работа/Форпост, тестовое задание/1.jpg"
    outputfile = "D:/Работа/Форпост, тестовое задание/2.jpg"
    confidence_thresh = 0.6

    try:
        opts, args = getopt.getopt(argv, "hi:o:", ["ifile=", "ofile=", "conf="])
    except getopt.GetoptError:
        print('test.py -i <inputfile> -o <outputfile> -c <confidence_thresh>' )
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -i <inputfile> -o <outputfile>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
        elif opt in ("-c", "--conf"):
            confidence_thresh = 0.6
    print('Input file is "', inputfile)
    print('Predicting')
    predict(inputfile, outputfile, confidence_thresh)
    print('Saved in "', outputfile)


if __name__ == "__main__":
    main(sys.argv[1:])
