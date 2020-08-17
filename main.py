# -*- coding: cp1251 -*-

import sys, getopt
from predict import predict

def main(argv):
    inputfile = "D:/������/�������, �������� �������/1.jpg"
    outputfile = "D:/������/�������, �������� �������/2.jpg"
    try:
        opts, args = getopt.getopt(argv, "hi:o:", ["ifile=", "ofile="])
    except getopt.GetoptError:
        print('test.py -i <inputfile> -o <outputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -i <inputfile> -o <outputfile>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
    print('Input file is "', inputfile)
    print('Predicting')
    predict(inputfile, outputfile)
    print('Saved in is "', outputfile)


if __name__ == "__main__":
    main(sys.argv[1:])
