#!/usr/bin/env python3

from multiprocessing import Pool

def f(x):
    return "Trained"

def main():
    with Pool(5) as p:
        print(p.map(f, [i for i in range(10000)]))

if __name__ == '__main__':
    main()
