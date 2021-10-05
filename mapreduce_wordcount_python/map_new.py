import sys
import time

for line in sys.stdin:
    ss = line.strip().split(' ')
    #time.sleep(100000)
    for word in ss:
        print('\t'.join([word.strip(), '1']))
