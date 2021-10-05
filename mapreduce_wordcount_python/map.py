#!/usr/local/bin/python

import sys
import time

for line in sys.stdin:
    ss = line.strip().split(' ')
    for s in ss:
        #time.sleep(100000)
        if s.strip() != "":
            print "%s\t%s" % (s, 1)

