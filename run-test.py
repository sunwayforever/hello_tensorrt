#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 2022-07-05 14:17
import sys
import subprocess

app =sys.argv[1][0:-4]
print(f"testing {app}")
a = subprocess.run(f"{app}.elf", stdout=subprocess.PIPE, stderr = subprocess.PIPE)
a = a.stdout.decode("utf-8").splitlines()[-2:]
a = a[0] + a[1]

f = open(f"test/{app}.gold")
b = f.read().splitlines()
b = b[0] + " " + b[1]

for (x, y) in zip(a.split(), b.split()):
    if abs(float(x) - float(y)) > 1e-4:
        print("FAIL")
        sys.exit(-1)
print("PASS")
