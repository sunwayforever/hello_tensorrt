#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 2022-07-05 14:17
import sys
import subprocess

app =sys.argv[1][0:-4]
print(f"testing {app}: ", end = "", flush = True)
a = subprocess.run(f"{app}.elf", stdout=subprocess.PIPE, stderr = subprocess.PIPE)
a = a.stdout.decode("utf-8").splitlines()[-2:]
a = a[0] + a[1]

f = open(f"test/{app}.gold")
b = f.read().splitlines()
b = b[0] + " " + b[1]

for (x, y) in zip(a.split(), b.split()):
    if abs(float(x) - float(y)) > 1e-4:
        print(f"\x1b[31mFAIL\x1b[0m")
        sys.exit(0)
print(f"\x1b[32mPASS\x1b[0m")
