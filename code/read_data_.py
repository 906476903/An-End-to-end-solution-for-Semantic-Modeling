# -*- coding: utf-8 -*-
import os
import random
import time
import numpy as np
from PIL import Image
import json
import sys
reload(sys)
sys.setdefaultencoding( "utf-8" )



dict = {}

input = open("./public/test.json","r")
input = list(input)
length = len(input)
for i in range(length):
    dict = json.loads(input[i])
    f = open("test.txt",'a')
    f.write(str(dict["content"]).replace('\r','').replace('\n','').replace('\t','') + '\n')
    f = open("test_index.txt",'a')
    f.write(str(dict["id"]) + '\n')
