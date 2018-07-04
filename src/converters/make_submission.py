import sys
sys.path.append('../../')
from utils.donal_args import args
path_1 = sys.argv[1]
path_2 = sys.argv[2]
f1 = open(path_1)
f2 = open(path_2)
f = open('submission.csv','wb')

f.write('aid,uid,score\n')
f1.readline()
for line in f1:
    line = line.strip() +','+ f2.readline()
    f.write(line)
