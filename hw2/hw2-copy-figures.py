import shutil, errno

import os

def copyanything(src, dst):
    try:
        shutil.copytree(src, dst)
    except OSError as exc: # python >2.5
        if exc.errno == errno.ENOTDIR:
            shutil.copy(src, dst)
        else: raise


results = [988970911,
941530379,
349685028,
783952587,
156884050,
449777493,
983956549,
106841919,
994865007,
87401771,
782990781,
666671943,
944565074,
195339946,
312443606,
721505406,
41157021,
790121321,
805213998,
963255433]


root_src = os.path.join(os.path.dirname(__file__ ) + "\exp_figures\\")
root_dst = "d:\\good_exp_figures"

if not os.path.exists(root_dst):
	    os.makedirs(root_dst)

for dirname in results:
	src = root_src + "\\" + str(dirname) 
	dst = root_dst + "\\" + str(dirname)
	copyanything(src, dst)