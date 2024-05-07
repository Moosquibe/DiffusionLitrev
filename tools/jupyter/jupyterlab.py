#import os
#import stat
import sys

from jupyterlab.labapp import main

#python_wrapper_path = os.path.abspath("hermetic_python.sh")

#with open(python_wrapper_path, "w") as f:
#    f.write(f"""#!/bin/bash\ncd "{os.getcwd()}"\nexec "{sys.executable}" -S "$@" """)
#os.chmod(python_wrapper_path, os.stat(python_wrapper_path).st_mode | stat.S_IEXEC)
#sys.executable = python_wrapper_path

if __name__ == "__main__":
    sys.exit(main())