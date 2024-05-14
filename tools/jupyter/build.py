import sys

from jupyterlab.commands import build
from jupyterlab.coreconfig import CoreConfig

def build_jupyterlab():
    # Create a CoreConfig object if you need to specify config options
    config = CoreConfig()
    
    # Use the build function with the specified config
    try:
        build_result = build()
        if build_result.status == 'ok':
            print("JupyterLab built successfully!")
        else:
            print("Build failed with status:", build_result.status)
    except Exception as e:
        print("An error occurred during the build:", str(e))

if __name__ == "__main__":
    sys.exit(build_jupyterlab())