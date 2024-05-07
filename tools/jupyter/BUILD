load("@pip//:requirements.bzl", "requirement")
load(":rules.bzl", "jupyter_notebook")

exports_files([
    "jupyterlab.py",
])


jupyter_notebook(
    name = "empty_notebook",
    deps = [
        requirement("numpy")
    ]
)

#py_library(
#    name = "jupyter_utils"
#    srcs = [
#        "jupyter_utils.py"
#    ]
#)