load("@pip//:requirements.bzl", "requirement")
load(":rules.bzl", "jupyter_notebook")

exports_files([
    "jupyterlab.py",
])

py_binary(
    name = "build_jupyterlab",
    main = "build.py",
    srcs = ["build.py"],
    deps = [
        requirement("jupyterlab")
    ]
)

jupyter_notebook(
    name = "empty_notebook",
    deps = [
        ":build_jupyterlab",
        requirement("numpy")
    ]
)

#py_library(
#    name = "jupyter_utils"
#    srcs = [
#        "jupyter_utils.py"
#    ]
#)