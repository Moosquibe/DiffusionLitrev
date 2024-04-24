load("@rules_python//python:defs.bzl", "py_binary")
load("@rules_python//python:pip.bzl", "compile_pip_requirements")


py_binary(
    name = "train",
    srcs = ["train.py"],
    deps = [
        "@pypi//torch:pkg",
        ":layers"
    ]
)

py_library(
    name = "layers",
    srcs = ["srcs/modules/multi_scale_convolution.py"],
    deps = [
        "@pypi//torch:pkg",
        "@pypi//numpy:pkg"
    ]
)

compile_pip_requirements(
    name = "requirements",
    src = "requirements.in",
    requirements_txt = "requirements.txt",
    requirements_windows = "requirements_windows.txt",
)



