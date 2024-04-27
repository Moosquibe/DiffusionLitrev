load("@npm//:defs.bzl", "npm_link_all_packages")
load("@pip//:requirements.bzl", "requirement")
load("@rules_python//python:defs.bzl", "py_binary")
load("@rules_python//python:pip.bzl", "compile_pip_requirements")

py_binary(
    name = "train",
    srcs = ["train.py"],
    deps = [
        requirement("torch"),
        requirement("pyright"),
        ":layers",
    ],
)

py_library(
    name = "layers",
    srcs = ["srcs/modules/multi_scale_convolution.py"],
    deps = [
        requirement("torch"),
        requirement("numpy"),
    ],
)

compile_pip_requirements(
    name = "requirements",
    src = "requirements.in",
    requirements_txt = "requirements.txt",
    requirements_windows = "requirements_windows.txt",
)

npm_link_all_packages(name = "node_modules")
