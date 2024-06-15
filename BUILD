load("@pip//:requirements.bzl", "requirement")
load("@rules_python//python:defs.bzl", "py_binary")
load("@rules_python//python:pip.bzl", "compile_pip_requirements")

py_binary(
    name = "scratch",
    srcs = ["src/scratch.py"],
    deps = [
        requirement("torch"),
        requirement("pyright"),
        ":modules",
    ],
)

py_library(
    name = "modules",
    srcs = ["srcs/modules/multi_scale_convolution.py"],
    deps = [
        requirement("torch"),
        requirement("numpy"),
    ],
)

py_library(
    name = "models",
    srcs = ["srcs/models/pdm.py"],
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

alias(
    name = "jupyterlab",
    actual = "//tools/jupyter:jupyterlab",
)
