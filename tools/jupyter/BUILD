load("@rules_python//python:defs.bzl", "py_binary")
load("@pip//:requirements.bzl", "requirement")

py_binary(
    name = "jupyterlab",
    main = "jupyterlab.py",
    srcs = ["jupyterlab.py"],
    deps = [
        requirement("jupyterlab"),
        requirement("matplotlib"),
        requirement("numpy"),
        requirement("seaborn"),
        requirement("torch"),
        requirement("torchvision"),
        requirement("tqdm")
    ],
    visibility = ["//visibility:public"],
)