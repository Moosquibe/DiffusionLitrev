load("@pip//:requirements.bzl", "requirement")
load("@rules_python//python:defs.bzl", "py_binary")

py_binary(
    name = "jupyterlab",
    srcs = ["jupyterlab.py"],
    main = "jupyterlab.py",
    visibility = ["//visibility:public"],
    deps = [
        requirement("jupyterlab"),
        requirement("matplotlib"),
        requirement("numpy"),
        requirement("seaborn"),
        requirement("torch"),
        requirement("torchvision"),
        requirement("tqdm"),
    ],
)
