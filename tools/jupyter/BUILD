load("@rules_python//python:defs.bzl", "py_binary")
load("@pip//:requirements.bzl", "requirement")

py_binary(
    name = "jupyterlab",
    main = "jupyterlab.py",
    srcs = ["jupyterlab.py"],
    deps = [
        requirement("jupyterlab"),
    ],
    visibility = ["//visibility:public"],
)