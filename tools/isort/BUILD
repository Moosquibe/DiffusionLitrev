load("@pip//:requirements.bzl", "requirement")
load("@rules_python//python:defs.bzl", "py_binary")

py_binary(
    name = "isort",
    srcs = ["main.py"],
    main = "main.py",
    deps = [
        requirement("isort"),
    ],
)
