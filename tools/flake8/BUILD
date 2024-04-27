load("@pip//:requirements.bzl", "requirement")
load("@rules_python//python:defs.bzl", "py_binary")

py_binary(
    name = "flake8",
    srcs = ["main.py"],
    main = "main.py",
    deps = [
        requirement("flake8"),
    ],
)
