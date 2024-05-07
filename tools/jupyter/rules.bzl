"""
Build rule for jupyter notebooks
"""
load("@rules_python//python:defs.bzl", "py_library", "py_binary")
load("@pip//:requirements.bzl", "requirement")

def jupyter_notebook(name, deps = None, data = None):
    
    notebook_url_prefix = "lab/tree"

    py_library(
        name = "{0}_lib".format(name),
        deps = deps,
        data = data
    )

    py_binary(
        name = name,
        main = "jupyterlab.py",
        srcs = ["jupyterlab.py"],
        args = [
            "--LabApp.default_url={0}/$(location {1}.ipynb)".format(notebook_url_prefix, name),
            "-h",
        ],
        data = ["{0}.ipynb".format(name)],
        deps = [
            "{0}_lib".format(name),
            requirement("jupyterlab"),
            #"//tools/jupyter/jupyter_utils",
        ]
    )