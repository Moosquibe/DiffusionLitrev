#! /bin/bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd -P)"

#####################
# Bazel file linting
#####################
echo
echo "Linting Bazel"
echo -----------------
# Find all Bazel-ish files - these templates come from Buildifier's default search list
BAZEL_FILES=$(find ${REPO_ROOT} -type f \
            \(   -name "*.bzl" \
              -o -name "*.sky" \
              -o -name "BUILD.bazel" \
              -o -name "BUILD" \
              -o -name "*.BUILD" \
              -o -name "BUILD.*.bazel" \
              -o -name "BUILD.*.oss" \
              -o -name "MODULE.bazel" \
              -o -name "WORKSPACE" \
              -o -name "WORKSPACE.bazel" \
              -o -name "WORKSPACE.oss" \
              -o -name "WORKSPACE.*.bazel" \
              -o -name "WORKSPACE.*.oss" \) \
              -print)
BUILDIFIER_ARGS=("-lint=fix" "-mode=fix" "-v=false")
BUILDIFIER_INVOCATION="bazel run -- //tools/buildifier ${BUILDIFIER_ARGS[@]}"
echo $BAZEL_FILES | xargs ${BUILDIFIER_INVOCATION}

#################
# Python linting
#################
echo
echo "Linting Python imports"
echo ------------------------
bazel run -- //tools/isort ${REPO_ROOT} --dont-follow-links
echo
echo "Formatting Python code"
echo ------------------------
bazel run -- //tools/black ${REPO_ROOT}
# Ensure flake8 compliance
echo
echo "Linting Python code"
echo ---------------------
bazel run -- //tools/flake8 ${REPO_ROOT}
echo
echo "Doing static type checking"
echo ----------------------------
bazel run -- //tools/pyright ${REPO_ROOT}