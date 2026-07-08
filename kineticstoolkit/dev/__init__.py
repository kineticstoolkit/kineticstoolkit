#!/usr/bin/env python3
#
# Copyright 2020-2025 Félix Chénier

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Provide functions for Kinetics Toolkit development."""

__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2020-2025 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"


import doctest
import os
import pydoc
import shutil
import subprocess

import kineticstoolkit.config

# Module(s) in development
from kineticstoolkit.dev import kinetics  # noqa: F401 unused-import


def __dir__():
    return [
        "run_unit_tests",
        "run_extensions_tests",
        "run_style_formatter",
        "run_static_type_checker",
        "run_linter",
        "run_doc_tests",
        "run_sphinx",
        "run_tests",
    ]


def run_unit_tests() -> None:  # pragma: no cover
    """Run all unit tests."""
    # Run pytest in another process to ensure that the workspace is and stays
    # clean, and all Matplotlib windows are closed correctly after the tests.
    print("Running kineticstoolkit unit tests...")

    cwd = os.getcwd()
    os.chdir(kineticstoolkit.config.root_folder + "/tests")
    subprocess.call(
        [
            "coverage",
            "run",
            "--source",
            "../kineticstoolkit",
            "--omit",
            "../kineticstoolkit/external/*",
            "-m",
            "pytest",
            "--ignore=interactive",
        ],
        env=kineticstoolkit.config.env,
    )
    subprocess.call(["coverage", "html"], env=kineticstoolkit.config.env)
    print(
        "Result in \nfile://"
        + kineticstoolkit.config.root_folder
        + "/tests/htmlcov/index.html"
    )
    os.chdir(cwd)


def run_extensions_tests() -> None:  # pragma: no cover
    """Run all extension' unit tests."""
    print("Running kineticstoolkit_extensions unit tests...")

    from kineticstoolkit_extensions import root_folder  # noqa

    cwd = os.getcwd()
    os.chdir(root_folder)
    subprocess.call(
        [
            "coverage",
            "run",
            "--source",
            "../kineticstoolkit_extensions",
            "-m",
            "pytest",
        ],
        env=kineticstoolkit.config.env,
    )
    subprocess.call(["coverage", "html"], env=kineticstoolkit.config.env)
    print("Result in \nfile://" + root_folder + "/htmlcov/index.html")
    os.chdir(cwd)


def run_style_formatter() -> None:  # pragma: no cover
    """Run style formatter (ruff)."""
    subprocess.call(
        ["ruff", "format", kineticstoolkit.config.root_folder],
        env=kineticstoolkit.config.env,
    )


def run_static_type_checker() -> None:  # pragma: no cover
    """Run static typing checker (mypy)."""
    # Run pytest in another process to ensure that the workspace is and stays
    # clean, and all Matplotlib windows are closed correctly after the tests.
    print("Running mypy...")
    cwd = os.getcwd()
    os.chdir(kineticstoolkit.config.root_folder)
    subprocess.call(
        [
            "mypy",
            "--config-file",
            "kineticstoolkit/mypy.ini",
            "-p",
            "kineticstoolkit",
        ],
        env=kineticstoolkit.config.env,
    )
    os.chdir(cwd)


def run_linter() -> None:  # pragma: no cover
    """Run linter (ruff)."""
    subprocess.call(
        [
            "ruff",
            "check",
            "--fix",
            kineticstoolkit.config.root_folder + "/kineticstoolkit",
        ],
        env=kineticstoolkit.config.env,
    )


def run_doc_tests() -> None:  # pragma: no cover
    """Run all doc tests."""
    print("Running doc tests...")
    cwd = os.getcwd()
    os.chdir(kineticstoolkit.config.root_folder + "/kineticstoolkit/dev")

    write_doc_depth = [0]

    def write_doc(fid, instance_name):
        write_doc_depth[0] += 1
        fid.write(f"\n\n========\n{instance_name}\n========\n\n")

        try:
            ktk = __import__("kineticstoolkit")  # noqa: F841
            instance = eval(instance_name)
        except Exception as e:
            print(e)
            write_doc_depth[0] -= 1
            return

        fid.write(pydoc.getdoc(instance))

        if write_doc_depth[0] < 4:
            items = dir(instance)
            for item_name in items:
                if "__" not in item_name:
                    write_doc(fid, f"{instance_name}.{item_name}")

        write_doc_depth[0] -= 1

    with open("alldoc.txt", "w") as fid:
        fid.write(">>> import kineticstoolkit.lab as ktk\n")
        fid.write(">>> import numpy as np\n")
        fid.write(">>> import pandas as pd\n")
        write_doc(fid, "ktk")

    print(
        doctest.testfile(
            "alldoc.txt", optionflags=doctest.NORMALIZE_WHITESPACE, report=True
        )
    )

    # os.remove("alldoc.txt")
    os.chdir(cwd)


def run_sphinx() -> None:  # pragma: no cover
    """Generate the API doc."""
    print("Generating API documentation...")
    cwd = os.getcwd()
    os.chdir(kineticstoolkit.config.root_folder)
    try:
        shutil.rmtree("docs/api")
    except FileNotFoundError:
        pass
    try:
        shutil.rmtree("docs/_build")
    except FileNotFoundError:
        pass
    os.mkdir("docs/_build")
    try:
        os.symlink("kineticstoolkit", "ktk")
    except FileExistsError:
        pass
    subprocess.call(["sphinx-build", "-W", "docs", "docs/_build/html"])
    print(
        "Result in \nfile://"
        + kineticstoolkit.config.root_folder
        + "/docs/_build/html/index.html"
    )
    os.remove("ktk")
    os.chdir(cwd)


def run_tests() -> None:  # pragma: no cover
    """Run all testing and building functions."""
    run_style_formatter()
    run_doc_tests()
    run_linter()
    run_static_type_checker()
    run_unit_tests()
    run_extensions_tests()
    run_sphinx()
