# Copyright (c) 2023 by Microsoft Corporation.
# Licensed under the MIT license.


"""Setup scripts."""

import pathlib
import sys
from setuptools import find_packages, setup

if len(sys.argv) <= 1:
    sys.argv += ["install", "--user"]

root_path = pathlib.Path(__file__).parent.absolute()


def install():
    setup(
        name="parrot",
        version="0.1",
        author="Chaofan Lin",
        package_dir={"": "."},
        packages=find_packages("."),
    )


print("Installing Parrot ...")
install()
