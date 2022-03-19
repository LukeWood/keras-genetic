# Copyright 2022 Luke Wood
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Setup script."""

import pathlib

from setuptools import find_packages
from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

setup(
    name="keras-genetic",
    description="Train keras models with genetic algorithms.",
    version="0.0.1",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/lukewood/keras-genetic",
    author="Luke Wood",
    author_email="lukewoodcs@gmail.com",
    license="Apache License 2.0",
    install_requires=[],
    extras_require={
        "tests": ["flake8", "isort", "black", "pytest"],
    },
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.7",
        "Operating System :: Unix",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Topic :: Software Development",
    ],
    packages=find_packages(exclude=("*_test.py",)),
)
