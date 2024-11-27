# Copyright 2018 The PECOS Developers
# Copyright 2018 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract
# DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

import subprocess
from pathlib import Path


def main():
    # See if Cython has been installed...

    current_location = Path.parent(Path.resolve(__file__))

    cython_dirs = [
        "cysparsesim",
    ]
    failed = {}

    for d in cython_dirs:
        path = Path(current_location / d)

        p = subprocess.Popen(  # noqa: S602
            "python setup.py build_ext --inplace",  # noqa: S607
            shell=True,
            cwd=path,
            stderr=subprocess.PIPE,
        )
        p.wait()
        _, error = p.communicate()

        if p.returncode:
            failed[d] = error

    return failed, cython_dirs


if __name__ == "__main__":
    failed, cython_dirs = main()

    successful = set(cython_dirs) - set(failed.keys())

    if successful:
        print(f"\nSUCCESSFUL COMPILATION ({len(successful)}/{len(cython_dirs)}):")
        for c in cython_dirs:
            if c in successful:
                print(c)

    if failed:
        print(f"\nFAILED ({len(failed)}/{len(cython_dirs)}):")

        for f, error in failed.items():
            print("--------------")
            print('Cython package "%s" failed to compile!' % f)

            print("\nError:\n")
            print(error.decode())
            print("--------------")

        print("\nRecommend compiling separately those that failed.")
