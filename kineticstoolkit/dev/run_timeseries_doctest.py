"""Run doctest on the complete package."""

import doctest
import os
import pydoc

import kineticstoolkit.lab as ktk

if __name__ == "__main__":  # pragma: no cover
    print("Testing TimeSeries Methods")

    with open("test.txt", "w") as fid:
        fid.write(">>> import kineticstoolkit.lab as ktk\n")
        fid.write(">>> import numpy as np\n")
        fid.write(">>> import pandas as pd\n")
        fid.write("\n\n")

        fid.write(pydoc.getdoc(ktk.TimeSeries))
        fid.write("\n\n")

        for _name, method in pydoc.allmethods(ktk.TimeSeries).items():
            fid.write(pydoc.getdoc(method))
            fid.write("\n\n")

    print(
        doctest.testfile(
            "test.txt", optionflags=doctest.NORMALIZE_WHITESPACE, report=True
        )
    )

    os.remove("test.txt")
