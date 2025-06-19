#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
"""
Kinetics Toolkit
================

To get started, please consult Kinetics Toolkit's
[website](https://kineticstoolkit.uqam.ca)

>>> import kineticstoolkit as ktk
"""
__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2020-2025 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"


# Import classes
from kineticstoolkit.timeseries import TimeSeries, TimeSeriesEvent  # noqa
from kineticstoolkit.player import Player as Player  # noqa

# Import functions
from kineticstoolkit.tools import change_defaults  # noqa
from kineticstoolkit.files import load, save, read_c3d, write_c3d  # noqa
from kineticstoolkit import _repr  # noqa

# Import modules
from kineticstoolkit import filters  # noqa
from kineticstoolkit import kinematics  # noqa
from kineticstoolkit import cycles  # noqa
from kineticstoolkit import doc  # noqa
from kineticstoolkit import gui  # noqa
from kineticstoolkit import geometry  # noqa
from kineticstoolkit import dev  # noqa
from kineticstoolkit import config  # noqa

# Import extensions
try:
    import kineticstoolkit_extensions as ext
except ModuleNotFoundError:
    pass
except Exception as e:
    raise RuntimeError(
        "The following error has been raised when trying to import the "
        "package `kineticstoolkit_extensions`. Please ensure that both "
        "`kineticstoolkit` and `kineticstoolkit_extensions` are up to date. "
        "If this error persists, please uninstall "
        "`kineticstoolkit_extensions` and report this issue on the issue "
        "tracker at "
        "https://github.com/kineticstoolkit/kineticstoolkit/issues \n\n"
        f"{e}"
    )


def __dir__():
    return [
        "TimeSeries",
        "TimeSeriesEvent",
        "Player",
        "load",
        "save",
        "read_c3d",
        "write_c3d",
        "import_extensions",
        "filters",
        "kinematics",
        "cycles",
        "doc",
        "geometry",
        "ext",
        "change_defaults",
    ]


# Check if a serious warning has been issued for this version.
try:
    from requests_cache import CachedSession  # noqa
    from datetime import timedelta  # noqa
    import json  # noqa
    import warnings  # noqa

    session = CachedSession(
        "kineticstoolkit",
        backend="filesystem",
        use_temp=True,
        expire_after=timedelta(hours=24),
    )
    res = session.get(
        "https://kineticstoolkit.uqam.ca/api/import_check.php",
        params={"version": config.version},
    )
    contents = json.loads(res.content)
    if res.ok and "warning" in contents:
        warnings.warn(contents["warning"])
except Exception:
    pass


if __name__ == "__main__":  # pragma: no cover
    import doctest

    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE)
