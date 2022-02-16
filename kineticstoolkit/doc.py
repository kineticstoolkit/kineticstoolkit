#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2022 Félix Chénier

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
Provide documentation tools to learn using Kinetics Toolkit.
"""

__author__ = "Félix Chénier"
__copyright__ = "Copyright (C) 2022 Félix Chénier"
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"


import kineticstoolkit.config as config
import os
from kineticstoolkit.decorators import unstable


@unstable
def download(
        name: str = "", **kwargs) -> str:
    """
    Download example data and return its local file name.

    This function download example data from github. These data are the same
    that are used to generate the
    [documentation website](https://kineticstoolkit.uqam.ca).

    These example data are volatile; they are supplied only for the user to
    reproduce the tutorials. Therefore, these data may change according to
    changes in the tutorials.

    Parameters
    ----------
    name
        A string that indicates which data to download. Run the function
        without argument to obtain an up-to-date list of available data.

    Returns
    -------
    str
        The file name (with complete path) of the downloaded sample data.

    """
    # Additional information for developers:
    # kwargs may include force_download=True, to force download from github.
    # In standard case, the local git version is used to save on download time.
    if 'force_download' not in kwargs:
        kwargs['force_download'] = False

    file_list = {
        'dataframe_example1.csv':
            'timeseries/sample1.csv',
        'dataframe_example2.csv':
            'timeseries/sample2.csv',
        'wheelchair_kinetics.ktk.zip':
            'timeseries/smartwheel.ktk.zip',
        'wheelchair_kinetics.csv':
            'pushrimkinetics/sample_sw_csvtxt.csv',
        'wheelchair_kinetics.txt':
            'pushrimkinetics/sample_sw_csvtxt.csv',
        'wheelchair_kinetics_offsets.csv':
            'pushrimkinetics/sample_swl_overground_propulsion_withrubber.csv',
        'wheelchair_kinematics.c3d':
            'kinematics/sprintbasket.c3d',
        'wheelchair_racing_full_kinematics.c3d':
            'kinematics/racing_full.c3d',
        'noisy_signals.ktk.zip':
            'filters/sample_noisy.ktk.zip',
        'types_of_noise.ktk.zip':
            'filters/sample_noises.ktk.zip',
    }

    if name == "":
        print(f"Available data are {list(file_list.keys())}.")
        return ""

    try:
        import requests
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "The requests module is an optional dependency of Kinetics "
            "Toolkit. It must be installed to allow downloading data."
        )

    # Set the file name to look for
    try:
        file_name = file_list[name]
    except KeyError:
        raise ValueError(
            f"The requested name must be in {list(file_list.keys())}"
        )

    # Try to get it locally from the local git repository
    path = config.root_folder + '/data/' + file_name
    if os.path.exists(path) is True and kwargs['force_download'] is False:
        return path

    else:
        # Otherwise download it.
        file = requests.get(
            'https://github.com/felixchenier/kineticstoolkit/raw/master/data/'
            + file_name
        )
        path = config.temp_folder + '/' + name
        open(path, 'wb').write(file.content)

        return path
