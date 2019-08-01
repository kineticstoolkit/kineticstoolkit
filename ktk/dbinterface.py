#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions that manage projects as hosted on Felix Chenier's BIOMEC database.

A tutorial is also available.
"""

from dataclasses import dataclass
from typing import Dict, List
from . import gui as _gui
from . import _repr

import requests
import os


@dataclass
class File:
    """File class."""

    dbid: int = 0
    dbfid: str = ''
    description: str = ''
    filename: str = ''

    def __repr__(self):
        """
        Return the string representation.

        Returns
        -------
        string
            The class' string representation.
        """
        return _repr._format_class_attributes(self)

    def __str__(self):
        return _repr._format_class_attributes(self)


@dataclass
class Trial:
    """Trial class that contains files."""

    files: Dict[str, File]
    dbid: int = 0
    trial_type: str = ''
    repetition: int = 0
    description: str = ''
    notes: str = ''

    def __repr__(self):
        """
        Return the string representation.

        Returns
        -------
        string
            The class' string representation.
        """
        return _repr._format_class_attributes(self)

    def __str__(self):
        return _repr._format_class_attributes(self)



@dataclass
class Session:
    """Session class that contains trials."""

    trials: Dict[str, Trial]
    dbid: int = 0
    date: str = ''
    place: str = ''
    repetition: int = 0
    notes: str = ''

    def __repr__(self):
        """
        Return the string representation.

        Returns
        -------
        string
            The class' string representation.
        """
        return _repr._format_class_attributes(self)

    def __str__(self):
        return _repr._format_class_attributes(self)



@dataclass
class Participant:
    """Participant class that contains sessions."""

    sessions: Dict[str, Session]
    dbid: int = 0
    uid: int = 0
    label: str = ''
    sex: str = ''
    date_of_birth: str = ''
    date_of_injury: str = ''
    dominant_side: str = ''
    pathology: str = ''
    ais: str = ''
    traumatic: str = ''

    def __repr__(self):
        """
        Return the string representation.

        Returns
        -------
        string
            The class' string representation.
        """
        return _repr._format_class_attributes(self)

    def __str__(self):
        return _repr._format_class_attributes(self)


@dataclass
class Project():
    """Project class that contains participants."""

    participants: Dict[str, Participant]
    dbid: int = 0
    label: str = ''
    root_folder: str = ''
    files: str = ''
    missing_files: str = ''
    duplicate_files: str = ''

    def __repr__(self):
        """
        Return the string representation.

        Returns
        -------
        string
            The class' string representation.
        """
        return _repr._format_class_attributes(self)

    def __str__(self):
        return _repr._format_class_attributes(self)


def fetch_project(project_label, user='', password='', root_folder='',
                  url='https://biomec.uqam.ca'):
    """
    Fetch a project information from Felix Chenier's BIOMEC database.

    Parameters
    ----------
    project_label : str
        Project label, for example 'FC_XX18E'.
    user : str, optional
        User name on BIOMEC. The default is ''. If left this way, a dialog
        box will prompt the user to enter his credentials.
    password : str, optional
        Password on BIOMEC. The default is ''.
    root_folder : str, optional
        Location of the project's data files. The default is ''. If left this
        way, a dialog box will prompt the user to select the root folder.
    url : str, optional
        URL for the BIOMEC database. The default is 'https://biomec.uqam.ca'.

    Returns
    -------
    project : Project
        Project information.

    """
    # Get username and password of not supplied
    if user == '':
        user, password = _gui.get_credentials()

    # Append the relative url to the base url
    url += '/py/pywrapper.php'

    # Do the request and executes it
    result = requests.post(url, data={
                                    'command': 'fetchall',
                                    'projectlabel': project_label,
                                    'username': user,
                                    'password': password}, verify=False)

    content = result.content.decode("iso8859_15")
    global project
    exec(content)

    # Add root folder
    if root_folder == '':
        project.root_folder = _gui.get_folder()
    else:
        project.root_folder = root_folder

    #Scan all files in root folder
    folder_list = []
    file_list = []
    for folder, _, files in os.walk(project.root_folder):
        if len(files) > 0:
            for file in files:
                folder_list.append(folder)
                file_list.append(file)

    #Assign files to File instances
    project.files = []
    project.missing_files = []
    project.duplicate_files = []

    for participant_id in project.participants.keys():
        participant = project.participants[participant_id]
        for session_id in participant.sessions.keys():
            session = participant.sessions[session_id]
            for trial_id in session.trials.keys():
                trial = session.trials[trial_id]
                for file_id in trial.files.keys():
                    file = trial.files[file_id]
                    dbfid = file.dbfid

                    # Now find this file
                    file_found = False
                    file_duplicate = False

                    for i in range(0, len(file_list)):
                        if dbfid in file_list[i]:

                            if file_found == False:
                                file_found = True
                                file.filename = os.path.join(folder_list[i],
                                                              file_list[i])
                            else:
                                file_duplicate = True

                    file_tuple = (dbfid, participant_id, session_id,
                                  trial_id, file_id)

                    if file_duplicate:
                        project.duplicate_files.append(file_tuple)
                    elif file_found:
                        project.files.append(file_tuple)
                    else:
                        project.missing_files.append(file_tuple)


    return project
