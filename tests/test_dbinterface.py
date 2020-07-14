#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2020 Félix Chénier
#
# This file is not for redistribution.
"""
Unit tests for ktk.DBInterface class.

NOTE ON TEST DATABASE AND DUMMY USER.

ktkDBInterfaceTest and ktkDBInterfaceTutorial run on a local database
that has the same format as the one on biomec.uqam.ca. However, for the
tests and tutorials to pass, a special user (dummyUser) must exist and
have rights on a specially crafter project (dummyProject).

To add this user, enter these SQL commands in mysql:

SELECT ProjectID FROM Projects WHERE ProjectLabel = "dummyProject" INTO @ProjectID;
INSERT INTO Users (username, password, role) VALUES ("dummyUser", "0726aee645e102b5607e7ed5ad4a029a", "STUDENT");
SELECT UserID FROM Users WHERE username = "dummyUser" INTO @UserID;
INSERT INTO ProjectsUsers (ProjectID, UserID) VALUES (@ProjectID, @UserID);

---------
IMPORTANT
---------
For security reasons, the dummyUser must not exist on the real database
%iomec.uqam.ca. To remove this user, run these SQL commands in mysql:

SELECT ProjectID FROM Projects WHERE ProjectLabel = "dummyProject" INTO @ProjectID;
SELECT UserID FROM Users WHERE username = "dummyUser" INTO @UserID;
SELECT ProjectUserID FROM ProjectsUsers WHERE ProjectID = @ProjectID AND UserID = @UserID INTO @ProjectUserID;
DELETE FROM ProjectsUsers WHERE ProjectUserID = @ProjectUserID;
DELETE FROM Users WHERE UserID = @UserID;

"""

import ktk
import os
import shutil
import warnings


def test_connect():
    """Test the connection to BIOMEC database."""
    project_label = 'dummyProject'
    username = 'dummyUser'
    password = 'dummyPassword'
    root_folder = (ktk.config.root_folder +
                   '/doc/data/dbinterface/FC_XX18A')
    url = 'http://localhost/biomec'

    project = ktk.DBInterface(project_label, user=username, password=password,
                              root_folder=root_folder, url=url)

    # Navigating in the project
    project
    project.get()
    project.get('P1')
    project.get('P1')['Sessions']
    project.get('P1', 'GymnaseN1')
    project.get('P1', 'GymnaseN1')['Trials']
    project.get('P1', 'GymnaseN1', 'Run1')
    project.get('P1', 'GymnaseN1', 'Run1')['Files']
    project.get('P1', 'GymnaseN1', 'Run1', 'Kinematics')
    filename = project.get('P1', 'GymnaseN1', 'Run1', 'Kinematics')['FileName']
    assert os.path.basename(filename) == 'kinematics1_dbfid6681n.c3d'


def test_load_save():
    """Test the load and save methods."""
    project_label = 'dummyProject'
    username = 'dummyUser'
    password = 'dummyPassword'
    root_folder = (ktk.config.root_folder +
                   '/doc/data/dbinterface/FC_XX18A')
    url = 'http://localhost/biomec'

    project = ktk.DBInterface(project_label, user=username, password=password,
                              root_folder=root_folder, url=url)

    # For example, let's say we just synchronized the kinematics for Run1 of
    # participant 1
    synced_kinematics = {'dummy_data':
                         'Normally we would save something more useful'}

    # Save these kinematics as a file that is referenced in BIOMEC
    project.save('P1', 'GymnaseN1', 'Run1', 'SyncedKinematics',
                 synced_kinematics)

    # Load back data saved to BIOMEC
    test = project.load('P1', 'GymnaseN1', 'Run1', 'SyncedKinematics')

    assert test == synced_kinematics

    # Cleanup
    shutil.rmtree(root_folder + '/SyncedKinematics')


def test_batch_fix_file_type():
    """Test the batch_fix_file_type method."""
    project_label = 'dummyProject'
    username = 'dummyUser'
    password = 'dummyPassword'
    root_folder = (ktk.config.root_folder +
                   '/doc/data/dbinterface/FC_XX18A')
    url = 'http://localhost/biomec'

    project = ktk.DBInterface(project_label, user=username, password=password,
                              root_folder=root_folder, url=url)

    file_list = []
    for trial in ['Walk1', 'Walk2', 'Run1', 'Run2']:
        file_list.append(project.get(
            'P1', 'GymnaseN1', trial, 'Kinematics')['FileName'])

    # Let say we synchronized these files using an external software, and then we
    # exported the synchronized files into a separate folder.
    # (Here we will simply copy those files into a separate folder).
    try:
        shutil.rmtree(root_folder + '/synchronized_files')
    except Exception:
        pass

    os.mkdir(root_folder + '/synchronized_files')

    for file in file_list:
        dest_file = file.replace(root_folder, root_folder +
                                 '/synchronized_files')
        shutil.copyfile(file, dest_file)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        project.refresh()
    assert len(project.duplicates) > 0

    project.batch_fix_file_type(root_folder + '/synchronized_files',
                                'SyncedKinematics',
                                create_file_entries=True,
                                dry_run=False)

    project.refresh()
    assert len(project.duplicates) == 0

    # Cleanup
    shutil.rmtree(root_folder + '/synchronized_files')
    project.refresh()


def test_tag_files():
    """Test the tag_files method."""
    project_label = 'dummyProject'
    username = 'dummyUser'
    password = 'dummyPassword'
    root_folder = (ktk.config.root_folder +
                   '/doc/data/dbinterface/FC_XX18A')
    url = 'http://localhost/biomec'

    project = ktk.DBInterface(project_label, user=username, password=password,
                              root_folder=root_folder, url=url)

    project.tag_files(include_trial_name=True, dry_run=False)
    assert ('Run1' in
            project.get('P1', 'GymnaseN1', 'Run1', 'Kinematics')['FileName'])

    project.tag_files(include_trial_name=False, dry_run=False)
    assert ('Run1' not in
            project.get('P1', 'GymnaseN1', 'Run1', 'Kinematics')['FileName'])


if __name__ == "__main__":
    import pytest
    pytest.main([__file__])