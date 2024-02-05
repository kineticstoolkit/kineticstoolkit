# Test Player API
import kineticstoolkit.lab as ktk

# Download and read markers from a sample C3D file
filename = ktk.doc.download("kinematics_tennis_serve.c3d")
markers = ktk.read_c3d(filename)["Points"]

# Create another person
markers2 = markers.copy()
keys = list(markers2.data.keys())
for key in keys:
    markers2.data[key] += [[1.0, 0.0, 0.0, 0.0]]
    markers2.rename_data(key, key.replace("Derrick", "Viktor"), in_place=True)


interconnections = dict()  # Will contain all segment definitions

interconnections["LLowerLimb"] = {
    "Color": [0, 0.5, 1],  # In RGB format (here, greenish blue)
    "Links": [  # List of lines that span lists of markers
        ["*LTOE", "*LHEE", "*LANK", "*LTOE"],
        ["*LANK", "*LKNE", "*LASI"],
        ["*LKNE", "*LPSI"],
    ],
}

interconnections["RLowerLimb"] = {
    "Color": [0, 0.5, 1],
    "Links": [
        ["*RTOE", "*RHEE", "*RANK", "*RTOE"],
        ["*RANK", "*RKNE", "*RASI"],
        ["*RKNE", "*RPSI"],
    ],
}

interconnections["LUpperLimb"] = {
    "Color": [0, 0.5, 1],
    "Links": [
        ["*LSHO", "*LELB", "*LWRA", "*LFIN"],
        ["*LELB", "*LWRB", "*LFIN"],
        ["*LWRA", "*LWRB"],
    ],
}

interconnections["RUpperLimb"] = {
    "Color": [1, 0.5, 0],
    "Links": [
        ["*RSHO", "*RELB", "*RWRA", "*RFIN"],
        ["*RELB", "*RWRB", "*RFIN"],
        ["*RWRA", "*RWRB"],
    ],
}

interconnections["Head"] = {
    "Color": [1, 0.5, 1],
    "Links": [
        ["*C7", "*LFHD", "*RFHD", "*C7"],
        ["*C7", "*LBHD", "*RBHD", "*C7"],
        ["*LBHD", "*LFHD"],
        ["*RBHD", "*RFHD"],
    ],
}

interconnections["TrunkPelvis"] = {
    "Color": [0.5, 1, 0.5],
    "Links": [
        ["*LASI", "*STRN", "*RASI"],
        ["*STRN", "*CLAV"],
        ["*LPSI", "*T10", "*RPSI"],
        ["*T10", "*C7"],
        ["*LASI", "*LSHO", "*LPSI"],
        ["*RASI", "*RSHO", "*RPSI"],
        [
            "*LPSI",
            "*LASI",
            "*RASI",
            "*RPSI",
            "*LPSI",
        ],
        [
            "*LSHO",
            "*CLAV",
            "*RSHO",
            "*C7",
            "*LSHO",
        ],
    ],
}

# In this file, the up axis is z:
p = ktk.Player(markers, markers2, up="z", interconnections=interconnections)
