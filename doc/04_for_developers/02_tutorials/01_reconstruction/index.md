---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Reconstructing virtual markers

In this tutorial, we will reconstruct virtual markers for anatomic landmarks that were not physically instrumented during the movement acquisition. We usually do this kind of reconstruction when it is not practical or feasible to stick a marker on an anatomical landmark. Instead, we track clusters of markers on rigid bodies affixed to the segment, and we express the position of virtual markers relative to these clusters.

## Flowchart test

```{mermaid}
graph TD

cluster_probe("
    Cluster configuration:<br>
    Probe1 <br>
    ... <br>
    Probe_n <br>
    ProbeTip")
class cluster_probe config


subgraph "1. Create the rigid body cluster"

    %% Any acquisition

    file_any_c3d[("
        Any recording with <br>
        with the rigid body <br>
        of 3 markers")]
    class file_any_c3d data

    file_any_c3d --> read_any_c3d
    read_any_c3d[["Read c3d file"]]
    read_any_c3d --> ts_any

    ts_any("
        TimeSeries with: <br>
        RigidBodyMarker1 <br>
        RigidBodyMarker2 <br>
        RigidBodyMarker3") --> create_rigid_body_cluster
    class ts_any data

    create_rigid_body_cluster[["Create rigid body cluster"]] --> cluster_rigid_body
    cluster_rigid_body("
        Cluster configuration:<br>
        RigidBodyMarker1<br>
        RigidBodyMarker2<br>
        RigidBodyMarker3")
    class cluster_rigid_body config

end


subgraph "2. Add the probed marker to the rigid body cluster"

    %% Part 1 Probing acquisition
    file_probing_c3d[("
        Probing recording <br>
        with a rigid body (3 markers) <br>
        and a probe (n markers)")]
    class file_probing_c3d data

    file_probing_c3d --> read_probing_c3d
    read_probing_c3d[["Read c3d file"]]
    read_probing_c3d --> ts_probing

    ts_probing("
        TimeSeries with <br>
        RigidBodyMarker1 <br>
        RigidBodyMarker2 <br>
        RigidBodyMarker3 <br>
        Probe1 <br>
        ... <br>
        Probe_n")
    class ts_probing data

    %% Part 2.1 Tracking Rigid Body
    ts_probing --> track_rigid_body_cluster_probing
    cluster_rigid_body --> track_rigid_body_cluster_probing
    track_rigid_body_cluster_probing[["Track rigid body cluster"]]    
    track_rigid_body_cluster_probing --> ts_rigid_body_probing
    
    ts_rigid_body_probing("
        TimeSeries with <br>
        RigidBodyMarker1 <br>
        RigidBodyMarker2 <br>
        RigidBodyMarker3")
    class ts_rigid_body_probing data
    
    %% Part 2.2 Tracking Probe
    ts_probing --> track_probe_probing
    track_probe_probing[["Track probe cluster"]] --> ts_probe_probing
    cluster_probe --> track_probe_probing

    ts_probe_probing("
        TimeSeries with <br>
        Probe1 <br>
        ... <br>
        Probe_n <br>
        ProbeTip")
    class ts_probe_probing data
    
    %% Part 3 Adding the marker
    ts_probe_probing -- "ProbeTip becomes<br>RigidBodyMarker4" --> extend_rigid_body
    ts_rigid_body_probing --> extend_rigid_body
    cluster_rigid_body --> extend_rigid_body
    
    extend_rigid_body[["Extend rigid body cluster"]] --> new_rigid_body_cluster
    
    new_rigid_body_cluster("
        New cluster configuration:<br>
        RigidBodyMarker1<br>
        RigidBodyMarker2<br>
        RigidBodyMarker3<br>
        RigidBodyMarker4")
    class new_rigid_body_cluster config
    

end


subgraph "3. Reconstruct all markers during the task"

    file_task_c3d[("
        Task recording with <br>
        with a rigid body <br>
        of only 3 markers")]
    class file_task_c3d data

    file_task_c3d --> read_three_c3d
    read_three_c3d[["Read c3d file"]] --> ts_task

    ts_task("
        TimeSeries with: <br>
        RigidBodyMarker1 <br>
        RigidBodyMarker2 <br>
        RigidBodyMarker3")
    class ts_task data
    
    new_rigid_body_cluster --> track_rigid_body_task
    ts_task --> track_rigid_body_task
    
    track_rigid_body_task[["Track rigid body cluster"]] --> ts_rigid_body_task
    
    ts_rigid_body_task("
        TimeSeries with <br>
        RigidBodyMarker1 <br>
        RigidBodyMarker2 <br>
        RigidBodyMarker3 <br>
        RigidBodyMarker4")
    class ts_rigid_body_task data

end

classDef data fill:#7cf,color:black
classDef config fill:#faa,color:black



```

This process has two steps:

1. A calibration step with several very short calibration acquisitions:

    a) A static acquisition of a few seconds where we can see every marker.

    b) Sometimes, probing acquisitions, one for each virtual marker. In each of these short acquisitions, we point the anatomical landmark using a calibrated probe. The aim is to express these landmarks as part of their segment cluster. Since they move rigidly with the marker clusters, then we could reconstruct the landmarks during the analysed tasks, using the tracked clusters.

2. An task analysis step where the clusters are tracked and the virtual markers are reconstructed into the task acquisition.

```{code-cell}
import kineticstoolkit.lab as ktk
import numpy as np
```

## Read and visualize marker trajectories

We proceed exactly as in the previous tutorials, but this time we will perform the analysis based on a minimal set of markers. Let's say that for the right arm and forearm, all we have is one real marker on the lateral epicondyle, and two plates of three markers affixed to the arm and forearm segments (we will show every other in blue for easier visualization).

```{code-cell}
# Read the markers
markers = ktk.kinematics.read_c3d_file(
    ktk.config.root_folder + '/data/kinematics/sample_propulsion.c3d')

# Set every unnecessary markers to blue
keep_white = ['LateralEpicondyleR', 'ArmR1', 'ArmR2', 'ArmR3',
        'ForearmR1', 'ForearmR2', 'ForearmR3']

for marker_name in markers.data:
    if marker_name not in keep_white:
        markers = markers.add_data_info(marker_name, 'Color', 'b')

# Set the point of view for 3D visualization
viewing_options = {
    'zoom': 3.5,
    'azimuth': 0.8,
    'elevation': 0.16,
    'translation': (0.2, -0.7)
}

# Create the player
player = ktk.Player(markers, **viewing_options)
player.to_html5(start_time=0, stop_time=1)
```

The aim of this tutorial is to reconstruct the right acromion, medial epicondyle and both styloids using static and probing acquisitions. Let's begin.

## Calibration: Defining cluster configurations using a static acquisition

In the static acquisition, every marker should be visible. We use this trial to define, for each cluster, how the cluster's markers are located each relative to the other.

For this example, we will create clusters 'ArmR' and 'ForearmR'.

```{code-cell}
clusters = dict()

# Read the static trial
markers_static = ktk.kinematics.read_c3d_file(
    ktk.config.root_folder + '/data/kinematics/sample_static.c3d')

# Show this trial, just to inspect it
player = ktk.Player(markers_static, **viewing_options)
player.to_html5(start_time=0, stop_time=0.5)
```

Using this trial, we now define the arm cluster:

```{code-cell}
clusters['ArmR'] = ktk.kinematics.create_cluster(
    markers_static,
    marker_names=['ArmR1', 'ArmR2', 'ArmR3', 'LateralEpicondyleR'])

clusters['ArmR']
```

We proceed the same way for the forearm:

```{code-cell}
clusters['ForearmR'] = ktk.kinematics.create_cluster(
    markers_static,
    marker_names=['ForearmR1', 'ForearmR2', 'ForearmR3'])

clusters['ForearmR']
```

For the probe, we will define its cluster from its known specifications. Every 6 local point is expressed relative to a reference frame that is centered at the probe tip:

```{code-cell}
clusters['Probe'] = {
    'ProbeTip': np.array(
        [[0.0, 0.0, 0.0, 1.0]]),
    'Probe1': np.array(
        [[0.0021213, -0.0158328, 0.0864285, 1.0]]),
    'Probe2': np.array(
        [[0.0021213, 0.0158508, 0.0864285, 1.0]]),
    'Probe3': np.array(
        [[0.0020575, 0.0160096, 0.1309445, 1.0]]),
    'Probe4': np.array(
        [[0.0021213, 0.0161204, 0.1754395, 1.0]]),
    'Probe5': np.array(
        [[0.0017070, -0.0155780, 0.1753805, 1.0]]),
    'Probe6': np.array(
        [[0.0017762, -0.0156057, 0.1308888, 1.0]]),
}

clusters['Probe']
```

Now that we defined these clusters, we will be able to track those in every other acquisition. This process can be done using the [track_cluster()](../../../api/kineticstoolkit.kinematics.track_cluster.rst) function.

## Calibration: Defining the virtual marker configurations based on probing acquisitions

Now we will go though every probing acquisition and apply the same process on each acquisition:

1. Locate the probe tip using the probe cluster;
2. Add the probe tip to the segment's cluster.

We will go step by step with the acromion, then we will do the other ones.

```{code-cell}
# Load the markers from the acromion probing trial
probing_markers = ktk.kinematics.read_c3d_file(
    ktk.config.root_folder + '/data/kinematics/sample_probing_acromion_R.c3d')

# Track the probe cluster
tracked_markers = ktk.kinematics.track_cluster(
    probing_markers,
    clusters['Probe']
)

# Look at the contents of the tracked_markers TimeSeries
tracked_markers.data
```

We see that even if the probe tip was not a real marker, its position was reconstructed based on the tracking of the other probe markers. We will add the probe tip to the markers, as the location of the acromion.

```{code-cell}
probing_markers.data['AcromionR'] = tracked_markers.data['ProbeTip']
```

Now that the probing markers contain the new marker 'AcromionR', we can add it to the arm cluster.

```{code-cell}
clusters['ArmR'] = ktk.kinematics.extend_cluster(
    probing_markers, clusters['ArmR'], new_point = 'AcromionR'
)

# Look at the new content of the arm cluster
clusters['ArmR']
```

Now, we can process every other probing acquisition the same way.

```{code-cell}
# Right medial epicondyle
probing_markers = ktk.kinematics.read_c3d_file(
    ktk.config.root_folder
    + '/data/kinematics/sample_probing_medial_epicondyle_R.c3d')

tracked_markers = ktk.kinematics.track_cluster(
    probing_markers, clusters['Probe']
)

probing_markers.data['MedialEpicondyleR'] = tracked_markers.data['ProbeTip']

clusters['ArmR'] = ktk.kinematics.extend_cluster(
    probing_markers, clusters['ArmR'], new_point = 'MedialEpicondyleR'
)

# Right radial styloid
probing_markers = ktk.kinematics.read_c3d_file(
    ktk.config.root_folder
    + '/data/kinematics/sample_probing_radial_styloid_R.c3d')

tracked_markers = ktk.kinematics.track_cluster(
    probing_markers, clusters['Probe']
)

probing_markers.data['RadialStyloidR'] = tracked_markers.data['ProbeTip']

clusters['ForearmR'] = ktk.kinematics.extend_cluster(
    probing_markers, clusters['ForearmR'], new_point = 'RadialStyloidR'
)

# Right ulnar styloid
probing_markers = ktk.kinematics.read_c3d_file(
    ktk.config.root_folder
    + '/data/kinematics/sample_probing_ulnar_styloid_R.c3d')

tracked_markers = ktk.kinematics.track_cluster(
    probing_markers, clusters['Probe']
)

probing_markers.data['UlnarStyloidR'] = tracked_markers.data['ProbeTip']

clusters['ForearmR'] = ktk.kinematics.extend_cluster(
    probing_markers, clusters['ForearmR'], new_point = 'UlnarStyloidR'
)
```

Now every markers that belong to a cluster are defined, be it real or virtual:

```{code-cell}
clusters['ArmR']
```

```{code-cell}
clusters['ForearmR']
```

## Task analysis: Tracking the clusters

Now that we defined the clusters and inluded virtual markers to it, we are ready to process the experimental trial we loaded at the beginning of this tutorial. We already loaded the markers; we will now track the cluster to obtain the position of the virtual markers.

```{code-cell}
markers = markers.merge(
    ktk.kinematics.track_cluster(
        markers, clusters['ArmR']
    )
)

markers = markers.merge(
    ktk.kinematics.track_cluster(
        markers, clusters['ForearmR']
    )
)

# Show those rigid bodies and markers in a player
player = ktk.Player(markers, **viewing_options)

player.to_html5(start_time=0, stop_time=1)
```

That is it, we reconstructed the acromion, medial epicondyle and both styloids from probing acquisitions,  without requiring physical markers on these landmarks. We can conclude by adding links for clearer visualization. From now one, we could continue our analysis and calculate the elbow angles as in the previous tutorial.

```{code-cell}
# Add the segments
segments = {
    'ArmR': {
        'Color': [1, 0.25, 0],
        'Links': [['AcromionR', 'MedialEpicondyleR'],
                  ['AcromionR', 'LateralEpicondyleR'],
                  ['MedialEpicondyleR', 'LateralEpicondyleR']]
    },
    'ForearmR': {
        'Color': [1, 0.5, 0],
        'Links': [['MedialEpicondyleR', 'RadialStyloidR'],
                  ['MedialEpicondyleR', 'UlnarStyloidR'],
                  ['LateralEpicondyleR', 'RadialStyloidR'],
                  ['LateralEpicondyleR', 'UlnarStyloidR'],
                  ['UlnarStyloidR', 'RadialStyloidR']]
    }
}

player = ktk.Player(markers, segments=segments, **viewing_options)
player.to_html5(start_time=0, stop_time=1)
```

For more information on kinematics, please check the [API Reference for the kinematics module](../../../api/kineticstoolkit.kinematics.rst).