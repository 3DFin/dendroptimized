
# Dendroptimized

## Optimized C++ algorithms for dendromatics. 

### Implementation and dependencies:

It relies on the Eigen library for matrix and vector operations, Taskflow for parallel processing primitives, nanoflann for nearest neighbor searches, and Wenzel Jakob’s DisjointSet for computing connected components. These libraries are vendored as submodules into the third_party directory.
Binding are implemented via Nanobind.

### Available algorithms:

- Parallel drop in replacement for dendromatics voxelization
- Ad hoc Parallel "reduced" DBSCAN (should only work in some Dendromatics specific contexts)

To be added in a near future
- C++ Ad Hoc dist_axes computation

## Installing Building

Dendromptimized should be available on PyPI `pip install dendroptimized` should be enough but it is meant to be included into the dendomatics package

## Acknowledgement

Dendroptimized has been developed at the Centre of Wildfire Research of Swansea University (UK) in collaboration with the Research Institute of Biodiversity (CSIC, Spain) and the Department of Mining Exploitation of the University of Oviedo (Spain).

Funding provided by the UK NERC project (NE/T001194/1):

'Advancing 3D Fuel Mapping for Wildfire Behaviour and Risk Mitigation Modelling'

and by the Spanish Knowledge Generation project (PID2021-126790NB-I00):

‘Advancing carbon emission estimations from wildfires applying artificial intelligence to 3D terrestrial point clouds’.
