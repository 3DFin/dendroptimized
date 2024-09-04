
# Dendroptimized

## Optimized C++ algorithms for dendromatics. 

### Implementation and dependencies:

It relies on the [`Eigen`](https://eigen.tuxfamily.org/) library for matrix and vector operations, [`Taskflow`](https://taskflow.github.io) for parallel processing primitives, [`nanoflann`](https://github.com/jlblancoc/nanoflann) for nearest neighbor searches, and Wenzel Jakob’s [`DisjointSet`](https://github.com/wjakob/dset) for computing connected components. These libraries are vendored as submodules into the third_party directory.
Binding are implemented via [`nanobind`](https://github.com/wjakob/nanobind).

### Available algorithms:

- Parallel drop in replacement for dendromatics voxelization
- _ad hoc_ parallel "reduced" DBSCAN (should only work in some `dendromatics` specific contexts)

To be added in a near future
- C++ _ad hoc_ approximate dist axes computation

## Installing / Building

`dendroptimized` is available on `PyPI`.
`pip install dendroptimized` should be enough but it is meant to be used/called by the `dendromatics` package

`dendroptimized` use `scikit-build-core` as its build system. It is PEP 517 compatible and thus build should be as easy as:

```shell
git clone https://github.com/3DFin/dendroptimized
cd dendroptimized
python -m build 
```

## Testing

Some basic tests and benchmarks are provided in the tests directory. Tests can be run in a clean and reproducible environments via `tox` (`tox run` and `tox run -e bench`).

## Acknowledgement

`dendroptimized` has been developed at the Centre of Wildfire Research of Swansea University (UK) in collaboration with the Research Institute of Biodiversity (CSIC, Spain) and the Department of Mining Exploitation of the University of Oviedo (Spain).

Funding provided by the UK NERC project (NE/T001194/1):

'Advancing 3D Fuel Mapping for Wildfire Behaviour and Risk Mitigation Modelling'

and by the Spanish Knowledge Generation project (PID2021-126790NB-I00):

‘Advancing carbon emission estimations from wildfires applying artificial intelligence to 3D terrestrial point clouds’.
