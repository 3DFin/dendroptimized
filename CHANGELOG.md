# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
  - Potential indexing issue in voxelize when x,y,z fields are not at column 0,1,2
  - Fix a inconsistency in border labeling in CC extraction

### Changed
  - Update taskflow to v3.10

## [0.2.1] - 2025-18-02

### Fixed
  - Crash on Windows OS due to using LTGC.
  In this case, the MSVC runtime requirements are higher than those used by CloudCompare / PyQt (from pip), which results in some crashes.

## [0.2.0] - 2025-16-02

### Changed
  - Update submodules (small performance improvements)
  - Add py 3.13 compatibility

## [0.1.0] - 2024-09-04

### Added

- This file!
- Fast voxel computation.
- Connected components extraction / clustering (DBSCAN).
