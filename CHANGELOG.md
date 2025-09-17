# Changelog

## Unreleased

## v0.1.4 - 2025-09-17
- dispatch on DimensionalData [#144](https://github.com/EarthyScience/EasyHybrid.jl/pull/140)
- fix split_data (prepare_data) to prepare train, val split outside of train function [#147](https://github.com/EarthyScience/EasyHybrid.jl/pull/146)
- simply pass Dataframe to hybrid model for forward run [#135](https://github.com/EarthyScience/EasyHybrid.jl/pull/135)
- custom output folder in train: train(; folder_to_save = out_folder
- further cleanup of projects folder. Most projects will be moved to their own repos

## v0.1.3 - 2025-09-01

- example of apparent and intrinsic 'true' Q10 with constant and varying Q10. For ELLIS summerschool [#131](https://github.com/EarthyScience/EasyHybrid.jl/pull/131)
- legend in trainboard for quantiles in neural parameters [#132](https://github.com/EarthyScience/EasyHybrid.jl/pull/132)
- subsampling for trainboard plot fixed [#130](https://github.com/EarthyScience/EasyHybrid.jl/pull/130)
- fix prepare_data for when you input a Dataframe with NaNs or missing [#129](https://github.com/EarthyScience/EasyHybrid.jl/pull/129) 

## v0.1.2 - 2025-08-30

- Added hyperparameter training in [#109](https://github.com/EarthyScience/EasyHybrid.jl/pull/109)

## v0.1.1 - 2025-08-21

- License update and badge CC-SA

## v0.1.0 - 2025-08-18

- Initial log, prepare first registered release.