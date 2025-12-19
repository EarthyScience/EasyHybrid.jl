# Changelog

## Unreleased
- Maximizing and Minimizing metric, e.g. :nse and :nseLoss - this is breaking. In training_loss, you now have to use :nseLoss if you want to use 1 - NSE as loss function, :nse is now the real NSE -> 1 not the loss anymore. In loss_types for monitoring and plotting training and validation metrics, you can now also have functions that need to be maximized, e.g. :nse, :pearson (#179)
- in loss_types [:nse, :mse] the first element, i.e. :nse in this example, is used in train_board plot - now you can use different metrics in training_loss and loss_types (#179)
- first element in loss_types as y-axis label in train_board (#179)
- :kge and :kgeLoss, Kling-Gupta efficiency and respective loss can be used in loss_types and training_loss, respectively (#179)
- new functionality in the tune function that allows to switch targets for a hybrid_model. Only train on SOC: tune(hm, df, targets = [:SOC]), train same hybrid model hm but on SOC and MAOC: tune(hm, df, targets = [:SOC, :MAOC]) (#189)

## v0.1.7 - 2025-12-02
- cleaned up dependencies, i.e. rm Flux and others, added explicit imports and formatter [#183](https://github.com/EarthyScience/EasyHybrid.jl/pull/183)

## v0.1.6 - 2025-11-06
- introduces a generic custom input loss function feature in [#163](https://github.com/EarthyScience/EasyHybrid.jl/pull/163)

## v0.1.5 - 2025-09-29
- initial option for cross-validation [#153](https://github.com/EarthyScience/EasyHybrid.jl/pull/153)
- fix hyperopt script in projects (plotting, filenames, extra output) [#155](https://github.com/EarthyScience/EasyHybrid.jl/pull/155) [#156](https://github.com/EarthyScience/EasyHybrid.jl/pull/156)
- tests included for constructHybridModel [#154](https://github.com/EarthyScience/EasyHybrid.jl/pull/154)
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
