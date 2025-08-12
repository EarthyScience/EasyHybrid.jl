using Downloads
using NCDatasets
using DataFrames

export load_timeseries_netcdf

"""
    load_timeseries_netcdf(path::AbstractString; timedim::AbstractString = "time") -> DataFrame

Reads a NetCDF file where all data variables are 1D over the specified `timedim`
and returns a tidy DataFrame with one row per time step.

- Only includes variables whose sole dimension is `timedim`.
- Does not attempt to parse or convert time units; all columns are read as-is.
"""
function load_timeseries_netcdf(path::AbstractString; timedim::AbstractString = "time")
    localpath = startswith(path, "http") ? Downloads.download(path) : path
    ds = NCDataset(localpath, "r")
    # Identify variables that are 1D over the specified time dimension
    temporal_vars = filter(name -> begin
        v = ds[name]
        dnames = NCDatasets.dimnames(v)
        length(dnames) == 1 && dnames[1] == timedim
    end, keys(ds))
    df = DataFrame()
    for name in temporal_vars
        df[!, Symbol(name)] = ds[name][:]
    end
    close(ds)
    return df
end
