# activate the project's environment and instantiate dependencies
using NCDatasets, DataFrames, Dates

struct FluxNetSite
    timeseries     :: DataFrame            # time‐series
    scalars        :: Dict{Symbol,Any}     # 0-D & length-1 vars
    profiles       :: DataFrame  # length-N vars (profiles)
    gattrs         :: Dict{String,Any}     # global file attributes
end

"""
    load_fluxnet_nc(path::AbstractString; timevar="date") -> FluxNetSite

Read a FluxNet .nc file into a FluxNetSite struct:
  1. "time" column from the CF-time variable
  2. all 1-D vars whose sole dimension is `timevar` as time-series
  3. 0-D & length-1 vars as scalars
  4. all other 1-D vars (length>1) as profiles
"""
function load_fluxnet_nc(path; timevar="date", timedim="time", soildim = "depth", soilvar = "depth")
    ds = Dataset(path, "r")
    try
        # -- decode CF-time --------------------------------------------------
        tv    = ds[timevar]
        units = get(tv.attrib, "units", "no units on $timevar")
        unit, epoch = split(units, " since ")
        t0    = DateTime(epoch)
        rawt  = tv[:]
        Δ     = unit=="days"    ? Day    :
                unit=="hours"   ? Hour   :
                unit=="minutes" ? Minute :
                unit=="seconds" ? Second :
                error("unsupported time unit $unit")
        times = t0 .+ Δ.(rawt)

        # -- pick up time-series vars by matching the time dimension --------
        temporal = filter(name -> begin
            v = ds[name]
            dnames = dimnames(v)
            length(dnames) == 1 && dnames[1] == timedim
        end, keys(ds))
        df = DataFrame(time = times)
        for name in temporal
            df[!, Symbol(name)] = ds[name][:]
        end

        # -- collect scalars (0-D & length-1) -------------------------------
        scalars = Dict{Symbol,Any}()
        for name in keys(ds)
            v      = ds[name]
            dnames = dimnames(v)
            if isempty(dnames) ||
               (length(dnames) == 1 && dnames[1] != timedim && length(v[:]) == 1)
                scalars[Symbol(name)] = isempty(dnames) ? v[] : v[:][1]
            end
        end

        # -- collect profiles (1-D length>1, non-time) ----------------------
        profiles = Dict{Symbol,Vector}()
        for name in keys(ds)
            v      = ds[name]
            dnames = dimnames(v)
            if length(dnames) == 1 && dnames[1] != timedim && length(v[:]) > 1
                profiles[Symbol(name)] = v[:]
            end
        end

        profiles = filter(name -> begin
            v = ds[name]
            dnames = dimnames(v)
            length(dnames) == 1 && dnames[1] == soildim
        end, keys(ds))
        dfsoil = DataFrame(depth = ds[soilvar][:])
        for name in profiles
            dfsoil[!, Symbol(name)] = ds[name][:]
        end

        # -- global file attributes ----------------------------------------
        gattrs = Dict(k => ds.attrib[k] for k in keys(ds.attrib))

        return FluxNetSite(df, scalars, dfsoil, gattrs)
    finally
        close(ds)
    end
end
