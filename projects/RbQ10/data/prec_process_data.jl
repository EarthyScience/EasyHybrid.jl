using Dates

dfall=CSV.read(joinpath(@__DIR__, "RESP_07_08_09_10_filled.csv"), DataFrame, normalizenames=true, missingstring="NA") # /Net/Groups/BGI/scratch/bahrens/DataHeinemeyerRh

dfall.timesteps = map(eachrow(dfall)) do r
    dlist = (r.year,r.month,r.day,r.hour)
    any(ismissing,dlist) ? missing : DateTime(dlist...)
end
dfall

function datetime_to_fractional_year(t::DateTime)
    year_start = DateTime(year(t), 1, 1)
    year_end = DateTime(year(t) + 1, 1, 1)
    year_length = year_end - year_start
    time_into_year = t - year_start
    return year(t) + (Dates.value(time_into_year) / Dates.value(year_length))
end

# Convert DateTime to fractional year
dfall.fractional_year = map(dfall.timesteps) do t
    isnothing(t) ? missing : datetime_to_fractional_year(t)
end

lat,lon = 53,1

include("g_pot.jl")
hourofday(t) = (t-DateTime(year(t),month(t),day(t)))/Millisecond(1)/(1000*60*60) 

dfall.rgpot = map(t->g_pot(lat,lon,dayofyear(t),hourofday(t))/1000,dfall.timesteps);
dfall.rgpot2 = copy(dfall.rgpot)
dfall.rgpot2[dfall.rgpot2.<0.0] .= 0.0


# Process numeric or missing-containing columns
for col in names(dfall)
    what_type = eltype(dfall[!, col])
    if what_type <: Union{Missing, Real} || what_type <: Real
        dfall[!, col] = Float64.(coalesce.(dfall[!, col], NaN))
    end
end

rename!(dfall, :s_rtot => :R_soil, :s_rr => :R_root, :s_rmyc => :R_myc, :s_rh => :R_het) #

dfall.R_soil[dfall.R_soil.<0.0] .= NaN
dfall.R_root[dfall.R_root.<0.0] .= NaN
dfall.R_myc[dfall.R_myc.<0.0] .= NaN
dfall.R_het[dfall.R_het.<0.0] .= NaN

