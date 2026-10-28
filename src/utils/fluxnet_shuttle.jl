export fluxnet_shuttle_zip, load_fluxnet_shuttle

fluxnet_data_dir() = get(ENV, "FLUXNET_DATA_DIR", joinpath(tempdir(), "fluxnet_shuttle"))

function icos_fluxnet_product(site)
    query = """
    prefix cpmeta: <http://meta.icos-cp.eu/ontologies/cpmeta/>
    prefix prov: <http://www.w3.org/ns/prov#>
    select ?dobj ?fileName where {
        ?dobj cpmeta:hasObjectSpec <http://meta.icos-cp.eu/resources/cpmeta/miscFluxnetArchiveProduct> ;
              cpmeta:wasAcquiredBy/prov:wasAssociatedWith <http://meta.icos-cp.eu/resources/stations/ES_$site> ;
              cpmeta:hasName ?fileName .
        FILTER NOT EXISTS {[] cpmeta:isNextVersionOf ?dobj}
    }
    """
    io = IOBuffer()
    response = Downloads.request(
        "https://meta.icos-cp.eu/sparql";
        method = "POST",
        input = IOBuffer(query),
        output = io,
        headers = ["Content-Type" => "application/sparql-query", "Accept" => "text/csv"],
    )
    response.status == 200 || error("ICOS SPARQL query failed with HTTP status $(response.status)")
    hits = CSV.read(take!(io), DataFrame)
    isempty(hits) && error("No FLUXNET product for $site on the ICOS Carbon Portal")
    return (; id = basename(hits.dobj[1]), filename = hits.fileName[1])
end

"""
    fluxnet_shuttle_zip(site::AbstractString; data_dir) -> String

Return the path to the FLUXNET archive (ONEFlux v1.3 zip) of `site`, downloading it
from the ICOS Carbon Portal, the ICOS data hub of the
[FLUXNET Shuttle](https://github.com/fluxnet/shuttle), if it is not cached yet.

A zip in `data_dir` whose name contains `_<site>_FLUXNET_` is reused, so archives
downloaded with the `fluxnet-shuttle` command line tool (also from AmeriFlux or TERN)
work as well. `data_dir` defaults to `ENV["FLUXNET_DATA_DIR"]` or
`joinpath(tempdir(), "fluxnet_shuttle")`.

# Example
```julia
using EasyHybrid
zipfile = fluxnet_shuttle_zip("DE-Hai")
```
"""
function fluxnet_shuttle_zip(site::AbstractString; data_dir::AbstractString = fluxnet_data_dir())
    mkpath(data_dir)
    cached = filter(f -> occursin("_$(site)_FLUXNET_", f) && endswith(f, ".zip"), readdir(data_dir))
    isempty(cached) || return joinpath(data_dir, last(sort(cached)))
    product = icos_fluxnet_product(site)
    path = joinpath(data_dir, product.filename)
    @info "Downloading $(product.filename) from the ICOS Carbon Portal"
    # the cookie records the accepted CC-BY-4.0 data licence
    Downloads.download(
        "https://data.icos-cp.eu/objects/$(product.id)", path * ".part";
        headers = ["Cookie" => "CpLicenseAcceptedFor=$(product.id)"],
    )
    mv(path * ".part", path)
    return path
end

function fluxmet_hh_csv(zipfile)
    archive = ZipReader(read(zipfile))
    name = only(filter(contains("_FLUXMET_HH_"), zip_names(archive)))
    csv = joinpath(dirname(zipfile), name)
    isfile(csv) || zip_openentry(io -> write(csv, io), archive, name)
    return csv
end

"""
    load_fluxnet_shuttle(site::AbstractString; variables = nothing, years = nothing, data_dir) -> DataFrame

Load the half-hourly fluxes and meteorology (`FLUXMET_HH`) of a FLUXNET Shuttle archive
into a `DataFrame`. The archive is fetched with [`fluxnet_shuttle_zip`](@ref) and the
CSV is unpacked once next to it.

# Arguments
- `site`: FLUXNET site ID, e.g. `"DE-Hai"`.
- `variables`: columns to read (all if `nothing`); `TIMESTAMP_START` is always included.
- `years`: keep only these years (all if `nothing`), e.g. `2016:2018`.
- `data_dir`: cache directory for the archives, see [`fluxnet_shuttle_zip`](@ref).

The returned `DataFrame` has a leading `DateTime` column (from `TIMESTAMP_START`) and
`missing` for the FLUXNET missing value `-9999`.

# Example
```julia
using EasyHybrid
df = load_fluxnet_shuttle("DE-Hai"; variables = [:TA_F, :SW_IN_F, :NEE_VUT_USTAR50], years = 2016:2018)
```
"""
function load_fluxnet_shuttle(
        site::AbstractString;
        variables = nothing,
        years = nothing,
        data_dir::AbstractString = fluxnet_data_dir(),
    )
    csv = fluxmet_hh_csv(fluxnet_shuttle_zip(site; data_dir))
    cols = isnothing(variables) ? nothing : unique([:TIMESTAMP_START; Symbol.(variables)])
    df = CSV.read(csv, DataFrame; select = cols, missingstring = "-9999")
    df.DateTime = DateTime.(string.(df.TIMESTAMP_START), dateformat"yyyymmddHHMM")
    isnothing(years) || (df = df[in(years).(year.(df.DateTime)), :])
    return DataFrames.select!(df, :DateTime, DataFrames.Not(:DateTime))
end
