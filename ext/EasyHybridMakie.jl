module EasyHybridMakie

using EasyHybrid
using Makie
using Makie.Colors
import Makie

include("HybridTheme.jl")

@debug "Extension loaded!"

Makie.convert_single_argument(wt::WrappedTuples) = Matrix(wt)

function Makie.series(wt::WrappedTuples; axislegend = (;) , attributes...)
    data_matrix, merged_attributes = _series(wt, attributes)
    p = Makie.series(data_matrix; merged_attributes...)
    Makie.axislegend(p.axis; merge=true, axislegend...)
    return p
end

function _series(wt::WrappedTuples, attributes)
    data_matrix = Matrix(wt)'
    plot_attributes = Makie.Attributes(;
        labels = string.(keys(wt))
        )
    user_attributes = Makie.Attributes(; attributes...)
    merged_attributes = merge(user_attributes, plot_attributes)
    return data_matrix, merged_attributes
end

function __init__()
    @debug "setting theme_easy_hybrid"
    # hybrid_latex = merge(theme_easy_hybrid(), theme_latexfonts())
    hybrid_latex = theme_easy_hybrid()
    set_theme!(hybrid_latex)
end

end