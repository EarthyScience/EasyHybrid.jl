function hybrid_colors(alpha=1.0)
    colors = [RGB(0.0, 0.0, 0.0), RGB(0.082, 0.643, 0.918), RGB(0.91, 0.122, 0.361),
              RGB(0.929, 0.773, 0.0), RGB(0.588, 0.196, 0.722), RGB(0.361, 0.722, 0.361),
              RGB(0.522, 0.522, 0.522)]
    @. RGBAf(red(colors), green(colors), blue(colors), alpha)
end

"""
    theme_easy_hybrid()::Attributes
A theme by Lazaro Alonso (`@lazarusa`).
Code to set the theme to `theme_easy_hybrid` is as follows: `Makie.set_theme!(theme_easy_hybrid())`.
The first 3 colours in the palette definition are close to the ones used in the logo of Makie. The other two are close to those of the Julia logo.
"""
function theme_easy_hybrid()
    my_colors = hybrid_colors(0.8)
    my_markers = [:circle, :utriangle, :rect, :diamond, :dtriangle, :diamond, :pentagon]
    my_linestyle = [:solid, :dash, :dot, :dashdot, :dashdotdot, :dash]
    # cycle1 = Cycle([:color, :linestyle], covary=true)
    cycle1 = Cycle([:color], covary=true)
    cycle2 = Cycle([:color, :marker, :strokecolor], covary=true)
    cycle3 = Cycle([:color, :marker, :linestyle, :strokecolor], covary=true)
    cycle4 = Cycle([:color, :linestyle, :strokecolor], covary=true)
    cycle5 = Cycle([:color, :marker, :stemcolor, :stemlinestyle], covary=true)
    Theme(
        font="CMU Serif",
        fontsize=16,
        size = (600, 400),
        palette=(color=my_colors, marker=my_markers, linestyle=my_linestyle,
            strokecolor=my_colors, patchcolor = my_colors,
            stemcolor = my_colors, stemlinestyle=my_linestyle,
            mediancolor = my_colors, whiskercolor = my_colors),
        Lines=(cycle=cycle1,),
        Series=(color=my_colors,),
        Scatter=(cycle=cycle2,),
        ScatterLines = (cycle=cycle3,),
        Density = (cycle = cycle4,),
        BarPlot = (cycle = Cycle([:color, :strokecolor], covary=true),),
        BoxPlot = (cycle = Cycle([:color, :strokecolor, :whiskercolor], covary=true),),
        Errorbars = (cycle = [:color],),
        Hist = (cycle = Cycle([:color, :strokecolor], covary=true),),
        Stairs = (cycle=cycle1,),
        Stem= (cycle=cycle5,),
        Text = (cycle = [:color],),
        Violin = (cycle = [:color, :strokecolor, :mediancolor],),
        strokewidth=0.5,
        colormap= :plasma,
        Axis = (
            xlabel = "x",
            ylabel = "y",
            xtickalign=1,
            ytickalign=1,
            yticksize=10,
            xticksize=10,
            xgridstyle=:dash, ygridstyle=:dash,
            xminorgridstyle=:dash, yminorgridstyle=:dash,
            xminorgridvisible = true,
            yminorgridvisible = true,
            # xtrimspine = true,
            # ytrimspine = true,
            # rightspinevisible = false,
            # topspinevisible = false
            spinewidth=0.5,
            titlefont = :regular,
        ),
        Legend=(framecolor=(:black, 0.35),
            backgroundcolor=(:white, 0.5)),
        Axis3 = (
            zlabelrotation = 0π,
            xlabeloffset = 50,
            ylabeloffset = 55,
            zlabeloffset = 70,
            xgridcolor = (:black, 0.07),
            ygridcolor = (:black, 0.07),
            zgridcolor = (:black, 0.07),
            perspectiveness = 0.5f0,
            azimuth = 1.275π * 1.77,
        ),
        Colorbar=(
            label = "f(x,y)",
            ticksize=15,
            tickalign=1,
            spinewidth=0.25,
            minorticksvisible= true),
        LScene = (
            show_axis = true,
            backgroundcolor = :white)
    )
end