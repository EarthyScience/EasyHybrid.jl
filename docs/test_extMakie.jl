using EasyHybrid
using GLMakie
using GLMakie.Makie.GeometryBasics: AbstractPoint
using DataStructures: CircularBuffer

n_epochs = [0.9]
t_arr = sin.(rand(1))
v_arr = cos.(rand(1))
fig, ax, plt = lossplot(n_epochs, t_arr, v_arr; axis= (; xlabel="Epochs", ylabel ="Loss",))
hidespines!(ax, :r, :t)
fig

# ! do buffer!
zoom_epochs = 50
n_epochs_buffer = CircularBuffer{Int64}(zoom_epochs)
fill!(n_epochs_buffer, 0)
t_arr_buffer = CircularBuffer{Float64}(zoom_epochs)
fill!(t_arr_buffer, t_arr[1])
v_arr_buffer = CircularBuffer{Float64}(zoom_epochs)
fill!(v_arr_buffer, v_arr[1])

ax_z = Axis(fig[1, 1],
    width=Relative(0.35),
    height=Relative(0.35),
    halign=0.95,
    valign=1,
    xlabel="",
    ylabel="",
    rightspinecolor = :dodgerblue,
    leftspinecolor = :dodgerblue,
    topspinecolor = :dodgerblue,
    bottomspinecolor = :dodgerblue,
    title="Zoomed View")

plt_z = lossplot!(ax_z, n_epochs_buffer, t_arr_buffer, v_arr_buffer)
# hidespines!(ax_z, :l, :t)
translate!(ax_z.blockscene, 0, 0, 150)
fig

o_ax_z = ax_z.scene.viewport[].origin

function current_rect2(n_epochs_buffer, t_arr_buffer, v_arr_buffer, zoom_epochs, epoch)
    xzoom_rect = epoch < zoom_epochs ? epoch : zoom_epochs
    mn_tv = minimum(map(minimum, [t_arr_buffer, v_arr_buffer]))
    mx_tv = maximum(map(maximum, [t_arr_buffer, v_arr_buffer]))
    z_rect = Rect2(minimum(n_epochs_buffer), 0.95*mn_tv, xzoom_rect, 1.05*(mx_tv - mn_tv))

    return z_rect
end

z_rect = current_rect2(n_epochs_buffer, t_arr_buffer, v_arr_buffer, zoom_epochs, 0)

plt_b = lines!(ax, z_rect, color=:dodgerblue, linewidth=1)
fig

# scatter!(fig.scene, Point2f(o_ax_z))
# scatter!(fig.scene, Point2f(o_ax_z) + Point2f(first(ax_z.scene.viewport[].widths), 0))

function _project_points_to_figure(ax, p::AbstractPoint)
    return ax.scene.viewport[].origin + Makie.project(ax.scene, p)
end

function _axis_bottom_points(ax_z)
    left_point = Point2f(ax_z.scene.viewport[].origin)
    x_right = first(ax_z.scene.viewport[].widths)
    right_point = left_point + Point2f(x_right, 0)
    return [left_point, right_point]
end

_axis_bottom_points(ax_z)

Legend(fig[1,1, Top()],ax, plt; nbanks=3, framewidth=0.25, halign=0)
fig

for epoch in 1:1000
    # push a new data point
    n_tv = sin(rand())/epoch
    n_vv = cos(rand())/epoch
    push!(n_epochs, epoch)
    push!(t_arr, n_tv)
    push!(v_arr, n_vv)
    #! now the buffers
    push!(n_epochs_buffer, epoch)
    push!(t_arr_buffer, n_tv)
    push!(v_arr_buffer, n_vv)

    new_z_rect = current_rect2(n_epochs_buffer, t_arr_buffer, v_arr_buffer, zoom_epochs, epoch)
    
    #? now that all are updated and synchronized we can update the plot
    
    update!(plt, n_epochs, t_arr, v_arr)
    update!(plt_z, n_epochs_buffer, t_arr_buffer, v_arr_buffer)
    update!(plt_b, arg1=new_z_rect)
    autolimits!(ax)
    autolimits!(ax_z)
    sleep(0.002)
end
fig


# oo = _project_points_to_figure(ax, Point2f(1000, 0.01))
# scatter!(fig.scene, Point2f(oo); color = :olive, markersize=15)
# fig

ax.yscale=log10

# oo2 = _project_points_to_figure(ax, Point2f(1000, 0.02))
# scatter!(fig.scene, Point2f(oo2); color = :orange, markersize=15)

ax.xscale=log10
fig








fig, ax, plt = lossplot(rand(10), rand(10))
scatter!(rand(10), label = "some dots")
Legend(fig[0,1], ax, plt; position =:ct, nbanks=3, tellheight=true, tellwidth=false)
fig

fig, ax, plt = lossplot(rand(10), rand(10); validation_label = "validate me")
scatter!(rand(10), label = "some dots")
Legend(fig[0,1], ax, plt; position =:ct, nbanks=3, tellheight=true, tellwidth=false)
fig