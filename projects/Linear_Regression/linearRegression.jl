# CC BY-SA 4.0
# activate the project's environment and instantiate dependencies
using Pkg
Pkg.activate("projects/Linear_Regression")
Pkg.develop(path=pwd())
Pkg.instantiate()

# start using the package
using EasyHybrid
using GLMakie
using AlgebraOfGraphics

ds_k = gen_linear_data(; seed=123)

NN = Chain(Dense(2, 15, Lux.relu), Dense(15, 15, Lux.relu), Dense(15, 1));
lhm = LinearHM(NN, (:x1, :x2), (:x3,), (:obs,), 0.0f0)

out = train(lhm, ds_k, (:β, ); nepochs=2500, batchsize=100, opt=Adam(0.001));

## Plotting results
series(out.ps_history; axis=(; xlabel = "epoch", ylabel=""))
# with AoG
yvars = [:obs]
xvars = Symbol.(string.(yvars) .* "_pred")

layers = visual(Scatter, alpha = 0.35)
plt = data(out.train_obs_pred) * layers * mapping(xvars, yvars, col=dims(1) => renamer(string.(yvars)))
plt *= mapping(color = dims(1) => renamer(string.(xvars))=> "Metrics")
# linear
l_linear = linear() * visual(color=:grey25)
plt += data(out.train_obs_pred) * l_linear *  mapping(xvars, yvars, col=dims(1) => renamer(string.(yvars)))

let
   draw(plt, scales(
        X = (; label = rich("Prediction", font=:bold)),
        Y = (; label = "Observation"),
        Color = (; palette = [:tomato, :teal, :orange, :dodgerblue3])
   ),
    legend = (; position=:right, titleposition=:top, merge=false),
    facet = (; linkxaxes = :none, linkyaxes = :none,),
) 
end