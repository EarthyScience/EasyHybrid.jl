### A Pluto.jl notebook ###
# v0.20.8

using Markdown
using InteractiveUtils

# ╔═╡ 16845c40-51bc-11f0-0f10-eda3434e9d35
begin
	cd("C:/Users/bahrens/Desktop/EasyHybrid/")
	using Pkg
	Pkg.activate("projects/RsoilComponents")
	Pkg.develop(path=pwd())
	Pkg.instantiate()
end

# ╔═╡ 8adc69ff-8f60-46af-ac0b-d028fa01ab7d
using EasyHybrid

# ╔═╡ c5ec6ede-9b3b-43e6-a528-b06215100787
using GLMakie

# ╔═╡ 492120e9-e723-4fe4-8569-7f9180fef2a5
using AlgebraOfGraphics

# ╔═╡ 5e40df9d-cbe8-4acb-90c0-dd6d4de110cf
using Statistics

# ╔═╡ 93cd9523-018d-4997-a16d-a984f108e952
md"""
# setup
"""

# ╔═╡ e8f73818-89b2-486d-aec4-c28a0960f687
script_dir = @__DIR__

# ╔═╡ 0ac0b9c7-5b72-49c4-bd42-b41ce59591a5
include(joinpath(script_dir, "data", "prec_process_data.jl"))

# ╔═╡ 9093867f-5220-4908-8d0c-a1297f368388
df = dfall[!, Not(:timesteps)]

# ╔═╡ 7629fc3c-accd-48e9-874e-f1134beaabc6
ds_keyed = to_keyedArray(Float32.(df))

# ╔═╡ a662406e-54bb-4640-9569-93a17b6af380
target_names = [:R_soil]

# ╔═╡ 2e17b560-2743-4c4f-aedf-712b08cdf2cc
hybridRs = RbQ10_2p(target_names, (:cham_temp_filled,), 2.5f0, 1.f0)

# ╔═╡ Cell order:
# ╠═93cd9523-018d-4997-a16d-a984f108e952
# ╠═16845c40-51bc-11f0-0f10-eda3434e9d35
# ╠═8adc69ff-8f60-46af-ac0b-d028fa01ab7d
# ╠═c5ec6ede-9b3b-43e6-a528-b06215100787
# ╠═492120e9-e723-4fe4-8569-7f9180fef2a5
# ╠═5e40df9d-cbe8-4acb-90c0-dd6d4de110cf
# ╠═e8f73818-89b2-486d-aec4-c28a0960f687
# ╠═0ac0b9c7-5b72-49c4-bd42-b41ce59591a5
# ╠═9093867f-5220-4908-8d0c-a1297f368388
# ╠═7629fc3c-accd-48e9-874e-f1134beaabc6
# ╠═a662406e-54bb-4640-9569-93a17b6af380
# ╠═2e17b560-2743-4c4f-aedf-712b08cdf2cc
