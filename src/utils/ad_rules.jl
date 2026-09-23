# AD utilities to cleanly isolate derivative-ignoring logic
@inline _ignore_derivatives(x) = ChainRulesCore.ignore_derivatives(x)
@inline _ignore_derivatives(f::Function) = ChainRulesCore.ignore_derivatives(f)

# Helper to detach state tree from AD gradient computation
@inline _drop_state_gradient(st) = ChainRulesCore.ignore_derivatives(st)

# Zygote traces Base's keyword-arg name sorting; the results are Bool/Symbol.
for (fname, nargs) in ((:diff_names, 2), (:sym_in, 2), (:merge_names, 2))
    isdefined(Base, fname) || continue
    args = ntuple(_ -> :(::Any), nargs)
    @eval ChainRulesCore.@non_differentiable Base.$fname($(args...))
end
