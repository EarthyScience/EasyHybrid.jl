# Zygote traces Base's keyword-arg name sorting; the results are Bool/Symbol.
for (fname, nargs) in ((:diff_names, 2), (:sym_in, 2), (:merge_names, 2))
    isdefined(Base, fname) || continue
    args = ntuple(_ -> :(::Any), nargs)
    @eval ChainRulesCore.@non_differentiable Base.$fname($(args...))
end
