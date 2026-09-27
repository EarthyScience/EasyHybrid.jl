module EasyHybridEnzymeExt

using EasyHybrid: EasyHybrid
using Enzyme: Enzyme
using Lux: AutoEnzyme

@inline function EasyHybrid._resolve_autodiff_backend(backend::AutoEnzyme{Nothing, F}) where {F}
    fa = F === Nothing ? Enzyme.Const : F
    return AutoEnzyme(;
        mode = Enzyme.set_runtime_activity(Enzyme.Reverse),
        function_annotation = fa,
    )
end

end # module
