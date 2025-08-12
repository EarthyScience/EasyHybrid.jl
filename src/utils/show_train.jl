function _print_nested_keys(io::IO, nt::NamedTuple; indent=4)
    prefix = " " ^ indent
    maxkey = maximum(length.(string.(keys(nt))))  # for alignment
    for (k, v) in pairs(nt)
        kstr = string(k)
        pad  = " " ^ (maxkey - length(kstr) + 2)
        if isa(v, NamedTuple)
            println(io, prefix, kstr, pad, "(", join(propertynames(v), ", "), ")")
        else
            sz = size(v)
            if sz == ()
                println(io, prefix, kstr)  # scalar
            else
                printstyled(io, prefix * kstr * pad; color=10)
                printstyled(io, string(sz); color=:light_black)
                println(io)
            end
        end
    end
end

function Base.show(io::IO, ::MIME"text/plain", tr::TrainResults)
    for name in propertynames(tr)
        val = getproperty(tr, name)

        # Top-level field
        printstyled(io, "  $(name)"; color=6)
        printstyled(io, ": "; color=:yellow)

        # Summary line
        if isa(val, AbstractArray)
            printstyled(io, "$(size(val))"; color=:light_black)
            println(io)
            try
                first_el = first(val)
                if isa(first_el, NamedTuple)
                    _print_nested_keys(io, first_el; indent=4)
                end
            catch
                # Ignore empty arrays
            end
        elseif val isa DataFrame
            printstyled(io, "$(size(val,1))×$(size(val,2)) DataFrame"; color=:light_black)
            println(io)
            printstyled(io, "    "; color=:blue)
            println(io, join(names(val), ", "))
        elseif isa(val, NamedTuple)
            println(io)
            _print_nested_keys(io, val; indent=4)
        else
            println(io)
        end
    end
end