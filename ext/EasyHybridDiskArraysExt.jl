module EasyHybridDiskArraysExt

using EasyHybrid: EasyHybrid
using DiskArrays: DiskArrays, AbstractDiskArray, RegularChunks, approx_chunksize, eachchunk

# Disk-backed arrays (Zarr, NetCDF, ... e.g. inside a `YAXArray`) are trained on lazily.
EasyHybrid._is_disk_backed(::AbstractDiskArray) = true

# Chunk length along dimension `d`, used to align lazy read blocks with storage chunks.
function EasyHybrid._storage_chunk_length(a::AbstractDiskArray, d::Int)
    return Int(approx_chunksize(eachchunk(a))[d])
end

# Offset of the chunk grid along dimension `d` (e.g. a view starting mid-chunk).
function EasyHybrid._storage_chunk_offset(a::AbstractDiskArray, d::Int)
    c = eachchunk(a).chunks[d]
    return c isa RegularChunks ? Int(c.offset) : 0
end

end
