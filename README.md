<img src="https://github.com/Jutho/TensorOperations.jl/blob/master/docs/src/assets/logo.svg" width="150">

# TensorOperations.jl

Fast tensor operations using a convenient Einstein index notation.

| **Documentation** | **Build Status** |
|:-----------------:|:----------------:|
| [![][docs-stable-img]][docs-stable-url] [![][docs-dev-img]][docs-dev-url] | [![CI][ci-img]][ci-url] [![CI (Julia nightly)][ci-julia-nightly-img]][ci-julia-nightly-url] [![][codecov-img]][codecov-url] |

| **Digital Object Identifier** | **Downloads** |
|:-----------------------------:|:-------------:|
[![DOI][doi-img]][doi-url] | [![TensorOperations Downloads][genie-img]][genie-url]

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://jutho.github.io/TensorOperations.jl/latest

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://jutho.github.io/TensorOperations.jl/stable

[github-img]: https://github.com/Jutho/TensorOperations.jl/workflows/CI/badge.svg
[github-url]: https://github.com/Jutho/TensorOperations.jl/actions?query=workflow%3ACI

[ci-img]: https://github.com/Jutho/TensorOperations.jl/workflows/CI/badge.svg
[ci-url]: https://github.com/Jutho/TensorOperations.jl/actions?query=workflow%3ACI

[ci-julia-nightly-img]: https://github.com/Jutho/TensorOperations.jl/workflows/CI%20(Julia%20nightly)/badge.svg
[ci-julia-nightly-url]: https://github.com/Jutho/TensorOperations.jl/actions?query=workflow%3A%22CI+%28Julia+nightly%29%22

[codecov-img]: https://codecov.io/gh/Jutho/TensorOperations.jl/branch/master/graph/badge.svg
[codecov-url]: https://codecov.io/gh/Jutho/TensorOperations.jl

[doi-img]: https://zenodo.org/badge/DOI/10.5281/zenodo.3245497.svg
[doi-url]: https://doi.org/10.5281/zenodo.3245497

[genie-img]: https://shields.io/endpoint?url=https://pkgs.genieframework.com/api/v1/badge/TensorOperations
[genie-url]: https://pkgs.genieframework.com?packages=TensorOperations

## New in version 3.2.5

*   With Julia 1.9, CUDA support is an extension, so the CUDA package will only be
    required if actually needed.

## What's new in v3

*   Switched to CUDA.jl instead of CuArrays.jl, which effectively restricts support to
    Julia 1.4 and higher.
*   The default cache size for intermediate results is now the minimum of either 4GB or one
    quarter of your total memory (obtained via `Sys.total_memory()`). Furthermore, the
    structure (i.e. `size`) and `eltype` of the temporaries is now also used as lookup key
    in the LRU cache, such that you can run the same code on different objects with
    different sizes or element types, without constantly having to reallocate the
    temporaries. Finally, the task rather than `threadid` is used to make the cache
    compatible with concurrency at any level.

    As a consequence, different objects for the same temporary location can now be cached,
    such that the cache can grow out of size quickly. Once the cache is not able to hold all
    the temporary objects needed for your simulation, it might actually deteriorate
    perfomance, and you might be better off disabling the cache alltogether with
    `TensorOperations.disable_cache()`.

> **WARNING:** TensorOperations 3.0 contains breaking changes if you did implement support
for custom array / tensor types by overloading `checked_similar_from_indices` etc.

### Code example
TensorOperations.jl is mostly used through the `@tensor` macro which allows one to express
a given operation in terms of
[index notation](https://en.wikipedia.org/wiki/Abstract_index_notation) format, a.k.a.
[Einstein notation](https://en.wikipedia.org/wiki/Einstein_notation)
(using Einstein's summation convention).

```julia
using TensorOperations
α=randn()
A=randn(5,5,5,5,5,5)
B=randn(5,5,5)
C=randn(5,5,5)
D=zeros(5,5,5)
@tensor begin
    D[a,b,c] = A[a,e,f,c,f,g]*B[g,b,e] + α*C[c,a,b]
    E[a,b,c] := A[a,e,f,c,f,g]*B[g,b,e] + α*C[c,a,b]
end
```

In the second to last line, the result of the operation will be stored in the preallocated
array `D`, whereas the last line uses a different assignment operator `:=` in order to
define and allocate a new array `E` of the correct size. The contents of `D` and `E` will
be equal.

For more information, please see the documentation.
