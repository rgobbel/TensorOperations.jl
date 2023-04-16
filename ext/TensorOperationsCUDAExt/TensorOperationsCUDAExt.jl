module TensorOperationsCUDAExt

using TensorOperations
isdefined(Base, :get_extension) ? (using CUDA) : (using ..CUDA)

export @cutensor

if CUDA.functional() && CUDA.has_cutensor()
    const CuArray = CUDA.CuArray
    const CublasFloat = CUDA.CUBLAS.CublasFloat
    const CublasReal = CUDA.CUBLAS.CublasReal
    for s in (:handle, :CuTensorDescriptor, :cudaDataType,
              :cutensorContractionDescriptor_t, :cutensorContractionFind_t,
              :cutensorContractionPlan_t,
              :CUTENSOR_OP_IDENTITY, :CUTENSOR_OP_CONJ, :CUTENSOR_OP_ADD,
              :CUTENSOR_ALGO_DEFAULT,  :CUTENSOR_WORKSPACE_RECOMMENDED,
              :cutensorPermutation, :cutensorElementwiseBinary, :cutensorReduction,
              :cutensorReductionGetWorkspace, :cutensorComputeType,
              :cutensorGetAlignmentRequirement, :cutensorInitContractionDescriptor,
              :cutensorInitContractionFind, :cutensorContractionGetWorkspace,
              :cutensorInitContractionPlan, :cutensorContraction)
        eval(:(const $s = CUDA.CUTENSOR.$s))
    end
    if isdefined(CUDA, :default_stream)
        const default_stream = CUDA.default_stream
    else
        const default_stream = CUDA.CuDefaultStream
    end
    include("implementation/cuarray.jl")
    @nospecialize
    include("indexnotation/cutensormacros.jl")
    @specialize
end

end
