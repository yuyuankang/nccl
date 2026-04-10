/*************************************************************************
 * Copyright (c) 2015-2023, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "argcheck.h" // Need some checks here since we access comm
#include "collectives.h"
#include "enqueue.h"
#include "nccl.h"
#include "nvtx_payload_schemas.h"

#ifdef PII_ENABLED
#include <cuda_runtime.h>
#include <mutex>
#include <atomic>
#endif

const char* ncclFuncToString(ncclFunc_t fn) {
  switch (fn) {
  case ncclFuncAllGather: return "AllGather";
  case ncclFuncAllReduce: return "AllReduce";
  case ncclFuncBroadcast: return "Broadcast";
  case ncclFuncRecv: return "Recv";
  case ncclFuncReduce: return "Reduce";
  case ncclFuncReduceScatter: return "ReduceScatter";
  case ncclFuncSendRecv: return "SendRecv";
  case ncclFuncSend: return "Send";
  default: return "Invalid";
  }
}

const char* ncclDevRedOpToString(ncclDevRedOp_t op) {
  switch (op) {
  case ncclDevSum: return "Sum";
  case ncclDevProd: return "Prod";
  case ncclDevMinMax: return "MinMax";
  case ncclDevPreMulSum: return "PreMulSum";
  case ncclDevSumPostDiv: return "SumPostDiv";
  default: return "Unknown";
  }
}

const char* ncclDatatypeToString(ncclDataType_t type) {
  switch (type) {
  case ncclInt8: return "ncclInt8";
  case ncclInt32: return "ncclInt32";
  case ncclUint32: return "ncclUint32";
  case ncclInt64: return "ncclInt64";
  case ncclUint64: return "ncclUint64";
  case ncclFloat16: return "ncclFloat16";
  case ncclFloat32: return "ncclFloat32";
  case ncclFloat64: return "ncclFloat64";
  case ncclBfloat16: return "ncclBfloat16";
  case ncclFloat8e4m3: return "ncclFloat8e4m3";
  case ncclFloat8e5m2: return "ncclFloat8e5m2";
  default: return "Unknown";
  }
}

const char* ncclAlgoToString(int algo) {
  switch (algo) {
  case NCCL_ALGO_TREE: return "TREE";
  case NCCL_ALGO_RING: return "RING";
  case NCCL_ALGO_COLLNET_DIRECT: return "COLLNET_DIRECT";
  case NCCL_ALGO_COLLNET_CHAIN: return "COLLNET_CHAIN";
  case NCCL_ALGO_NVLS: return "NVLS";
  case NCCL_ALGO_NVLS_TREE: return "NVLS_TREE";
  case NCCL_ALGO_PAT: return "PAT";
  default: return "Unknown";
  }
}

const char* ncclProtoToString(int proto) {
  switch (proto) {
  case NCCL_PROTO_LL: return "LL";
  case NCCL_PROTO_LL128: return "LL128";
  case NCCL_PROTO_SIMPLE: return "SIMPLE";
  default: return "Unknown";
  }
}

NCCL_API(ncclResult_t, ncclAllGather, const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {
  // Just pass the size of one message and not the total bytes sent/received.
  NVTX3_FUNC_WITH_PARAMS(AllGather, NcclNvtxParamsAllGather,
    NVTX3_PAYLOAD(comm ? comm->commHash : 0, sendcount * ncclTypeSize(datatype)));

  struct ncclInfo info = { ncclFuncAllGather, "AllGather",
    sendbuff, recvbuff, sendcount, datatype, ncclSum, 0, comm, stream, /* Args */
    ALLGATHER_CHUNKSTEPS, ALLGATHER_SLICESTEPS };
  return ncclEnqueueCheck(&info);
}

/* ──────────────── PII: AllReduce wire-inflation knobs ──────────────── */
NCCL_PARAM(ARInflateFactor,   "AR_INFLATE_FACTOR",   1);
NCCL_PARAM(ARInflateBytes,    "AR_INFLATE_BYTES",    0);            // static mode: fixed dummy bytes per AR (0 = use factor mode)
NCCL_PARAM(ARInflateMaxBytes, "AR_INFLATE_MAX_BYTES", 256L*1024*1024);

#ifdef PII_ENABLED
/*
 * Process-global dummy device buffer for AR inflation.
 *
 * Allocated EXACTLY ONCE (fixed size = NCCL_AR_INFLATE_MAX_BYTES) at the
 * first non-capturing AllReduce call. vLLM issues many ARs during weight
 * load, KV-cache init, and pre-capture warmup, so the first call reliably
 * lands before any CUDA graph capture begins. The buffer is never freed
 * or reallocated, keeping every captured-graph pointer stable.
 */
static void*               piiDummyBuf      = nullptr;
static size_t              piiDummyBufBytes  = 0;
static void*               piiDummyRegHandle = nullptr;  /* ncclCommRegister handle */
static ncclComm*           piiDummyRegComm   = nullptr;  /* comm used for registration */
static std::once_flag      piiDummyOnce;
static std::atomic<bool>   piiDummyInitFailed{false};
static std::atomic<bool>   piiDummyClampWarned{false};
static std::atomic<bool>   piiDummyCaptureSkipWarned{false};
static std::atomic<bool>   piiDummyEnqueueWarned{false};

/* Forward declaration — defined in register/register.cc */
extern ncclResult_t ncclRegister(struct ncclComm* comm, void* data, size_t size, bool isGraph, void** handle);

static void piiInitDummyBufOnce() {
  size_t bytes = (size_t)ncclParamARInflateMaxBytes();
  bytes = (bytes / 16) * 16;  /* keep aligned for all common dtypes */
  if (bytes == 0) {
    piiDummyInitFailed.store(true);
    WARN("pii AR inflate: NCCL_AR_INFLATE_MAX_BYTES=0, inflation disabled");
    return;
  }
  cudaError_t err = cudaMalloc(&piiDummyBuf, bytes);
  if (err != cudaSuccess) {
    WARN("pii AR inflate: cudaMalloc(%zu) failed: %s — inflation disabled",
         bytes, cudaGetErrorString(err));
    piiDummyBuf = nullptr;
    piiDummyInitFailed.store(true);
    return;
  }
  cudaMemset(piiDummyBuf, 0, bytes);
  piiDummyBufBytes = bytes;
  INFO(NCCL_COLL, "pii AR inflate: dummy buffer pre-allocated %zu bytes", bytes);
}

/* Register piiDummyBuf with the NCCL comm so it uses GDRDMA (not bounce
   buffers). Must be called after piiInitDummyBufOnce and outside graph
   capture. Safe to call multiple times — only registers once. */
static void piiRegisterDummyBuf(ncclComm* comm) {
  if (piiDummyRegHandle != nullptr || piiDummyBuf == nullptr) return;
  ncclResult_t ret = ncclRegister(comm, piiDummyBuf, piiDummyBufBytes, false, &piiDummyRegHandle);
  if (ret == ncclSuccess) {
    piiDummyRegComm = comm;
    INFO(NCCL_COLL, "pii AR inflate: registered dummy buffer for GDRDMA (%zu bytes)", piiDummyBufBytes);
  } else {
    WARN("pii AR inflate: ncclRegister failed (%d), will use bounce buffers", (int)ret);
    piiDummyRegHandle = nullptr;
  }
}
#endif /* PII_ENABLED */
/* ─────────────────────────────────────────────────────────────────── */

NCCL_API(ncclResult_t, ncclAllReduce, const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
  NVTX3_FUNC_WITH_PARAMS(AllReduce, NcclNvtxParamsAllReduce,
    NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), op));

  /* 1. Real AllReduce — unchanged. */
  struct ncclInfo info = { ncclFuncAllReduce, "AllReduce",
    sendbuff, recvbuff, count, datatype, op, 0, comm, stream, /* Args */
    ALLREDUCE_CHUNKSTEPS, ALLREDUCE_SLICESTEPS };
  NCCLCHECK(ncclEnqueueCheck(&info));

#ifdef PII_ENABLED
  /* 2. Optional dummy AllReduce on a REGISTERED buffer.
   *
   *    The dummy buffer (piiDummyBuf) is registered with ncclCommRegister
   *    so NCCL uses the GDRDMA path (NIC reads/writes directly from/to
   *    GPU memory). Without registration, NCCL falls back to host-memory
   *    bounce buffers and the dummy traffic bypasses the GPU M2PCIe.
   *
   *    Two modes:
   *      Static: NCCL_AR_INFLATE_BYTES=N  → send N bytes per AR call
   *      Factor: NCCL_AR_INFLATE_FACTOR=F → send (F-1)×original bytes
   *    Static takes precedence when > 0.
   *    Errors are SWALLOWED — the real AR already succeeded. */
  do {
    const int64_t staticBytes = ncclParamARInflateBytes();
    const int factor = ncclParamARInflateFactor();
    if (comm == nullptr || count == 0 ||
        (staticBytes <= 0 && factor <= 1) ||
        piiDummyInitFailed.load(std::memory_order_relaxed))
      break;

    /* First-call allocation guard: must not cudaMalloc during capture. */
    if (piiDummyBuf == nullptr) {
      cudaStreamCaptureStatus capStatus = cudaStreamCaptureStatusNone;
      cudaStreamIsCapturing(stream, &capStatus);
      if (capStatus != cudaStreamCaptureStatusNone) {
        if (!piiDummyCaptureSkipWarned.exchange(true))
          WARN("pii AR inflate: first AR during stream capture; "
               "skipping until a pre-capture call initializes the buffer");
        break;
      }
      std::call_once(piiDummyOnce, piiInitDummyBufOnce);
      /* Register for GDRDMA immediately after allocation. */
      if (piiDummyBuf != nullptr)
        piiRegisterDummyBuf(comm);
    }
    if (piiDummyBuf == nullptr) break;

    const size_t typeSize = ncclTypeSize(datatype);
    size_t dummyBytes;
    if (staticBytes > 0) {
      dummyBytes = ((size_t)staticBytes / typeSize) * typeSize;
    } else {
      dummyBytes = count * (size_t)(factor - 1) * typeSize;
    }
    if (dummyBytes > piiDummyBufBytes) {
      if (!piiDummyClampWarned.exchange(true))
        WARN("pii AR inflate: dummy %zu B > pre-allocated %zu B; clamping. "
             "Raise NCCL_AR_INFLATE_MAX_BYTES if needed.",
             dummyBytes, piiDummyBufBytes);
      dummyBytes = (piiDummyBufBytes / typeSize) * typeSize;
    }
    size_t dummyCount = dummyBytes / typeSize;
    if (dummyCount == 0) break;

    struct ncclInfo dummyInfo = { ncclFuncAllReduce, "AllReduce",
      piiDummyBuf, piiDummyBuf, dummyCount, datatype, op, 0, comm, stream,
      ALLREDUCE_CHUNKSTEPS, ALLREDUCE_SLICESTEPS };
    ncclResult_t dummyRet = ncclEnqueueCheck(&dummyInfo);
    if (dummyRet != ncclSuccess && !piiDummyEnqueueWarned.exchange(true))
      WARN("pii AR inflate: dummy ncclEnqueueCheck returned %d; "
           "real AR result is unaffected", (int)dummyRet);
  } while (0);
#endif /* PII_ENABLED */

  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclBroadcast, const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclBroadcast(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream) {
  NVTX3_FUNC_WITH_PARAMS(Broadcast, NcclNvtxParamsBroadcast,
    NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), root));

  struct ncclInfo info = { ncclFuncBroadcast, "Broadcast",
    sendbuff, recvbuff, count, datatype, ncclSum, root, comm, stream, /* Args */
    BROADCAST_CHUNKSTEPS, BROADCAST_SLICESTEPS };
  return ncclEnqueueCheck(&info);
}
/* Deprecated original "in place" function, similar to MPI */
NCCL_API(ncclResult_t, ncclBcast, void* buff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclBcast(void* buff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream) {
  return ncclBroadcast(buff, buff, count, datatype, root, comm, stream);
}

NCCL_API(ncclResult_t, ncclReduce, const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclReduce(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) {
  NVTX3_FUNC_WITH_PARAMS(Reduce, NcclNvtxParamsReduce,
    NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), root, op));

  struct ncclInfo info = { ncclFuncReduce, "Reduce",
    sendbuff, recvbuff, count, datatype, op, root, comm, stream, /* Args */
    REDUCE_CHUNKSTEPS, REDUCE_SLICESTEPS };
  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclReduceScatter, const void* sendbuff, void* recvbuff, size_t recvcount,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclReduceScatter(const void* sendbuff, void* recvbuff, size_t recvcount,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
  NVTX3_FUNC_WITH_PARAMS(ReduceScatter, NcclNvtxParamsReduceScatter,
    NVTX3_PAYLOAD(comm ? comm->commHash : 0, recvcount * ncclTypeSize(datatype), op));

  struct ncclInfo info = { ncclFuncReduceScatter, "ReduceScatter",
    sendbuff, recvbuff, recvcount, datatype, op, 0, comm, stream, /* Args */
    REDUCESCATTER_CHUNKSTEPS, REDUCESCATTER_SLICESTEPS };
  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclSend, const void* sendbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclSend(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream) {
  NVTX3_FUNC_WITH_PARAMS(Send, NcclNvtxParamsSendRecv,
    NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), peer));

  struct ncclInfo info = { ncclFuncSend, "Send",
    NULL, (void*)sendbuff, count, datatype, ncclSum, peer, comm, stream, /* Args */
    1, 1 };
  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclRecv, void* recvbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream) {
  NVTX3_FUNC_WITH_PARAMS(Recv, NcclNvtxParamsSendRecv,
    NVTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), peer));

  struct ncclInfo info = { ncclFuncRecv, "Recv",
    NULL, recvbuff, count, datatype, ncclSum, peer, comm, stream, /* Args */
    1, 1 };
  return ncclEnqueueCheck(&info);
}
