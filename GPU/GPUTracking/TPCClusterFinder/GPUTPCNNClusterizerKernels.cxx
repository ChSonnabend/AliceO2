// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTPCNNClusterizerKernels.cxx
/// \author Christian Sonnabend

#include "GPUTPCNNClusterizerKernels.h"
#include "GPUTPCCFClusterizer.h"
#include "GPUTPCCFDeconvolution.h"
#include "GPUTPCGeometry.h"

using namespace o2::gpu;
using namespace o2::gpu::tpccf;

#include "CfConsts.h"
#include "CfUtils.h"
#include "ClusterAccumulator.h"
#include "ML/3rdparty/GPUORTFloat16.h"

#if !defined(GPUCA_GPUCODE)
#include "GPUHostDataTypes.h"
#include "MCLabelAccumulator.h"
#endif

#ifdef GPUCA_GPUCODE
#include "GPUTPCCFClusterizer.inc"
#endif

template <class T, class processorType>
void printInput(int idx, T* input_data, processorType& processors, uint8_t sector)
{
  auto& clusterer = processors.tpcClusterer[sector];
  auto& clustererNN = processors.tpcNNClusterer[sector];
  int base_idx = idx * clustererNN.mNnClusterizerElementSize;
  int nRows = 2 * clustererNN.mNnClusterizerSizeInputRow + 1;
  int nPads = 2 * clustererNN.mNnClusterizerSizeInputPad + 1;
  int nTimes = 2 * clustererNN.mNnClusterizerSizeInputTime + 1;
  int nGrid = nRows * nPads * nTimes;
  int offset = 0;

  // Print the main grid
  std::string output = std::to_string(idx) + ": [";
  for (int r = 0; r < nRows; r++) {
    for (int p = 0; p < nPads; p++) {
      // std::string pad_data = std::to_string(idx) + ": [";
      for (int t = 0; t < nTimes; t++) {
        output += " " + std::to_string((float)input_data[base_idx + offset]);
        if ((r+1)*(p+1)*(t+1) != nGrid) {
          output += ", ";
        }
        offset++;
        // pad_data += std::to_string((float)input_data[base_idx + flat_idx]);
        // if (t != nTimes - 1) {
        //   pad_data += ", ";
        // } else {
        //   pad_data += "]";
        // }
      }
      // LOG(info) << pad_data;
    }
  }

  // If index data is present, print them (just before mean/sigma if present, otherwise at the end)
  if (clustererNN.mNnClusterizerAddIndexData) {
    // LOG(info) << idx << " :[sector: " << (float)input_data[base_idx + idx_offset]
    //           << ", row: " << (float)input_data[base_idx + idx_offset + 1]
    //           << ", pad: " << (float)input_data[base_idx + idx_offset + 2] << "]";
    output += ", " + std::to_string((float)input_data[base_idx + offset]) +
              ", " + std::to_string((float)input_data[base_idx + offset + 1]) +
              ", " + std::to_string((float)input_data[base_idx + offset + 2]);
    offset += 3;
  }
  // If mean/sigma are present, print them (they are at the end)
  if (clustererNN.mNnClusterizerAddMeanSigma) {
    output += ", " + std::to_string((float)input_data[base_idx + offset]) +
              ", " + std::to_string((float)input_data[base_idx + offset + 1]) +
              ", " + std::to_string((float)input_data[base_idx + offset + 2]) +
              ", " + std::to_string((float)input_data[base_idx + offset + 3]);
    // LOG(info) << idx << " :[mean_pad: " << (float)input_data[base_idx + clustererNN.mNnClusterizerElementSize - 4]
    //           << ", mean_time: " << (float)input_data[base_idx + clustererNN.mNnClusterizerElementSize - 3]
    //           << ", sigma_pad: " << (float)input_data[base_idx + clustererNN.mNnClusterizerElementSize - 2]
    //           << ", sigma_time: " << (float)input_data[base_idx + clustererNN.mNnClusterizerElementSize - 1] << "]";
  }
  output += "]";
  LOG(info) << idx << " :[" << base_idx << ", " << base_idx + clustererNN.mNnClusterizerElementSize << " / " << clustererNN.mNnClusterizerElementSize * clustererNN.mNnClusterizerBatchedMode << "]: " << output;
}

// Defining individual thread functions for data filling, determining the class label and running the CF clusterizer
template <>
GPUdii() void GPUTPCNNClusterizerKernels::Thread<GPUTPCNNClusterizerKernels::runCfClusterizer>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUSharedMemory& smem, processorType& processors, uint8_t sector, int8_t dtype, int8_t withMC, uint32_t batchStart)
{
  uint32_t glo_idx = get_global_id(0);
  auto& clusterer = processors.tpcClusterer[sector];
  auto& clustererNN = processors.tpcNNClusterer[sector];
  if (clustererNN.mOutputDataClass[glo_idx] == 0) { // default clusterizer should not be called in batched mode due to mess-up with thread indices
    if (clusterer.mPclusterPosInRow) {
      clusterer.mPclusterPosInRow[glo_idx] = clusterer.mNMaxClusterPerRow;
    }
    return;
  }
  CfArray2D<PackedCharge> chargeMap(reinterpret_cast<PackedCharge*>(clusterer.mPchargeMap));
  CPU_ONLY(MCLabelAccumulator labelAcc(clusterer));
  tpc::ClusterNative* clusterOut = clusterer.mPclusterByRow;
  o2::gpu::GPUTPCCFClusterizer::GPUSharedMemory smem_new;
  GPUTPCCFClusterizer::computeClustersImpl(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), clusterer, clusterer.mPmemory->fragment, smem_new, chargeMap, clusterer.mPfilteredPeakPositions, clusterer.Param().rec, CPU_PTR(&labelAcc), clusterer.mPmemory->counters.nClusters, clusterer.mNMaxClusterPerRow, clusterer.mPclusterInRow, clusterOut, clusterer.mPclusterPosInRow);
}

template <>
GPUdii() void GPUTPCNNClusterizerKernels::Thread<GPUTPCNNClusterizerKernels::fillInputNN>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUSharedMemory& smem, processorType& processors, uint8_t sector, int8_t dtype, int8_t onlyMC, uint32_t batchStart)
{
  uint32_t glo_idx = get_global_id(0);
  auto& clusterer = processors.tpcClusterer[sector];
  auto& clustererNN = processors.tpcNNClusterer[sector];
  uint32_t write_idx = glo_idx * clustererNN.mNnClusterizerElementSize; // Potential optimization: Either choose mNnClusterizerBatchedMode as a power of 2 or calculate from threadId and blockId

  CfArray2D<PackedCharge> chargeMap(reinterpret_cast<PackedCharge*>(clusterer.mPchargeMap));
  CfArray2D<uint8_t> isPeakMap(clusterer.mPpeakMap);
  CfChargePos peak = clusterer.mPfilteredPeakPositions[CAMath::Min(glo_idx + batchStart, (uint32_t)(clusterer.mPmemory->counters.nClusters - 1))];
  int32_t row = static_cast<int>(peak.row()), pad = static_cast<int>(peak.pad()), time = static_cast<int>(peak.time()); // Explicit casting to avoid conversion errors
  float central_charge = static_cast<float>(chargeMap[peak].unpack());
  int32_t row_offset = GPUTPCNNClusterizerKernels::rowOffset(row, clustererNN.mNnClusterizerSizeInputRow);

#ifndef GPUCA_GPUCODE
  GPUCA_UNROLL(U(), U());
#endif
  float tmp_meant = 0.f, tmp_sigmat = 0.f, tmp_meanp = 0.f, tmp_sigmap = 0.f, tmp_charge = 0.f;
  for (int32_t r = -clustererNN.mNnClusterizerSizeInputRow; r <= clustererNN.mNnClusterizerSizeInputRow; r++) {
    bool is_row_boundary = ((row + r) > (o2::tpc::constants::MAXGLOBALPADROW - 1)) || ((row + r) < 0);
    int32_t pad_offset = is_row_boundary ? 0 : GPUTPCNNClusterizerKernels::padOffset(row, row + r);
    for (int32_t p = -clustererNN.mNnClusterizerSizeInputPad + pad_offset; p <= clustererNN.mNnClusterizerSizeInputPad + pad_offset; p++) {
      bool is_boundary = is_row_boundary || GPUTPCNNClusterizerKernels::isBoundary(row + r + row_offset, pad + p, clustererNN.mNnClusterizerSizeInputRow);
      for (int32_t t = -clustererNN.mNnClusterizerSizeInputTime; t <= clustererNN.mNnClusterizerSizeInputTime; t++) {
        if (!is_boundary) {
          CfChargePos tmp_pos(row + r, pad + p, time + t);
          if (r == 0 && !clustererNN.mClusterFlags[2 * glo_idx] && CAMath::Abs(p) < 3 && CAMath::Abs(t) < 3 && p != 0 && t != 0) { // ordering is done for short circuit optimization
            clustererNN.mClusterFlags[2 * glo_idx] += CfUtils::isPeak(isPeakMap[tmp_pos]);
            clustererNN.mClusterFlags[2 * glo_idx + 1] = clustererNN.mClusterFlags[2 * glo_idx];
            if (CAMath::Abs(p) < 2 && CAMath::Abs(t) < 2 && p != 0 && t != 0) {
              tmp_charge += static_cast<float>(chargeMap[tmp_pos].unpack());
              if (p != 0 || t != 0) {
                tmp_meant += tmp_charge * t;
                tmp_sigmat += tmp_charge * t * t;
                tmp_meanp += tmp_charge * p;
                tmp_sigmap += tmp_charge * p * p;
              }
            }
          }
          if (dtype == 0) {
            clustererNN.mInputData_16[write_idx] = (OrtDataType::Float16_t)(static_cast<float>(chargeMap[tmp_pos].unpack()) / central_charge);
          } else if (dtype == 1) {
            clustererNN.mInputData_32[write_idx] = static_cast<float>(chargeMap[tmp_pos].unpack()) / central_charge;
          }
        } else {
          // Filling boundary just to make sure that no values are left unintentionally
          if (dtype == 0) {
            clustererNN.mInputData_16[write_idx] = (OrtDataType::Float16_t)(static_cast<float>(clustererNN.mNnClusterizerBoundaryFillValue));
          } else {
            clustererNN.mInputData_32[write_idx] = static_cast<float>(clustererNN.mNnClusterizerBoundaryFillValue);
          }
        }
        write_idx++;
      }
    }
  }
  if (clustererNN.mNnClusterizerAddIndexData) {
    if (dtype == 0) {
      clustererNN.mInputData_16[write_idx] = (OrtDataType::Float16_t)(static_cast<float>(sector) / o2::tpc::constants::MAXSECTOR);
      clustererNN.mInputData_16[write_idx + 1] = (OrtDataType::Float16_t)(static_cast<float>(row) / o2::tpc::constants::MAXGLOBALPADROW);
      clustererNN.mInputData_16[write_idx + 2] = (OrtDataType::Float16_t)(static_cast<float>(pad) / GPUTPCGeometry::NPads(row));
    } else {
      clustererNN.mInputData_32[write_idx] = static_cast<float>(sector) / o2::tpc::constants::MAXSECTOR;
      clustererNN.mInputData_32[write_idx + 1] = static_cast<float>(row) / o2::tpc::constants::MAXGLOBALPADROW;
      clustererNN.mInputData_32[write_idx + 2] = static_cast<float>(pad) / GPUTPCGeometry::NPads(row);
    }
  }
  if (clustererNN.mNnClusterizerAddMeanSigma) {
    if (dtype == 0) {
      clustererNN.mInputData_16[write_idx + 3] = (OrtDataType::Float16_t)(static_cast<float>(tmp_meanp / tmp_charge));
      clustererNN.mInputData_16[write_idx + 4] = (OrtDataType::Float16_t)(static_cast<float>(tmp_meant / tmp_charge));
      clustererNN.mInputData_16[write_idx + 5] = (OrtDataType::Float16_t)(static_cast<float>(CAMath::Sqrt(tmp_sigmap / tmp_charge - (tmp_meanp / tmp_charge) * (tmp_meanp / tmp_charge))));
      clustererNN.mInputData_16[write_idx + 6] = (OrtDataType::Float16_t)(static_cast<float>(CAMath::Sqrt(tmp_sigmat / tmp_charge - (tmp_meant / tmp_charge) * (tmp_meant / tmp_charge))));
    } else {
      clustererNN.mInputData_32[write_idx + 3] = static_cast<float>(tmp_meanp / tmp_charge);
      clustererNN.mInputData_32[write_idx + 4] = static_cast<float>(tmp_meant / tmp_charge);
      clustererNN.mInputData_32[write_idx + 5] = static_cast<float>(CAMath::Sqrt(tmp_sigmap / tmp_charge - (tmp_meanp / tmp_charge) * (tmp_meanp / tmp_charge)));
      clustererNN.mInputData_32[write_idx + 6] = static_cast<float>(CAMath::Sqrt(tmp_sigmat / tmp_charge - (tmp_meant / tmp_charge) * (tmp_meant / tmp_charge)));
    }
  }
}

template <>
GPUdii() void GPUTPCNNClusterizerKernels::Thread<GPUTPCNNClusterizerKernels::fillInputNNSingleElement>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUSharedMemory& smem, processorType& processors, uint8_t sector, int8_t dtype, int8_t onlyMC, uint32_t batchStart)
{
  uint32_t glo_idx = get_global_id(0);
  auto& clusterer = processors.tpcClusterer[sector];
  auto& clustererNN = processors.tpcNNClusterer[sector];
  uint32_t base_idx = CAMath::Floor((float)glo_idx / clustererNN.mNnClusterizerElementSize);
  uint32_t transient_index = glo_idx - (base_idx * clustererNN.mNnClusterizerElementSize);

  CfArray2D<PackedCharge> chargeMap(reinterpret_cast<PackedCharge*>(clusterer.mPchargeMap));
  CfArray2D<uint8_t> isPeakMap(clusterer.mPpeakMap);
  CfChargePos peak = clusterer.mPfilteredPeakPositions[CAMath::Min(base_idx + batchStart, (uint32_t)(clusterer.mPmemory->counters.nClusters - 1))];
  int32_t row = static_cast<int>(peak.row()), pad = static_cast<int>(peak.pad());

  if (clustererNN.mNnClusterizerAddIndexData && (int32_t)transient_index == (clustererNN.mNnClusterizerElementSize - 1)) {
    uint32_t top_idx = ((base_idx + 1) * clustererNN.mNnClusterizerElementSize) - (clustererNN.mNnClusterizerAddMeanSigma ? 4 : 0);
    if(!clustererNN.mNnClusterFlagsAreSet) {
      for (uint16_t i = 0; i < 8; i++) {
        Delta2 d = cfconsts::InnerNeighbors[i];
        CfChargePos tmp_pos = peak.delta(d);
        clustererNN.mClusterFlags[2 * base_idx] += CfUtils::isPeak(isPeakMap[tmp_pos]);
      }
      clustererNN.mClusterFlags[2 * base_idx + 1] = clustererNN.mClusterFlags[2 * base_idx];
    }
    if (dtype == 0) {
      clustererNN.mInputData_16[top_idx - 3] = (OrtDataType::Float16_t)(static_cast<float>(sector) / o2::tpc::constants::MAXSECTOR);
      clustererNN.mInputData_16[top_idx - 2] = (OrtDataType::Float16_t)(static_cast<float>(row) / o2::tpc::constants::MAXGLOBALPADROW);
      clustererNN.mInputData_16[top_idx - 1] = (OrtDataType::Float16_t)(static_cast<float>(pad) / GPUTPCGeometry::NPads(row));
    } else {
      clustererNN.mInputData_32[top_idx - 3] = static_cast<float>(sector) / o2::tpc::constants::MAXSECTOR;
      clustererNN.mInputData_32[top_idx - 2] = static_cast<float>(row) / o2::tpc::constants::MAXGLOBALPADROW;
      clustererNN.mInputData_32[top_idx - 1] = static_cast<float>(pad) / GPUTPCGeometry::NPads(row);
    }
    // if(base_idx == 1){
    //   LOG(info) << base_idx << " -> " << glo_idx << ", " << transient_index << " / " << clustererNN.mNnClusterizerElementSize << " -> " << top_idx << " (index data)";
    // }
  } else if (clustererNN.mNnClusterizerAddMeanSigma && ((int32_t)transient_index > (clustererNN.mNnClusterizerElementSize - 4))) {
    uint32_t top_idx = (base_idx + 1) * clustererNN.mNnClusterizerElementSize;
    float acc_charge = static_cast<float>(chargeMap[peak].unpack()), mean = 0.f, sigma = 0.f;
    bool pad_time_calculation = ((int32_t)transient_index == (clustererNN.mNnClusterizerElementSize - 2)); // -2 == pad, -3 == time
    for (uint16_t i = 0; i < 8; i++) {
      Delta2 d = cfconsts::InnerNeighbors[i];
      CfChargePos tmp_pos = peak.delta(d);
      float dd = 0;
      if (pad_time_calculation) {
        dd = static_cast<float>(d.x);
      } else {
        dd = static_cast<float>(d.y);
      }
      mean += static_cast<float>(chargeMap[tmp_pos].unpack()) * dd;
      sigma += static_cast<float>(chargeMap[tmp_pos].unpack()) * dd * dd;
      acc_charge += static_cast<float>(chargeMap[tmp_pos].unpack());
    }
    mean /= acc_charge;
    sigma = CAMath::Sqrt(sigma / acc_charge - mean * mean);
    // -4 = mean_pad, -3 = mean_time, -2 = sigma_pad, -1 = sigma_time
    if (dtype == 0) {
      clustererNN.mInputData_16[top_idx - 3 - pad_time_calculation] = (OrtDataType::Float16_t)(static_cast<float>(mean));
      clustererNN.mInputData_16[top_idx - 1 - pad_time_calculation] = (OrtDataType::Float16_t)(static_cast<float>(sigma));
    } else {
      clustererNN.mInputData_32[top_idx - 3 - pad_time_calculation] = static_cast<float>(mean);
      clustererNN.mInputData_32[top_idx - 1 - pad_time_calculation] = static_cast<float>(sigma);
    }
    // if(base_idx == 1){
    //   LOG(info) << base_idx << " -> " << glo_idx << ", " << transient_index << " / " << clustererNN.mNnClusterizerElementSize << " -> " << top_idx << ", " << pad_time_calculation << " (mean-sigma data)";
    // }
  } else if ((int32_t)transient_index < (clustererNN.mNnClusterizerElementSize - (clustererNN.mNnClusterizerAddIndexData ? 3 : 0) - (clustererNN.mNnClusterizerAddMeanSigma ? 4 : 0))) {
    int32_t time = static_cast<int>(peak.time());
    int32_t r = CAMath::Floor(transient_index / ((2 * clustererNN.mNnClusterizerSizeInputPad + 1) * (2 * clustererNN.mNnClusterizerSizeInputTime + 1))) - clustererNN.mNnClusterizerSizeInputRow;
    bool is_row_boundary = ((row + r) > (o2::tpc::constants::MAXGLOBALPADROW - 1)) || ((row + r) < 0);
    if (is_row_boundary) {
      if (dtype == 0) {
        clustererNN.mInputData_16[glo_idx] = (OrtDataType::Float16_t)(static_cast<float>(clustererNN.mNnClusterizerBoundaryFillValue));
      } else {
        clustererNN.mInputData_32[glo_idx] = static_cast<float>(clustererNN.mNnClusterizerBoundaryFillValue);
      }
    } else {
      int32_t row_offset = GPUTPCNNClusterizerKernels::rowOffset(row, clustererNN.mNnClusterizerSizeInputRow);
      int32_t pad_offset = ((row + r) >= 0) && ((row + r) < o2::tpc::constants::MAXGLOBALPADROW) ? GPUTPCNNClusterizerKernels::padOffset(row, row + r) : 0;
      int32_t rest_1 = transient_index % ((2 * clustererNN.mNnClusterizerSizeInputPad + 1) * (2 * clustererNN.mNnClusterizerSizeInputTime + 1));
      int32_t p = CAMath::Floor(rest_1 / (2 * clustererNN.mNnClusterizerSizeInputTime + 1)) - clustererNN.mNnClusterizerSizeInputPad + pad_offset;
      int32_t t = (rest_1 % (2 * clustererNN.mNnClusterizerSizeInputTime + 1)) - clustererNN.mNnClusterizerSizeInputTime;

      bool is_boundary = GPUTPCNNClusterizerKernels::isBoundary(row + r + row_offset, pad + p, clustererNN.mNnClusterizerSizeInputRow) || t < 0 || t >= TPC_MAX_FRAGMENT_LEN_GPU;

      // if(base_idx == 1){
      //   LOG(info) << base_idx << " -> " << glo_idx << ", " << transient_index << " / " << clustererNN.mNnClusterizerElementSize << " -> (" << r << ", " << p << ", " << t << ") " << row_offset << ", " << pad_offset << " (normal data)";
      // }

      if (!is_boundary) {
        float central_charge = static_cast<float>(chargeMap[peak].unpack());
        CfChargePos tmp_pos(row + r, pad + p, time + t);
        if (dtype == 0) {
          clustererNN.mInputData_16[glo_idx] = (OrtDataType::Float16_t)(static_cast<float>(chargeMap[tmp_pos].unpack()) / central_charge);
        } else if (dtype == 1) {
          clustererNN.mInputData_32[glo_idx] = static_cast<float>(chargeMap[tmp_pos].unpack()) / central_charge;
        }
      } else {
        if (dtype == 0) {
          clustererNN.mInputData_16[glo_idx] = (OrtDataType::Float16_t)(static_cast<float>(clustererNN.mNnClusterizerBoundaryFillValue));
        } else {
          clustererNN.mInputData_32[glo_idx] = static_cast<float>(clustererNN.mNnClusterizerBoundaryFillValue);
        }
      }
    }
  }
}

template <>
GPUdii() void GPUTPCNNClusterizerKernels::Thread<GPUTPCNNClusterizerKernels::determineClass1Labels>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUSharedMemory& smem, processorType& processors, uint8_t sector, int8_t dtype, int8_t onlyMC, uint32_t batchStart)
{
  uint32_t glo_idx = get_global_id(0);
  if (dtype == 0) {
    processors.tpcNNClusterer[sector].mOutputDataClass[glo_idx + batchStart] = (int)((processors.tpcNNClusterer[sector].mModelProbabilities_16[glo_idx]).ToFloat() > processors.tpcNNClusterer[sector].mNnClassThreshold);
  } else if (dtype == 1) {
    processors.tpcNNClusterer[sector].mOutputDataClass[glo_idx + batchStart] = (int)(processors.tpcNNClusterer[sector].mModelProbabilities_32[glo_idx] > processors.tpcNNClusterer[sector].mNnClassThreshold);
  }
  // if (glo_idx == 1) {
  //   LOG(info) << "Class1 label for idx " << glo_idx + batchStart << "(with glo_idx: " << glo_idx << ") is " << processors.tpcNNClusterer[sector].mOutputDataClass[glo_idx + batchStart] << " with probability " << (dtype == 0 ? processors.tpcNNClusterer[sector].mModelProbabilities_16[glo_idx].ToFloat() : processors.tpcNNClusterer[sector].mModelProbabilities_32[glo_idx]);
  //   printInput(glo_idx, processors.tpcNNClusterer[sector].mInputData_32, processors, sector);
  // }
}

template <>
GPUdii() void GPUTPCNNClusterizerKernels::Thread<GPUTPCNNClusterizerKernels::determineClass2Labels>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUSharedMemory& smem, processorType& processors, uint8_t sector, int8_t dtype, int8_t onlyMC, uint32_t batchStart)
{
  auto& clustererNN = processors.tpcNNClusterer[sector];
  uint32_t glo_idx = get_global_id(0);
  uint32_t elem_iterator = glo_idx * clustererNN.mNnClusterizerModelClassNumOutputNodes;
  float current_max_prob = 0.f; // If the neural network doesn't contain the softmax as a last layer, the outputs can range in [-infty, infty]
  uint32_t class_label = 0;
  for (uint32_t pIdx = elem_iterator; pIdx < elem_iterator + clustererNN.mNnClusterizerModelClassNumOutputNodes; pIdx++) {
    if (pIdx == elem_iterator) {
      if (dtype == 0) {
        current_max_prob = static_cast<float>(clustererNN.mModelProbabilities_16[pIdx]);
      } else if (dtype == 1) {
        current_max_prob = clustererNN.mModelProbabilities_32[pIdx];
      }
    } else {
      if (dtype == 0) {
        current_max_prob = CAMath::Max(current_max_prob, clustererNN.mModelProbabilities_16[pIdx].ToFloat());
      } else if (dtype == 1) {
        current_max_prob = CAMath::Max(current_max_prob, clustererNN.mModelProbabilities_32[pIdx]);
      }
    }
  }
  // uint32_t class_label = std::distance(elem_iterator, std::max_element(elem_iterator, elem_iterator + clustererNN.mNnClusterizerModelClassNumOutputNodes)); // Multiple outputs of the class network are the probabilities for each class. The highest one "wins"
  clustererNN.mOutputDataClass[glo_idx + batchStart] = class_label;
  if (class_label > 1) {
    clustererNN.mClusterFlags[2 * glo_idx] = 1;
    clustererNN.mClusterFlags[2 * glo_idx + 1] = 1;
  }
}

template <>
GPUdii() void GPUTPCNNClusterizerKernels::Thread<GPUTPCNNClusterizerKernels::publishClass1Regression>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUSharedMemory& smem, processorType& processors, uint8_t sector, int8_t dtype, int8_t withMC, uint32_t batchStart)
{
  uint32_t glo_idx = get_global_id(0);
  auto& clusterer = processors.tpcClusterer[sector];
  auto& clustererNN = processors.tpcNNClusterer[sector];

  uint32_t maxClusterNum = clusterer.mPmemory->counters.nClusters;
  uint32_t full_glo_idx = glo_idx + batchStart;
  if (full_glo_idx >= maxClusterNum) {
    return;
  }
  int32_t model_output_index = glo_idx * clustererNN.mNnClusterizerModelReg1NumOutputNodes;

  CfArray2D<PackedCharge> chargeMap(reinterpret_cast<PackedCharge*>(clusterer.mPchargeMap));
  CfChargePos peak = clusterer.mPfilteredPeakPositions[CAMath::Min(full_glo_idx, maxClusterNum - 1)];
  float central_charge = static_cast<float>(chargeMap[peak].unpack());

  CPU_ONLY(MCLabelAccumulator labelAccElem(clusterer));
  MCLabelAccumulator* labelAcc = CPU_PTR(&labelAccElem);
  tpc::ClusterNative* clusterOut = clusterer.mPclusterByRow;

  // LOG(info) << glo_idx << " -- " << model_output_index << " / " << clustererNN.outputDataReg1.size() << " / " << clustererNN.mNnClusterizerModelReg1NumOutputNodes << " -- " << clusterer.peakPositions.size() << " -- " << clusterer.centralCharges.size();

  // Calculate deltas and remove peak if sigmas are 0
  // Worsens pt resolution
  // if((chargeMap[peak.delta(cfconsts::InnerNeighbors[1])].unpack() == 0) && (chargeMap[peak.delta(cfconsts::InnerNeighbors[6])].unpack() == 0) && (central_charge <= clusterer.Param().rec.tpc.cfQMaxCutoffSinglePad) ||
  //    (chargeMap[peak.delta(cfconsts::InnerNeighbors[3])].unpack() == 0) && (chargeMap[peak.delta(cfconsts::InnerNeighbors[4])].unpack() == 0) && (central_charge <= clusterer.Param().rec.tpc.cfQMaxCutoffSingleTime)) {
  //   if (clusterer.mPclusterPosInRow) {
  //     clusterer.mPclusterPosInRow[full_glo_idx] = clusterer.mNMaxClusterPerRow;
  //   }
  //   return;
  // }

  if ((clustererNN.mOutputDataClass[full_glo_idx] > 0) || (clustererNN.mNnClusterizerUseClassification <= 0)) {

    ClusterAccumulator pc;

    // Publishing logic is taken from default clusterizer
    if (withMC) {
      ClusterAccumulator dummy_pc;
      CPU_ONLY(labelAcc->collect(peak, central_charge));
      GPUTPCCFClusterizer::buildCluster(
        clusterer.Param().rec,
        chargeMap,
        peak,
        smem.posBcast,
        smem.buf,
        smem.innerAboveThreshold,
        &dummy_pc,
        labelAcc);
    }
    if ((clusterer.mPmemory->fragment).isOverlap(peak.time())) {
      if (clusterer.mPclusterPosInRow) {
        clusterer.mPclusterPosInRow[full_glo_idx] = clusterer.mNMaxClusterPerRow;
      }
      return;
    }

    if (clustererNN.mInputData_32[(int)((clustererNN.mNnClusterizerElementSize - (clustererNN.mNnClusterizerAddIndexData ? 3 : 0) - (clustererNN.mNnClusterizerAddMeanSigma ? 4 : 0) - 1)/2) + (glo_idx * clustererNN.mNnClusterizerElementSize)] != 1) {
      // If the input data is not normalized, we need to normalize it here
      // This is done by dividing the output by the central charge
      // This is needed for the regression model to work correctly
      LOG(info) << "Warning: Input data is not normalized, central charge: " << (float)clustererNN.mInputData_32[423 + glo_idx*clustererNN.mNnClusterizerElementSize]
                << ", row: " << (int)peak.row()
                << ", pad: " << (int)peak.pad()
                << ", time: " << (int)peak.time()
                << ", glo_idx: " << glo_idx
                << ", full_glo_idx: " << full_glo_idx;
      if(glo_idx < 5) {
        printInput(glo_idx, clustererNN.mInputData_32, processors, sector);
      }
    }

    if (dtype == 0) {
      if (clustererNN.mOutputDataReg1_16[model_output_index].ToFloat() > 10 || clustererNN.mOutputDataReg1_16[model_output_index + 1].ToFloat() > 10 ||
        clustererNN.mOutputDataReg1_16[model_output_index + 2].ToFloat() > 10 || clustererNN.mOutputDataReg1_16[model_output_index + 3].ToFloat() > 10 ||
        clustererNN.mOutputDataReg1_16[model_output_index + 4].ToFloat() > 30) {
        LOG(info) << "Warning: model output values exceed limit, model: "
              << clustererNN.mOutputDataReg1_16[model_output_index].ToFloat() << ", "
              << clustererNN.mOutputDataReg1_16[model_output_index + 1].ToFloat() << ", "
              << clustererNN.mOutputDataReg1_16[model_output_index + 2].ToFloat() << ", "
              << clustererNN.mOutputDataReg1_16[model_output_index + 3].ToFloat() << ", "
              << clustererNN.mOutputDataReg1_16[model_output_index + 4].ToFloat()
              << ", row: " << (int)peak.row()
              << ", model_output_index: " << model_output_index
              << ", glo_idx: " << glo_idx
              << ", full_glo_idx: " << full_glo_idx;
        if(glo_idx < 5){
          printInput(glo_idx, clustererNN.mInputData_16, processors, sector);
        }
      }
      pc.setFull(central_charge * clustererNN.mOutputDataReg1_16[model_output_index + 4].ToFloat(),
            static_cast<float>(peak.pad()) + clustererNN.mOutputDataReg1_16[model_output_index].ToFloat(),
            clustererNN.mOutputDataReg1_16[model_output_index + 2].ToFloat(),
            (clusterer.mPmemory->fragment).start + static_cast<float>(peak.time()) + clustererNN.mOutputDataReg1_16[model_output_index + 1].ToFloat(),
            clustererNN.mOutputDataReg1_16[model_output_index + 3].ToFloat(),
            clustererNN.mClusterFlags[2 * glo_idx],
            clustererNN.mClusterFlags[2 * glo_idx + 1]);
    } else if (dtype == 1) {
      if (clustererNN.mOutputDataReg1_32[model_output_index] > 10 || clustererNN.mOutputDataReg1_32[model_output_index + 1] > 10 ||
        clustererNN.mOutputDataReg1_32[model_output_index + 2] > 10 || clustererNN.mOutputDataReg1_32[model_output_index + 3] > 10 ||
        clustererNN.mOutputDataReg1_32[model_output_index + 4] > 30) {
        LOG(info) << "Warning: model output values exceed limit, model: "
              << clustererNN.mOutputDataReg1_32[model_output_index] << ", "
              << clustererNN.mOutputDataReg1_32[model_output_index + 1] << ", "
              << clustererNN.mOutputDataReg1_32[model_output_index + 2] << ", "
              << clustererNN.mOutputDataReg1_32[model_output_index + 3] << ", "
              << clustererNN.mOutputDataReg1_32[model_output_index + 4]
              << ", row: " << (int)peak.row()
              << ", model_output_index: " << model_output_index
              << ", glo_idx: " << glo_idx
              << ", full_glo_idx: " << full_glo_idx;
        if(clustererNN.mOutputDataReg1_32[model_output_index] > 30 || clustererNN.mOutputDataReg1_32[model_output_index + 1] > 30 ||
          clustererNN.mOutputDataReg1_32[model_output_index + 2] > 30 || clustererNN.mOutputDataReg1_32[model_output_index + 3] > 30 ||
          clustererNN.mOutputDataReg1_32[model_output_index + 4] > 500) {
          printInput(glo_idx, clustererNN.mInputData_32, processors, sector);
          LOG(info) << glo_idx << ", " << full_glo_idx << ": Class model output: " << clustererNN.mOutputDataClass[full_glo_idx]
                    << ", Class flags: " << clustererNN.mClusterFlags[2 * glo_idx] << ", " << clustererNN.mClusterFlags[2 * glo_idx + 1];
        }
      }
      pc.setFull(central_charge * clustererNN.mOutputDataReg1_32[model_output_index + 4],
          static_cast<float>(peak.pad()) + clustererNN.mOutputDataReg1_32[model_output_index],
          clustererNN.mOutputDataReg1_32[model_output_index + 2],
          (clusterer.mPmemory->fragment).start + static_cast<float>(peak.time()) + clustererNN.mOutputDataReg1_32[model_output_index + 1],
          clustererNN.mOutputDataReg1_32[model_output_index + 3],
          clustererNN.mClusterFlags[2 * glo_idx],
          clustererNN.mClusterFlags[2 * glo_idx + 1]);
    }

    if ((static_cast<float>(peak.pad()) + clustererNN.mOutputDataReg1_32[model_output_index]) > 1000) {
      printf("Error: pad %f is larger than 1000, model: %f, row: %d, model_output_index: %d, glo_idx: %d, full_glo_idx: %d\n",
             static_cast<float>(peak.pad()), clustererNN.mOutputDataReg1_32[model_output_index], (int)peak.row(), model_output_index, glo_idx, full_glo_idx);
      printInput(glo_idx, clustererNN.mInputData_32, processors, sector);
    }

    tpc::ClusterNative myCluster;
    bool rejectCluster = !pc.toNative(peak, central_charge, myCluster, clusterer.Param(), chargeMap);
    if (rejectCluster) {
      if (clusterer.mPclusterPosInRow) {
        clusterer.mPclusterPosInRow[full_glo_idx] = clusterer.mNMaxClusterPerRow;
      }
      return;
    }

    uint32_t rowIndex = 0;
    if(clusterOut != nullptr) {
      rowIndex = GPUTPCCFClusterizer::sortIntoBuckets(
      clusterer,
      myCluster,
      peak.row(),
      clusterer.mNMaxClusterPerRow,
      clusterer.mPclusterInRow,
      clusterOut);
    } else if (clusterer.mPclusterPosInRow != nullptr) {
      rowIndex = clusterer.mPclusterPosInRow[full_glo_idx];
    }
    if (withMC) {
      CPU_ONLY(labelAcc->commit(peak.row(), rowIndex, clusterer.mNMaxClusterPerRow));
    }
  } else {
    if (clusterer.mPclusterPosInRow) {
      clusterer.mPclusterPosInRow[full_glo_idx] = clusterer.mNMaxClusterPerRow;
    }
    return;
  }
}

template <>
GPUdii() void GPUTPCNNClusterizerKernels::Thread<GPUTPCNNClusterizerKernels::publishClass2Regression>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUSharedMemory& smem, processorType& processors, uint8_t sector, int8_t dtype, int8_t withMC, uint32_t batchStart)
{
  uint32_t glo_idx = get_global_id(0);
  auto& clusterer = processors.tpcClusterer[sector];
  auto& clustererNN = processors.tpcNNClusterer[sector];

  CfArray2D<PackedCharge> chargeMap(reinterpret_cast<PackedCharge*>(clusterer.mPchargeMap));
  CfChargePos peak = clusterer.mPfilteredPeakPositions[CAMath::Min(glo_idx + batchStart, (uint32_t)(clusterer.mPmemory->counters.nClusters - 1))];
  float central_charge = static_cast<float>(chargeMap[peak].unpack());

  CPU_ONLY(MCLabelAccumulator labelAccElem(clusterer));
  MCLabelAccumulator* labelAcc = CPU_PTR(&labelAccElem);
  tpc::ClusterNative* clusterOut = clusterer.mPclusterByRow;
  uint32_t full_glo_idx = glo_idx + batchStart;
  uint32_t model_output_index = glo_idx * clustererNN.mNnClusterizerModelReg2NumOutputNodes;

  if ((clustererNN.mOutputDataClass[full_glo_idx] > 0) || !clustererNN.mNnClusterizerUseClassification) {

    ClusterAccumulator pc;

    if (withMC) {
      ClusterAccumulator dummy_pc;
      CPU_ONLY(labelAcc->collect(peak, central_charge));
      GPUTPCCFClusterizer::buildCluster(
        clusterer.Param().rec,
        chargeMap,
        peak,
        smem.posBcast,
        smem.buf,
        smem.innerAboveThreshold,
        &dummy_pc,
        labelAcc);
    }
    if ((clusterer.mPmemory->fragment).isOverlap(peak.time())) {
      if (clusterer.mPclusterPosInRow) {
        clusterer.mPclusterPosInRow[full_glo_idx] = clusterer.mNMaxClusterPerRow;
      }
      return;
    }

    // Cluster 1
    if (dtype == 0) {
      pc.setFull(central_charge * clustererNN.mOutputDataReg2_16[model_output_index + 8].ToFloat(),
                 static_cast<float>(peak.pad()) + clustererNN.mOutputDataReg2_16[model_output_index].ToFloat(),
                 clustererNN.mOutputDataReg2_16[model_output_index + 4].ToFloat(),
                 (clusterer.mPmemory->fragment).start + static_cast<float>(peak.time()) + clustererNN.mOutputDataReg2_16[model_output_index + 2].ToFloat(),
                 clustererNN.mOutputDataReg2_16[model_output_index + 6].ToFloat(),
                 clustererNN.mClusterFlags[2 * full_glo_idx],
                 clustererNN.mClusterFlags[2 * full_glo_idx + 1]);
    } else if (dtype == 1) {
      pc.setFull(central_charge * clustererNN.mOutputDataReg2_32[model_output_index + 8],
                 static_cast<float>(peak.pad()) + clustererNN.mOutputDataReg2_32[model_output_index],
                 clustererNN.mOutputDataReg2_32[model_output_index + 4],
                 (clusterer.mPmemory->fragment).start + static_cast<float>(peak.time()) + clustererNN.mOutputDataReg2_32[model_output_index + 2],
                 clustererNN.mOutputDataReg2_32[model_output_index + 6],
                 clustererNN.mClusterFlags[2 * full_glo_idx],
                 clustererNN.mClusterFlags[2 * full_glo_idx + 1]);
    }

    tpc::ClusterNative myCluster;
    bool rejectCluster = !pc.toNative(peak, central_charge, myCluster, clusterer.Param(), chargeMap);
    if (rejectCluster) {
      if (clusterer.mPclusterPosInRow) {
        clusterer.mPclusterPosInRow[full_glo_idx] = clusterer.mNMaxClusterPerRow;
      }
      return;
    }

    uint32_t rowIndex = 0;
    if (clusterOut != nullptr) {
      rowIndex = GPUTPCCFClusterizer::sortIntoBuckets(
        clusterer,
        myCluster,
        peak.row(),
        clusterer.mNMaxClusterPerRow,
        clusterer.mPclusterInRow,
        clusterOut);
      if (clusterer.mPclusterPosInRow != nullptr) {
        clusterer.mPclusterPosInRow[full_glo_idx] = rowIndex;
      }
    } else if (clusterer.mPclusterPosInRow) {
      rowIndex = clusterer.mPclusterPosInRow[full_glo_idx];
    }
    if (withMC) {
      CPU_ONLY(labelAcc->commit(peak.row(), rowIndex, clusterer.mNMaxClusterPerRow));
    }

    // Cluster 2
    if (dtype == 0) {
      pc.setFull(central_charge * clustererNN.mOutputDataReg2_16[model_output_index + 9].ToFloat(),
                 static_cast<float>(peak.pad()) + clustererNN.mOutputDataReg2_16[model_output_index + 1].ToFloat(),
                 clustererNN.mOutputDataReg2_16[model_output_index + 5].ToFloat(),
                 (clusterer.mPmemory->fragment).start + static_cast<float>(peak.time()) + clustererNN.mOutputDataReg2_16[model_output_index + 3].ToFloat(),
                 clustererNN.mOutputDataReg2_16[model_output_index + 7].ToFloat(),
                 clustererNN.mClusterFlags[2 * full_glo_idx],
                 clustererNN.mClusterFlags[2 * full_glo_idx + 1]);
    } else if (dtype == 1) {
      pc.setFull(central_charge * clustererNN.mOutputDataReg2_32[model_output_index + 9],
                 static_cast<float>(peak.pad()) + clustererNN.mOutputDataReg2_32[model_output_index + 1],
                 clustererNN.mOutputDataReg2_32[model_output_index + 5],
                 (clusterer.mPmemory->fragment).start + static_cast<float>(peak.time()) + clustererNN.mOutputDataReg2_32[model_output_index + 3],
                 clustererNN.mOutputDataReg2_32[model_output_index + 7],
                 clustererNN.mClusterFlags[2 * full_glo_idx],
                 clustererNN.mClusterFlags[2 * full_glo_idx + 1]);
    }

    rejectCluster = !pc.toNative(peak, central_charge, myCluster, clusterer.Param(), chargeMap);
    if (rejectCluster) {
      if (clusterer.mPclusterPosInRow) {
        clusterer.mPclusterPosInRow[full_glo_idx] = clusterer.mNMaxClusterPerRow;
      }
      return;
    }

    rowIndex = GPUTPCCFClusterizer::sortIntoBuckets(
      clusterer,
      myCluster,
      peak.row(),
      clusterer.mNMaxClusterPerRow,
      clusterer.mPclusterInRow,
      clusterOut);
    if (clusterer.mPclusterPosInRow != nullptr) {
      clusterer.mPclusterPosInRow[full_glo_idx] = rowIndex;
    }
    // CPU_ONLY(labelAcc->commit(peak.row(), rowIndex, clusterer.mNMaxClusterPerRow)); // -> Is this needed? How to handle MC labels for split clusters?
  } else {
    if (clusterer.mPclusterPosInRow) {
      clusterer.mPclusterPosInRow[full_glo_idx] = clusterer.mNMaxClusterPerRow;
    }
    return;
  }
}

// THe following arithmetic is done because the network is trained with a split between IROC and OROC boundary
GPUd() int32_t GPUTPCNNClusterizerKernels::padOffset(int32_t row_ref, int32_t row_current)
{
  return (int)((GPUTPCGeometry::NPads(row_current) - GPUTPCGeometry::NPads(row_ref)) / 2);
}

GPUd() int32_t GPUTPCNNClusterizerKernels::rowOffset(int32_t row, int32_t global_shift)
{
  return (row > 62 ? global_shift : 0);
}

GPUd() bool GPUTPCNNClusterizerKernels::isBoundary(int32_t row, int32_t pad, int32_t global_shift)
{
  if (pad < 0 || row < 0) { // Faster short-circuit
    return true;
  } else if (row < 63) {
    return (pad >= static_cast<int>(GPUTPCGeometry::NPads(row)));
  } else if (row < (63 + global_shift)) { // to account for the gap between IROC and OROC. Charge will be set to -1 in order to signal boundary to the neural network
    return true;
  } else if (row < (o2::tpc::constants::MAXGLOBALPADROW + global_shift)) {
    return (pad >= static_cast<int>(GPUTPCGeometry::NPads(row - global_shift)));
  } else {
    return true;
  }
}

// ---------------------------------
template <>
GPUdii() void GPUTPCNNClusterizerKernels::Thread<GPUTPCNNClusterizerKernels::removeAllSplitFlags>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUSharedMemory& smem, processorType& processors, uint8_t sector, int8_t dtype, int8_t onlyMC, uint batchStart)
{
  auto& clusterer = processors.tpcClusterer[sector];
  auto& clustererNN = processors.tpcNNClusterer[sector];
  int glo_idx = get_global_id(0);
  CfArray2D<PackedCharge> chargeMap(reinterpret_cast<PackedCharge*>(clusterer.mPchargeMap));
  CfChargePos pos = clusterer.mPpositions[glo_idx];
  PackedCharge newCharge(chargeMap[pos].unpack(), chargeMap[pos].has3x3Peak(), false);
  chargeMap[pos] = newCharge;
}

// ---------------------------------
template <>
GPUdii() void GPUTPCNNClusterizerKernels::Thread<GPUTPCNNClusterizerKernels::setDeconvolutionFlags>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUSharedMemory& smem, processorType& processors, uint8_t sector, int8_t dtype, int8_t onlyMC, uint batchStart)
{
  // Not run in batched mode. Only run once before applying the networks
  auto& clusterer = processors.tpcClusterer[sector];
  auto& clustererNN = processors.tpcNNClusterer[sector];
  CfArray2D<PackedCharge> chargeMap(reinterpret_cast<PackedCharge*>(clusterer.mPchargeMap));
  CfArray2D<uint8_t> peakMap(clusterer.mPpeakMap);
  SizeT idx = get_global_id(0);

  o2::gpu::GPUTPCCFDeconvolution::GPUSharedMemory smem2;

  bool iamDummy = (idx >= clusterer.mPmemory->counters.nPositions);
  idx = iamDummy ? clusterer.mPmemory->counters.nPositions - 1 : idx;

  CfChargePos pos = clusterer.mPpositions[idx];

  bool iamPeak = CfUtils::isPeak(peakMap[pos]);

  int8_t peakCount = (iamPeak) ? 1 : 0;

  uint16_t ll = get_local_id(0);
  uint16_t partId = ll;

  uint16_t in3x3 = 0;
  bool exclude3x3 = iamPeak || !pos.valid();
  partId = CfUtils::partition<SCRATCH_PAD_WORK_GROUP_SIZE>(smem2, ll, exclude3x3, SCRATCH_PAD_WORK_GROUP_SIZE, &in3x3);

  if (partId < in3x3) {
    smem2.posBcast1[partId] = pos;
  }
  GPUbarrier();

  CfUtils::blockLoad(
    peakMap,
    in3x3,
    SCRATCH_PAD_WORK_GROUP_SIZE,
    ll,
    0,
    8,
    cfconsts::InnerNeighbors,
    smem2.posBcast1,
    smem2.buf);

  uint8_t aboveThreshold = 0;
  if (partId < in3x3) {
    peakCount = GPUTPCNNClusterizerKernels::countPeaksInner(partId, smem2.buf, &aboveThreshold);
  }

  uint16_t in5x5 = 0;
  partId = CfUtils::partition<SCRATCH_PAD_WORK_GROUP_SIZE>(smem2, partId, peakCount > 0 && !exclude3x3, in3x3, &in5x5);

  if (partId < in5x5) {
    smem2.posBcast1[partId] = pos;
    smem2.aboveThresholdBcast[partId] = aboveThreshold;
  }
  GPUbarrier();

  CfUtils::condBlockLoad<uint8_t, true>(
    peakMap,
    in5x5,
    SCRATCH_PAD_WORK_GROUP_SIZE,
    ll,
    0,
    16,
    cfconsts::OuterNeighbors,
    smem2.posBcast1,
    smem2.aboveThresholdBcast,
    smem2.buf);

  if (partId < in5x5) {
    peakCount = GPUTPCNNClusterizerKernels::countPeaksOuter(partId, aboveThreshold, smem2.buf);
    peakCount *= -1;
  }

  if (iamDummy || !pos.valid()) {
    return;
  }

  bool has3x3 = (peakCount > 0);
  peakCount = CAMath::Abs(int32_t(peakCount));
  bool split = (peakCount > 1);

  peakCount = (peakCount == 0) ? 1 : peakCount;

  PackedCharge charge = chargeMap[pos];
  PackedCharge p(charge.unpack(), has3x3, split);

  chargeMap[pos] = p;

}

// ---------------------------------
template <>
GPUdii() void GPUTPCNNClusterizerKernels::Thread<GPUTPCNNClusterizerKernels::publishDeconvolutionFlags>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUSharedMemory& smem, processorType& processors, uint8_t sector, int8_t dtype, int8_t onlyMC, uint batchStart)
{
  uint32_t idx = get_global_id(0);
  auto& clusterer = processors.tpcClusterer[sector];
  auto& clustererNN = processors.tpcNNClusterer[sector];
  CfArray2D<PackedCharge> chargeMap(reinterpret_cast<PackedCharge*>(clusterer.mPchargeMap));
  // uint32_t base_idx = CAMath::Floor(idx / clustererNN.mNnClusterizerElementSize);
  clustererNN.mClusterFlags[2*idx] = 0;
  clustererNN.mClusterFlags[2*idx+1] = 0;
  CfChargePos peak = clusterer.mPfilteredPeakPositions[idx + batchStart];

  // for (int32_t p = -2; p <= 2; p++) {
  //   for (int32_t t = -2; t <= 2; t++) {
  //     CfChargePos d = peak.delta({(int16_t)p,(int16_t)t});
  //     PackedCharge charge = chargeMap[d];
  //     if(CAMath::Abs(p) < 2 && CAMath::Abs(t) < 2){
  //       clustererNN.mClusterFlags[2*idx] += (t != 0 && charge.isSplit());
  //       clustererNN.mClusterFlags[2*idx+1] += (p != 0 && charge.isSplit());
  //     } else {
  //       clustererNN.mClusterFlags[2*idx] += (t != 0 && charge.isSplit() && !charge.has3x3Peak());
  //       clustererNN.mClusterFlags[2*idx+1] += (p != 0 && charge.isSplit() && !charge.has3x3Peak());
  //     }
  //   }
  // }

  for(int i = 0; i < 8; i++) {
    Delta2 d = cfconsts::InnerNeighbors[i];
    CfChargePos tmp_pos = peak.delta(d);
    PackedCharge charge = chargeMap[tmp_pos];
    clustererNN.mClusterFlags[2 * idx] += (d.y != 0 && charge.isSplit());
    clustererNN.mClusterFlags[2 * idx + 1] += (d.x != 0 && charge.isSplit());
  }
  for(int i = 0; i < 16; i++) {
    Delta2 d = cfconsts::OuterNeighbors[i];
    CfChargePos tmp_pos = peak.delta(d);
    PackedCharge charge = chargeMap[tmp_pos];
    clustererNN.mClusterFlags[2 * idx] += (d.y != 0 && charge.isSplit() && !charge.has3x3Peak());
    clustererNN.mClusterFlags[2 * idx + 1] += (d.x != 0 && charge.isSplit() && !charge.has3x3Peak());
  }
}

GPUdi() uint8_t GPUTPCNNClusterizerKernels::countPeaksInner(
  uint16_t ll,
  const uint8_t* isPeak,
  uint8_t* aboveThreshold)
{
  uint8_t peaks = 0;
  GPUCA_UNROLL(U(), U())
  for (uint8_t i = 0; i < 8; i++) {
    uint8_t p = isPeak[ll * 8 + i];
    peaks += CfUtils::isPeak(p);
    *aboveThreshold |= uint8_t(CfUtils::isAboveThreshold(p)) << i;
  }

  return peaks;
}

GPUdi() uint8_t GPUTPCNNClusterizerKernels::countPeaksOuter(
  uint16_t ll,
  uint8_t aboveThreshold,
  const uint8_t* isPeak)
{
  uint8_t peaks = 0;
  GPUCA_UNROLL(U(), U())
  for (uint8_t i = 0; i < 16; i++) {
    uint8_t p = isPeak[ll * 16 + i];
    peaks += CfUtils::isPeak(p);
  }

  return peaks;
}
