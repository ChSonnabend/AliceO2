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

/// \file GPUTPCNNClusterizer.cxx
/// \author Christian Sonnabend

#include "Rtypes.h"
#include "TTree.h"
#include "TFile.h"
#include "TString.h"
#include "TSystem.h"
#include "TROOT.h"

#include "GPUTPCNNClusterizer.h"
#include "GPUTPCCFClusterizer.h"
#include "GPUTPCCFDeconvolution.h"

#include "CfConsts.h"
#include "CfUtils.h"
#include "ClusterAccumulator.h"
#if !defined(GPUCA_GPUCODE)
#include "GPUHostDataTypes.h"
#include "MCLabelAccumulator.h"
#endif

using namespace o2::gpu;
using namespace o2::gpu::tpccf;

template <>
GPUdii() void GPUTPCNNClusterizer::Thread<GPUTPCNNClusterizer::runCfClusterizer>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUSharedMemory& smem, processorType& clusterer, int8_t dtype, int8_t onlyMC, uint batchStart)
{
  uint glo_idx = get_global_id(0);
  if (clusterer.outputDataClass[glo_idx] == 0) { // default clusterizer should not be called in batched mode due to mess-up with thread indices
    return;
  }
  Array2D<PackedCharge> chargeMap(reinterpret_cast<PackedCharge*>(clusterer.mPchargeMap));
  CPU_ONLY(MCLabelAccumulator labelAcc(clusterer));
  tpc::ClusterNative* clusterOut = (onlyMC) ? nullptr : clusterer.mPclusterByRow;
  o2::gpu::GPUTPCCFClusterizer::GPUSharedMemory smem_new;
  GPUTPCCFClusterizer::computeClustersImpl(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), clusterer, clusterer.mPmemory->fragment, smem_new, chargeMap, clusterer.mPfilteredPeakPositions, clusterer.Param().rec, CPU_PTR(&labelAcc), clusterer.mPmemory->counters.nClusters, clusterer.mNMaxClusterPerRow, clusterer.mPclusterInRow, clusterOut, clusterer.mPclusterPosInRow);
}

template <>
GPUdii() void GPUTPCNNClusterizer::Thread<GPUTPCNNClusterizer::fillInputNN>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUSharedMemory& smem, processorType& clusterer, int8_t dtype, int8_t onlyMC, uint batchStart)
{
  GPUTPCNNClusterizer::fillInputData(nBlocks, nThreads, iBlock, iThread, clusterer, dtype, batchStart);
}

template <>
GPUdii() void GPUTPCNNClusterizer::Thread<GPUTPCNNClusterizer::determineClass1Labels>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUSharedMemory& smem, processorType& clusterer, int8_t dtype, int8_t onlyMC, uint batchStart)
{
  uint glo_idx = get_global_id(0);
  clusterer.outputDataClass[glo_idx + batchStart] = (int)(clusterer.modelProbabilities[glo_idx] > clusterer.nnClassThreshold);
}

template <>
GPUdii() void GPUTPCNNClusterizer::Thread<GPUTPCNNClusterizer::determineClass2Labels>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUSharedMemory& smem, processorType& clusterer, int8_t dtype, int8_t onlyMC, uint batchStart)
{
  uint glo_idx = get_global_id(0);
  auto elem_iterator = clusterer.modelProbabilities.begin() + (unsigned int)(glo_idx * clusterer.model_class.getNumOutputNodes()[0][1]);
  uint class_label = std::distance(elem_iterator, std::max_element(elem_iterator, elem_iterator + clusterer.model_class.getNumOutputNodes()[0][1]));
  clusterer.outputDataClass[glo_idx + batchStart] = class_label;
  if (class_label > 1) {
    clusterer.clusterFlags[glo_idx][0] = 1;
    clusterer.clusterFlags[glo_idx][1] = 1;
  } else {
    clusterer.clusterFlags[glo_idx][0] = 0;
    clusterer.clusterFlags[glo_idx][1] = 0;
  }
}

template <>
GPUdii() void GPUTPCNNClusterizer::Thread<GPUTPCNNClusterizer::publishClass1Regression>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUSharedMemory& smem, processorType& clusterer, int8_t dtype, int8_t onlyMC, uint batchStart)
{
  uint glo_idx = get_global_id(0);
  if (glo_idx >= clusterer.mPmemory->counters.nClusters) {
    return;
  }
  GPUTPCNNClusterizer::publishClustersReg1(glo_idx, smem, clusterer, dtype, onlyMC, batchStart);
}

template <>
GPUdii() void GPUTPCNNClusterizer::Thread<GPUTPCNNClusterizer::publishClass2Regression>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUSharedMemory& smem, processorType& clusterer, int8_t dtype, int8_t onlyMC, uint batchStart)
{
  uint glo_idx = get_global_id(0);
  if (glo_idx >= clusterer.mPmemory->counters.nClusters) {
    return;
  }
  GPUTPCNNClusterizer::publishClustersReg2(glo_idx, smem, clusterer, dtype, onlyMC, batchStart);
}


void GPUTPCNNClusterizer::applyNetworkClass(processorType& clusterer, int8_t dtype, uint batch_idx) {
  if(dtype == 0){
    clusterer.modelProbabilities = clusterer.model_class.inference<OrtDataType::Float16_t, float>(clusterer.inputData16);
  } else {
    clusterer.modelProbabilities = clusterer.model_class.inference<float, float>(clusterer.inputData32);
  }
}

void GPUTPCNNClusterizer::applyNetworkReg1(processorType& clusterer, int8_t dtype) {
  if(dtype == 0){
    clusterer.outputDataReg1 = clusterer.model_reg_1.inference<OrtDataType::Float16_t, float>(clusterer.inputData16);
  } else {
    clusterer.outputDataReg1 = clusterer.model_reg_1.inference<float, float>(clusterer.inputData32);
  }
}

void GPUTPCNNClusterizer::applyNetworkReg2(processorType& clusterer, int8_t dtype) {
  if(dtype == 0){
    clusterer.outputDataReg2 = clusterer.model_reg_2.inference<OrtDataType::Float16_t, float>(clusterer.inputData16);
  } else {
    clusterer.outputDataReg2 = clusterer.model_reg_2.inference<float, float>(clusterer.inputData32);
  }
}

int GPUTPCNNClusterizer::padOffset(int row_ref, int row_current, const GPUTPCGeometry& geo)
{
  return static_cast<int>((geo.NPads(row_current) - geo.NPads(row_ref)) / 2);
}

int GPUTPCNNClusterizer::rowOffset(int row, int global_shift)
{
  return (row > 62 ? global_shift : 0);
}

// ---------------------------------
bool GPUTPCNNClusterizer::isBoundary(int row, int pad, int global_shift, const GPUTPCGeometry& geo)
{
  if (pad < 0 || row < 0) { // Faster short-circuit
    return true;
  } else if (row < 63) {
    return (pad >= static_cast<int>(geo.NPads(row)));
  } else if (row < (63 + global_shift)) { // to account for the gap between IROC and OROC. Charge will be set to -1 in order to signal boundary to the neural network
    return true;
  } else if (row <= o2::tpc::constants::MAXGLOBALPADROW - 1 + global_shift) {
    return (pad >= static_cast<int>(geo.NPads(row - global_shift)));
  } else {
    return true;
  }
}

template<class T>
void GPUTPCNNClusterizer::printInput(int idx, std::vector<T> input_data, processorType& clusterer) {
  int tmp_idx = 0;
  int found_idx = idx/clusterer.nnClusterizerElementSize;
  LOG(info) << found_idx << " :[" << idx << ", " << idx + clusterer.nnClusterizerElementSize << " / " << input_data.size() << "]";
  for (int r = -clusterer.nnClusterizerSizeInputRow; r <= clusterer.nnClusterizerSizeInputRow; r++) {
    for (int p = -clusterer.nnClusterizerSizeInputPad; p <= clusterer.nnClusterizerSizeInputPad; p++) {
      std::string pad_data = std::to_string(found_idx) + ": [";
      for (int t = -clusterer.nnClusterizerSizeInputTime; t <= clusterer.nnClusterizerSizeInputTime; t++) {
        pad_data += std::to_string((float)input_data[idx + tmp_idx]);
        tmp_idx++;
        if(t != clusterer.nnClusterizerSizeInputTime){
          pad_data += ", ";
        } else {
          pad_data += "],";
        }
      }
      LOG(info) << pad_data;
    }
  }
  if(clusterer.nnClusterizerAddIndexData){
    LOG(info) << found_idx << " :[" << (float)input_data[idx + tmp_idx] << ", " << (float)input_data[idx + tmp_idx + 1] << ", " << (float)input_data[idx + tmp_idx + 2] << "]";
  }
}


// ---------------------------------
GPUd() void GPUTPCNNClusterizer::fillInputData(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, processorType& clusterer, int8_t dtype, uint batchStart)
{

  Array2D<PackedCharge> chargeMap(reinterpret_cast<PackedCharge*>(clusterer.mPchargeMap));
  Array2D<uint8_t> isPeakMap(clusterer.mPpeakMap);

  uint glo_idx = get_global_id(0);
  // Shouldn't be needed
  // if (glo_idx + batchStart >= clusterer.mPmemory->counters.nClusters)
  // {
  //   return;
  // }

  uint write_idx = glo_idx * clusterer.nnClusterizerElementSize; // For optimization: Either choose nnClusterizerBatchedMode as a power of 2 or calculate from threadId and blockId

  ChargePos peak = clusterer.mPfilteredPeakPositions[glo_idx + batchStart];
  int row = static_cast<int>(peak.row()), pad = static_cast<int>(peak.pad()), time = static_cast<int>(peak.time());
  float central_charge = static_cast<float>(chargeMap[peak].unpack());

  clusterer.peakPositions[glo_idx] = peak;
  clusterer.centralCharges[glo_idx] = central_charge;

  int row_offset = GPUTPCNNClusterizer::rowOffset(row, clusterer.nnClusterizerSizeInputRow);

  GPUCA_UNROLL(U(), U());
  for (int r = -clusterer.nnClusterizerSizeInputRow; r <= clusterer.nnClusterizerSizeInputRow; r++) {
    bool is_row_boundary = ((row + r) > (o2::tpc::constants::MAXGLOBALPADROW - 1)) || ((row + r) < 0);
    int pad_offset = (is_row_boundary || r == 0) ? 0 : GPUTPCNNClusterizer::padOffset(row, row + r, clusterer.Param().tpcGeometry);
    for (int p = -clusterer.nnClusterizerSizeInputPad + pad_offset; p <= clusterer.nnClusterizerSizeInputPad + pad_offset; p++) {
      bool is_boundary = is_row_boundary || GPUTPCNNClusterizer::isBoundary(row + r + row_offset, pad + p, clusterer.nnClusterizerSizeInputRow, clusterer.Param().tpcGeometry);
      for (int t = -clusterer.nnClusterizerSizeInputTime; t <= clusterer.nnClusterizerSizeInputTime; t++) {
        if (!is_boundary) {
          ChargePos tmp_pos(row + r, pad + p, time + t);
          if (r == 0 && !clusterer.clusterFlags[glo_idx][0] && std::abs(p) < 5 && std::abs(t) < 5 && p!=0 && t!=0) { // ordering is done for short circuit optimization
            clusterer.clusterFlags[glo_idx][0] = CfUtils::isPeak(isPeakMap[tmp_pos]);
            clusterer.clusterFlags[glo_idx][1] = clusterer.clusterFlags[glo_idx][0];
          }
          if(dtype == 0){
            clusterer.inputData16[write_idx] = (OrtDataType::Float16_t)(static_cast<float>(chargeMap[tmp_pos].unpack()) / central_charge);
          } else {
            clusterer.inputData32[write_idx] = static_cast<float>(chargeMap[tmp_pos].unpack()) / central_charge;
          }
        } else {
          if(dtype == 0){
            clusterer.inputData16[write_idx] = (OrtDataType::Float16_t)(static_cast<float>(clusterer.nnClusterizerBoundaryFillValue));
          } else {
            clusterer.inputData32[write_idx] = static_cast<float>(clusterer.nnClusterizerBoundaryFillValue);
          }
        }
        write_idx++;
      }
    }
  }
  if (clusterer.nnClusterizerAddIndexData) {
    if(dtype == 0){
      clusterer.inputData16[write_idx] = (OrtDataType::Float16_t)(clusterer.mISector / 36.f);
      clusterer.inputData16[write_idx + 1] = (OrtDataType::Float16_t)(row / 152.f);
      clusterer.inputData16[write_idx + 2] = (OrtDataType::Float16_t)(static_cast<float>(pad) / clusterer.Param().tpcGeometry.NPads(row));
    } else {
      clusterer.inputData32[write_idx] = clusterer.mISector / 36.f;
      clusterer.inputData32[write_idx + 1] = row / 152.f;
      clusterer.inputData32[write_idx + 2] = static_cast<float>(pad) / clusterer.Param().tpcGeometry.NPads(row);
    }
  }
}


// ---------------------------------
GPUd() void GPUTPCNNClusterizer::publishClustersReg1(uint glo_idx, GPUSharedMemory& smem, processorType& clusterer, int8_t dtype, int8_t onlyMC, uint batchStart)
{
  Array2D<PackedCharge> chargeMap(reinterpret_cast<PackedCharge*>(clusterer.mPchargeMap));
  CPU_ONLY(MCLabelAccumulator labelAccElem(clusterer));
  CPU_ONLY(MCLabelAccumulator* labelAcc = CPU_PTR(&labelAccElem));
  tpc::ClusterNative* clusterOut = (onlyMC) ? nullptr : clusterer.mPclusterByRow;
  uint full_glo_idx = glo_idx + batchStart;
  int model_output_index = glo_idx * clusterer.model_reg_1.getNumOutputNodes()[0][1];

  // LOG(info) << glo_idx << " -- " << model_output_index << " / " << clusterer.outputDataReg1.size() << " / " << clusterer.model_reg_1.getNumOutputNodes()[0][1] << " -- " << clusterer.peakPositions.size() << " -- " << clusterer.centralCharges.size();

  // if (full_glo_idx == 10000) {
  //   LOG(info) << "Cluster: modelProb " << clusterer.modelProbabilities[full_glo_idx] << "; regression: " << clusterer.outputDataReg1[model_output_index] << " / " << clusterer.outputDataReg1[model_output_index + 1] << " / " << clusterer.outputDataReg1[model_output_index + 2] << " / " << clusterer.outputDataReg1[model_output_index + 3] << " / " << clusterer.outputDataReg1[model_output_index + 4];
  //   GPUTPCNNClusterizer::printInput<float>(glo_idx * clusterer.nnClusterizerElementSize, clusterer.inputData32, clusterer);
  // }

  if (clusterer.outputDataClass[full_glo_idx] == 1) {

    ClusterAccumulator pc;

    if ((clusterer.mPmemory->fragment).isOverlap(clusterer.peakPositions[glo_idx].time())) {
      if (clusterer.mPclusterPosInRow) {
        clusterer.mPclusterPosInRow[full_glo_idx] = clusterer.mNMaxClusterPerRow;
      }
      return;
    }

    pc.setFull(clusterer.centralCharges[glo_idx] * clusterer.outputDataReg1[model_output_index + 4],
      static_cast<float>(clusterer.peakPositions[glo_idx].pad()) + clusterer.outputDataReg1[model_output_index],
      clusterer.outputDataReg1[model_output_index + 2],
      static_cast<float>(clusterer.peakPositions[glo_idx].time()) + static_cast<float>((clusterer.mPmemory->fragment).start) + clusterer.outputDataReg1[model_output_index + 1],
      clusterer.outputDataReg1[model_output_index + 3],
      clusterer.clusterFlags[glo_idx][0],
      clusterer.clusterFlags[glo_idx][1]);

    tpc::ClusterNative myCluster;
    bool rejectCluster = !pc.toNative(clusterer.peakPositions[glo_idx], clusterer.centralCharges[glo_idx], myCluster, clusterer.Param(), chargeMap);

    // LOG(info) << glo_idx << ": " << (int)clusterer.peakPositions[glo_idx].row() << " -- " << (int)clusterer.peakPositions[glo_idx].pad() << " <--> " << clusterer.inputData32[(glo_idx + 1) * clusterer.nnClusterizerElementSize - 2] * 152 << ", " << clusterer.inputData32[(glo_idx + 1) * clusterer.nnClusterizerElementSize - 1] * clusterer.Param().tpcGeometry.NPads((int)clusterer.peakPositions[glo_idx].row());

    // MC labels
    ClusterAccumulator dummy_pc;
    CPU_ONLY(labelAcc->collect(clusterer.peakPositions[glo_idx], chargeMap[clusterer.peakPositions[glo_idx]].unpack()));
    GPUTPCCFClusterizer::buildCluster(
      clusterer.Param().rec,
      chargeMap,
      clusterer.peakPositions[glo_idx],
      smem.posBcast,
      smem.buf,
      smem.innerAboveThreshold,
      &dummy_pc,
      labelAcc);
    dummy_pc.finalize(clusterer.peakPositions[glo_idx],
      chargeMap[clusterer.peakPositions[glo_idx]].unpack(),
      (clusterer.mPmemory->fragment).start);
    tpc::ClusterNative myDummyCluster;
    bool rejectDummy = !dummy_pc.toNative(clusterer.peakPositions[glo_idx], clusterer.centralCharges[glo_idx], myDummyCluster, clusterer.Param(), chargeMap);

    // if (std::abs(clusterer.outputDataReg1[model_output_index]) > 4 || std::abs(clusterer.outputDataReg1[model_output_index + 1]) > 4) {
    //   LOG(info) << "[NN, CF] Cluster analysis. fragment " << (clusterer.mPmemory->fragment).index << ", glo_idx " << glo_idx << " -- row " << (int)clusterer.peakPositions[glo_idx].row() << ", pad " << (int)clusterer.peakPositions[glo_idx].pad() << ", time " << (int)clusterer.peakPositions[glo_idx].time() + static_cast<float>((clusterer.mPmemory->fragment).start) << ", charge " << static_cast<float>(clusterer.centralCharges[glo_idx]) << " || "
    //   << static_cast<float>(myCluster.getQtot()) << ", " << static_cast<float>(myCluster.getPad()) << ", " << static_cast<float>(myCluster.getTime()) << ", "<< static_cast<float>(myCluster.getSigmaPad()) << ", " << static_cast<float>(myCluster.getSigmaTime()) << ", " << static_cast<float>(myCluster.getQmax()) << " || "
    //   << static_cast<float>(myDummyCluster.getQtot()) << ", " << static_cast<float>(myDummyCluster.getPad()) << ", " << static_cast<float>(myDummyCluster.getTime()) << ", "<< static_cast<float>(myDummyCluster.getSigmaPad()) << ", " << static_cast<float>(myDummyCluster.getSigmaTime()) << ", " << static_cast<float>(myDummyCluster.getQmax());
    //   printInput<float>(glo_idx * clusterer.nnClusterizerElementSize, clusterer.inputData32, clusterer);
    // }

    if (rejectCluster) {
      if (clusterer.mPclusterPosInRow) {
        clusterer.mPclusterPosInRow[full_glo_idx] = clusterer.mNMaxClusterPerRow;
      }
      return;
    }

    uint rowIndex = 0;
    if (clusterer.mPclusterByRow != nullptr) {
      rowIndex = GPUTPCCFClusterizer::sortIntoBuckets(
        clusterer,
        myCluster,
        clusterer.peakPositions[glo_idx].row(),
        clusterer.mNMaxClusterPerRow,
        clusterer.mPclusterInRow,
        clusterOut);
      if (clusterer.mPclusterPosInRow != nullptr) {
        clusterer.mPclusterPosInRow[full_glo_idx] = rowIndex;
      }
    } else if (clusterer.mPclusterPosInRow) {
      rowIndex = clusterer.mPclusterPosInRow[full_glo_idx];
    }
    CPU_ONLY(labelAcc->commit(clusterer.peakPositions[glo_idx].row(), rowIndex, clusterer.mNMaxClusterPerRow));
  } else {
    if (clusterer.mPclusterPosInRow) {
      clusterer.mPclusterPosInRow[full_glo_idx] = clusterer.mNMaxClusterPerRow;
    }
    return;
  }
}

// ---------------------------------
GPUd() void GPUTPCNNClusterizer::publishClustersReg2(uint glo_idx, GPUSharedMemory& smem, processorType& clusterer, int8_t dtype, int8_t onlyMC, uint batchStart)
{
  Array2D<PackedCharge> chargeMap(reinterpret_cast<PackedCharge*>(clusterer.mPchargeMap));
  CPU_ONLY(MCLabelAccumulator labelAccElem(clusterer));
  CPU_ONLY(MCLabelAccumulator* labelAcc = CPU_PTR(&labelAccElem));
  tpc::ClusterNative* clusterOut = (onlyMC) ? nullptr : clusterer.mPclusterByRow;
  uint full_glo_idx = glo_idx + batchStart;
  int model_output_index = glo_idx * clusterer.model_reg_2.getNumOutputNodes()[0][1];

  // LOG(info) << glo_idx << " -- " << model_output_index << " / " << clusterer.outputDataReg1.size() << " / " << clusterer.model_reg_1.getNumOutputNodes()[0][1] << " -- " << clusterer.peakPositions.size() << " -- " << clusterer.centralCharges.size();

  if (clusterer.outputDataClass[full_glo_idx] > 1) {

    ClusterAccumulator pc;

    if ((clusterer.mPmemory->fragment).isOverlap(clusterer.peakPositions[glo_idx].time())) {
      if (clusterer.mPclusterPosInRow) {
        clusterer.mPclusterPosInRow[full_glo_idx] = clusterer.mNMaxClusterPerRow;
      }
      return;
    }

    // Cluster 1
    pc.setFull(clusterer.centralCharges[glo_idx] * clusterer.outputDataReg2[model_output_index + 8],
      clusterer.peakPositions[glo_idx].pad() + clusterer.outputDataReg2[model_output_index],
      clusterer.outputDataReg2[model_output_index + 4],
      (clusterer.mPmemory->fragment).start + clusterer.peakPositions[glo_idx].time() + clusterer.outputDataReg2[model_output_index + 2],
      clusterer.outputDataReg2[model_output_index + 6],
      1, 1);

    tpc::ClusterNative myCluster;
    bool rejectCluster = !pc.toNative(clusterer.peakPositions[glo_idx], clusterer.centralCharges[glo_idx], myCluster, clusterer.Param(), chargeMap);
    if (rejectCluster) {
      if (clusterer.nnClusterizerVerbosity < 2) {
        LOG(warning) << "[NN, CF] Cluster rejected!";
      }
      if (clusterer.mPclusterPosInRow) {
        clusterer.mPclusterPosInRow[full_glo_idx] = clusterer.mNMaxClusterPerRow;
      }
      return;
    }

    uint rowIndex = 0;
    if (clusterer.mPclusterByRow != nullptr) {
      rowIndex = GPUTPCCFClusterizer::sortIntoBuckets(
        clusterer,
        myCluster,
        clusterer.peakPositions[glo_idx].row(),
        clusterer.mNMaxClusterPerRow,
        clusterer.mPclusterInRow,
        clusterOut);
      if (clusterer.mPclusterPosInRow != nullptr) {
        clusterer.mPclusterPosInRow[full_glo_idx] = rowIndex;
      }
    } else if (clusterer.mPclusterPosInRow) {
      rowIndex = clusterer.mPclusterPosInRow[full_glo_idx];
    }
    CPU_ONLY(labelAcc->commit(clusterer.peakPositions[glo_idx].row(), rowIndex, clusterer.mNMaxClusterPerRow));

    // Cluster 2
    pc.setFull(clusterer.centralCharges[glo_idx] * clusterer.outputDataReg2[model_output_index + 9],
      clusterer.peakPositions[glo_idx].pad() + clusterer.outputDataReg2[model_output_index + 1],
      clusterer.outputDataReg2[model_output_index + 5],
      (clusterer.mPmemory->fragment).start + clusterer.peakPositions[glo_idx].time() + clusterer.outputDataReg2[model_output_index + 3],
      clusterer.outputDataReg2[model_output_index + 7],
      1, 1);

    rejectCluster = !pc.toNative(clusterer.peakPositions[glo_idx], clusterer.centralCharges[glo_idx], myCluster, clusterer.Param(), chargeMap);
    if (rejectCluster) {
      if (clusterer.nnClusterizerVerbosity < 2) {
        LOG(warning) << "[NN, CF] Cluster rejected!";
      }
      if (clusterer.mPclusterPosInRow) {
        clusterer.mPclusterPosInRow[full_glo_idx] = clusterer.mNMaxClusterPerRow;
      }
      return;
    }

    if (clusterer.mPclusterByRow != nullptr) {
      rowIndex = GPUTPCCFClusterizer::sortIntoBuckets(
        clusterer,
        myCluster,
        clusterer.peakPositions[glo_idx].row(),
        clusterer.mNMaxClusterPerRow,
        clusterer.mPclusterInRow,
        clusterOut);
      if (clusterer.mPclusterPosInRow != nullptr) {
        clusterer.mPclusterPosInRow[full_glo_idx] = rowIndex;
      }
    } else if (clusterer.mPclusterPosInRow) {
      rowIndex = clusterer.mPclusterPosInRow[full_glo_idx];
    }
    // CPU_ONLY(labelAcc->commit(clusterer.peakPositions[glo_idx].row(), rowIndex, clusterer.mNMaxClusterPerRow)); // -> Is this needed? How to handle MC labels for split clusters?
  } else {
    if (clusterer.mPclusterPosInRow) {
      clusterer.mPclusterPosInRow[full_glo_idx] = clusterer.mNMaxClusterPerRow;
    }
    return;
  }
}

// ---------------------------------
void GPUTPCNNClusterizer::writeTrainingData(processorType& clusterer, int dtype)
{
  std::string outputFile = "custom_nn_training_data_reco_" + std::to_string(clusterer.mISector) + ".root"; // Fixed string concatenation
  TTree* tree = nullptr;
  TFile* file = nullptr;
  std::vector<std::string> branchNames;

  LOG(info) << "Writing training data to file " << outputFile;

  // Open file in UPDATE mode to check if it exists
  file = new TFile(outputFile.c_str(), "UPDATE");
  bool fileExists = file && !file->IsZombie();

  // Create branches if they don't exist
  std::vector<TBranch*> branches;
  std::vector<float> atomic_unit;

  if (fileExists) {
    tree = (TTree*)file->Get("tr_data");
    if (!tree) {
      LOG(error) << "Could not find tree in file " << outputFile;
      file->Close();
      return;
    }

    // Get existing branches and resize atomic_unit
    TObjArray* branch_list = tree->GetListOfBranches();
    atomic_unit.resize(branch_list->GetEntries());

    // Connect branches to atomic_unit elements
    for (int i = 0; i < branch_list->GetEntries(); i++) {
      TBranch* branch = (TBranch*)branch_list->At(i);
      tree->SetBranchAddress(branch->GetName(), &atomic_unit[i]);
      branches.push_back(branch);
    }
  } else {
    file = new TFile(outputFile.c_str(), "RECREATE");
    if (!file || file->IsZombie()) {
      LOG(error) << "Could not create new file " << outputFile;
      return;
    }
    tree = new TTree("tr_data", "Neural Network Input Data");

    if (branchNames.empty()) {
      // Generate default branch names if none provided
      int branch_idx = 0;
      for (int r = -clusterer.nnClusterizerSizeInputRow; r <= clusterer.nnClusterizerSizeInputRow; r++) {
        for (int p = -clusterer.nnClusterizerSizeInputPad; p <= clusterer.nnClusterizerSizeInputPad; p++) {
          for (int t = -clusterer.nnClusterizerSizeInputTime; t <= clusterer.nnClusterizerSizeInputTime; t++) {
            branchNames.push_back("in_row_" + std::to_string(r) + "_pad_" + std::to_string(p) + "_time_" + std::to_string(t));
            branch_idx++;
          }
        }
      }
    }
    branchNames.push_back("in_sector");
    branchNames.push_back("in_row");
    branchNames.push_back("in_pad");
    branchNames.push_back("idx_sector");
    branchNames.push_back("idx_row");
    branchNames.push_back("idx_pad");
    branchNames.push_back("idx_time");

    atomic_unit.resize(branchNames.size());
    for (size_t i = 0; i < branchNames.size(); i++) {
      branches.push_back(tree->Branch(branchNames[i].c_str(), &atomic_unit[i], (branchNames[i] + "/F").c_str()));
    }
  }

  // Fill tree
  uint size_element = branches.size();
  for (size_t entry = 0; entry < clusterer.mPmemory->counters.nClusters; entry++) {
    uint startIndex = entry * clusterer.nnClusterizerElementSize;

    for (size_t i = 0; i < clusterer.nnClusterizerElementSize; i++) {
      atomic_unit[i] = (dtype == 0) ?
                      static_cast<float>(clusterer.inputData16[startIndex + i]) :
                      clusterer.inputData32[startIndex + i];
    }

    // Additional indices
    if (clusterer.nnClusterizerAddIndexData) {
      atomic_unit[atomic_unit.size() - 4] = clusterer.mISector;
      atomic_unit[atomic_unit.size() - 3] = clusterer.peakPositions[entry].row();
      atomic_unit[atomic_unit.size() - 2] = clusterer.peakPositions[entry].pad();
      atomic_unit[atomic_unit.size() - 1] = clusterer.peakPositions[entry].time();
    }

    tree->Fill();
  }

  // Write and cleanup
  tree->Write("", TObject::kOverwrite);
  file->Close();
  delete file;
}

// ---------------------------------
void GPUTPCNNClusterizer::digitWriter(processorType& clusterer, std::string folder)
{

  ROOT::EnableThreadSafety();

  LOG(info) << "Streaming digits for NN clusterizer training, sector "
            << clusterer.mISector << ", fragment "
            << clusterer.mPmemory->fragment.index;

  if (gSystem->AccessPathName(folder.c_str())) {
    gSystem->mkdir(folder.c_str());
  }

  std::string outputFile = folder + "/tpcdigits_reco_" + std::to_string(clusterer.mISector) + ".root";
  TFile* file = TFile::Open(outputFile.c_str(), "UPDATE");

  if (!file || file->IsZombie()) {
    LOG(error) << "Error opening file: " << outputFile;
    return;
  }

  TTree* tree = (TTree*)file->Get("tr_data");
  std::vector<float> atomic_unit;
  std::vector<std::string> branchNames = {"sector", "row", "pad", "time", "charge", "has3x3Peak", "isSplit"};

  if (tree) {
    TObjArray* branch_list = tree->GetListOfBranches();
    int numBranches = branch_list->GetEntries();

    if (numBranches != branchNames.size()) {
      LOG(error) << "Mismatch in branch count: expected " << branchNames.size() << ", found " << numBranches;
      file->Close();
      delete file;
      return;
    }

    atomic_unit.resize(numBranches);
    for (int i = 0; i < numBranches; i++) {
      TBranch* branch = (TBranch*)branch_list->At(i);
      if (branch) {
        tree->SetBranchAddress(branch->GetName(), &atomic_unit[i]);
      }
    }
  } else {
    tree = new TTree("tr_data", "Neural Network Input Data");
    atomic_unit.resize(branchNames.size());

    for (size_t i = 0; i < branchNames.size(); i++) {
      tree->Branch(branchNames[i].c_str(), &atomic_unit[i], (branchNames[i] + "/F").c_str());
    }
  }

  Array2D<PackedCharge> chargeMap(reinterpret_cast<PackedCharge*>(clusterer.mPchargeMap));

  for (size_t entry = 0; entry < clusterer.mPmemory->counters.nPositions; entry++) {
    ChargePos pos = clusterer.mPpositions[entry];
    PackedCharge charge = chargeMap[pos];

    if (charge.unpack() > 0) {
      atomic_unit = {
        static_cast<float>(clusterer.mISector),
        static_cast<float>(pos.row()),
        static_cast<float>(pos.pad()),
        static_cast<float>(pos.time() + clusterer.mPmemory->fragment.start),
        static_cast<float>(charge.unpack()),
        static_cast<float>(charge.has3x3Peak()),
        static_cast<float>(charge.isSplit())
      };

      tree->Fill();
    }
  }

  tree->Write("", TObject::kOverwrite);
  file->Write();
  file->Close();
  delete file;

}

// ---------------------------------
void GPUTPCNNClusterizer::combineDigitFiles(int sector)
{
  LOG(info) << "Combining digit files for sector " << sector;

  std::string outputDir = "digits_stream";
  std::string outputFile = outputDir + "/tpcdigits_reco_" + std::to_string(sector) + ".root";

  // Check if the output directory exists
  if (gSystem->AccessPathName(outputDir.c_str())) {
    LOG(error) << "Output directory does not exist: " << outputDir;
    return;
  }

  // Find all fragment files for the given sector
  void* dir = gSystem->OpenDirectory(outputDir.c_str());
  if (!dir) {
    LOG(error) << "Error opening directory: " << outputDir;
    return;
  }

  // Combine the fragment files using hadd
  std::string haddCommand = "hadd -f " + outputFile + "tpcdigits_reco_" + std::to_string(sector) + "_*.root" ;

  int haddResult = gSystem->Exec(haddCommand.c_str());
  if (haddResult != 0) {
    LOG(error) << "Error combining files with hadd, command: " << haddCommand;
    return;
  }

  // Remove the individual fragment files
  gSystem->Exec(("rm -rf " + outputDir + "/tpcdigits_reco_" + std::to_string(sector) + "_*.root").c_str());

  LOG(info) << "Successfully combined digit files for sector " << sector << " into " << outputFile;
}

// ---------------------------------
template <>
GPUdii() void GPUTPCNNClusterizer::Thread<GPUTPCNNClusterizer::removeAllSplitFlags>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUSharedMemory& smem, processorType& clusterer, int8_t dtype, int8_t onlyMC, uint batchStart)
{
  int glo_idx = get_global_id(0);
  Array2D<PackedCharge> chargeMap(reinterpret_cast<PackedCharge*>(clusterer.mPchargeMap));
  ChargePos pos = clusterer.mPpositions[glo_idx];
  PackedCharge newCharge(chargeMap[pos].unpack(), chargeMap[pos].has3x3Peak(), false);
  chargeMap[pos] = newCharge;
}

// ---------------------------------
template <>
GPUdii() void GPUTPCNNClusterizer::Thread<GPUTPCNNClusterizer::setDeconvolutionFlags>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUSharedMemory& smem, processorType& clusterer, int8_t dtype, int8_t onlyMC, uint batchStart)
{
  Array2D<PackedCharge> chargeMap(reinterpret_cast<PackedCharge*>(clusterer.mPchargeMap));
  Array2D<uint8_t> peakMap(clusterer.mPpeakMap);
  SizeT idx = get_global_id(0);

  o2::gpu::GPUTPCCFDeconvolution::GPUSharedMemory smem2;

  bool iamDummy = (idx >= clusterer.mPmemory->counters.nPositions);
  idx = iamDummy ? clusterer.mPmemory->counters.nPositions - 1 : idx;

  ChargePos pos = clusterer.mPpositions[idx];

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
    peakCount = GPUTPCNNClusterizer::countPeaksInner(partId, smem2.buf, &aboveThreshold);
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
    peakCount = GPUTPCNNClusterizer::countPeaksOuter(partId, aboveThreshold, smem2.buf);
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
GPUdii() void GPUTPCNNClusterizer::Thread<GPUTPCNNClusterizer::publishDeconvolutionFlags>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUSharedMemory& smem, processorType& clusterer, int8_t dtype, int8_t onlyMC, uint batchStart)
{
  uint idx = get_global_id(0);
  Array2D<PackedCharge> chargeMap(reinterpret_cast<PackedCharge*>(clusterer.mPchargeMap));
  clusterer.clusterFlags[idx][0] = 0;
  clusterer.clusterFlags[idx][1] = 0;

  for (int p = -2; p <= 2; p++) {
    for (int t = -2; t <= 2; t++) {
      ChargePos d = clusterer.peakPositions[idx].delta({p,t});
      PackedCharge charge = chargeMap[d];
      if(std::abs(p) < 2 && std::abs(t) < 2){
        clusterer.clusterFlags[idx][0] += (t != 0 && charge.isSplit());
        clusterer.clusterFlags[idx][1] += (p != 0 && charge.isSplit());
      } else {
        clusterer.clusterFlags[idx][0] += (t != 0 && charge.isSplit() && !charge.has3x3Peak());
        clusterer.clusterFlags[idx][1] += (p != 0 && charge.isSplit() && !charge.has3x3Peak());
      }
    }
  }
}

GPUdi() uint8_t GPUTPCNNClusterizer::countPeaksInner(
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

GPUdi() uint8_t GPUTPCNNClusterizer::countPeaksOuter(
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
