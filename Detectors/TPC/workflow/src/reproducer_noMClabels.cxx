#include <cmath>
#include <boost/thread.hpp>
#include <stdlib.h>
#include <unordered_map>
#include <regex>
#include <chrono>
#include <thread>
#include <iostream>
#include <type_traits>
#include <tuple>

#include "Algorithm/RangeTokenizer.h"

#include "DataFormatsTPC/WorkflowHelper.h"
#include "DataFormatsTPC/ClusterNativeHelper.h"
#include "DataFormatsTPC/ClusterNative.h"
#include "DataFormatsTPC/ClusterGroupAttribute.h"
#include "DataFormatsTPC/Constants.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "DataFormatsGlobalTracking/TrackTuneParams.h"
#include "DataFormatsTPC/Defs.h"

#include "DetectorsRaw/HBFUtils.h"
#include "DetectorsBase/GRPGeomHelper.h"
#include "MathUtils/Utils.h"

#include "DPLUtils/RootTreeReader.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"

#include "GPUO2Interface.h"
#include "GPUO2InterfaceUtils.h"

#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/ConstMCTruthContainer.h"
#include "SimulationDataFormat/LabelContainer.h"
#include "SimulationDataFormat/IOMCTruthContainerView.h"
#include "SimulationDataFormat/MCTruthContainer.h"

#include "Framework/Logger.h"
#include "Framework/Task.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/CompletionPolicyHelpers.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/CallbacksPolicy.h"

#include "Headers/DataHeader.h"

#include "ML/onnx_interface.h"

#include "Steer/MCKinematicsReader.h"

#include "TPCWorkflow/ProcessingHelpers.h"
#include "TPCQC/Clusters.h"
#include "TPCBase/Painter.h"
#include "TPCBase/CalDet.h"
#include "TPCBase/Mapper.h"

#include "TFile.h"
#include "TTree.h"
#include "TKey.h"
#include "TSystem.h"
#include "TROOT.h"

using namespace o2;
using namespace GPUCA_NAMESPACE::gpu;
using namespace o2::tpc;
using namespace o2::framework;
using namespace o2::ml;
using namespace o2::base;
using namespace boost;

template <typename T>
using BranchDefinition = o2::framework::MakeRootTreeWriterSpec::BranchDefinition<T>;

struct customCluster {
  int sector = -1;
  int row = -1;
  int max_pad = -1;
  int max_time = -1;
  float cog_pad = -1.f;
  float cog_time = -1.f;
  float sigmaPad = -1.f;
  float sigmaTime = -1.f;
  float qMax = -1.f;
  float qTot = -1.f;
  uint8_t flag = 0;
  int mcTrkId = -1;
  int mcEvId = -1;
  int mcSrcId = -1;
  int index = -1;
  float label = 0.f; // holds e.g. the network class label / classification score
  float X = -1.f;
  float Y = -1.f;
  float Z = -1.f;

  ~customCluster(){}
};

namespace o2
{
namespace tpc
{

class repro : public Task
{
	public:
    repro(std::vector<int> tpc_sectors) { tpcSectors = tpc_sectors; };
		void init(InitContext&) final;
		void read_native(int, std::vector<customCluster>&);
		void write_custom_native(ProcessingContext&, std::vector<customCluster>&, bool = true);
		void run(ProcessingContext&) final;

	private:
		std::string inFileNative = "tpc-native-clusters.root";
		std::vector<int> tpcSectors;
    std::vector<customCluster> native_writer_map;

};
}
}

void repro::init(InitContext& ic)
{
  inFileNative = ic.options().get<std::string>("infile-native");
}

void repro::read_native(int sector, std::vector<customCluster>& native_map)
{

  ClusterNativeHelper::Reader tpcClusterReader;
  tpcClusterReader.init(inFileNative.c_str());

  ClusterNativeAccess clusterIndex;
  std::unique_ptr<ClusterNative[]> clusterBuffer;
  memset(&clusterIndex, 0, sizeof(clusterIndex));
  o2::tpc::ClusterNativeHelper::ConstMCLabelContainerViewWithBuffer clusterMCBuffer;

  qc::Clusters clusters;
  float current_time = 0, current_pad = 0, current_row = 0;

  for (unsigned long i = 0; i < tpcClusterReader.getTreeSize(); ++i) {
    tpcClusterReader.read(i);
    tpcClusterReader.fillIndex(clusterIndex, clusterBuffer, clusterMCBuffer);

    int nClustersSec = 0;
    for (int irow = 0; irow < o2::tpc::constants::MAXGLOBALPADROW; ++irow) {
      nClustersSec += clusterIndex.nClusters[sector][irow];
    }
    LOG(info) << "Native clusters in sector " << sector << ": " << nClustersSec;
    
    int count_clusters = 0;
    for (int irow = 0; irow < o2::tpc::constants::MAXGLOBALPADROW; ++irow) {
      count_clusters += clusterIndex.nClusters[sector][irow];
    }

    native_map.resize(count_clusters);
    count_clusters = 0;

    for (int irow = 0; irow < o2::tpc::constants::MAXGLOBALPADROW; ++irow) {
      const unsigned long nClusters = clusterIndex.nClusters[sector][irow];
      for (int icl = 0; icl < nClusters; ++icl) {
        const auto& cl = *(clusterIndex.clusters[sector][irow] + icl);
        clusters.processCluster(cl, Sector(sector), irow);
        current_pad = cl.getPad();
        current_time = cl.getTime();
        native_map[count_clusters] = customCluster{sector, irow, (int)round(current_pad), (int)round(current_time), current_pad, current_time, cl.getSigmaPad(), cl.getSigmaTime(), (float)cl.getQmax(), (float)cl.getQtot(), cl.getFlags(), -1, -1, -1, count_clusters, 0.f};
      }
    }
  }
}

// ---------------------------------
void repro::write_custom_native(ProcessingContext& pc, std::vector<customCluster>& native_map, bool perSector)
{

  // using MCLabelContainer = o2::dataformats::MCTruthContainer<o2::MCCompLabel>;

  std::vector<ClusterNativeContainer> cont(o2::tpc::constants::MAXSECTOR * o2::tpc::constants::MAXGLOBALPADROW);
  std::vector<o2::dataformats::MCLabelContainer> mcTruth(o2::tpc::constants::MAXSECTOR * o2::tpc::constants::MAXGLOBALPADROW);

  std::vector<std::vector<int>> cluster_sector_counter(o2::tpc::constants::MAXSECTOR, std::vector<int>(o2::tpc::constants::MAXGLOBALPADROW, 0));

  for (auto cls : native_map) {
    cluster_sector_counter[cls.sector][cls.row]++;
  }
  for (int sec = 0; sec < o2::tpc::constants::MAXSECTOR; sec++) {
    for (int row = 0; row < o2::tpc::constants::MAXGLOBALPADROW; row++) {
      cont[sec * o2::tpc::constants::MAXGLOBALPADROW + row].sector = sec;
      cont[sec * o2::tpc::constants::MAXGLOBALPADROW + row].globalPadRow = row;
      cont[sec * o2::tpc::constants::MAXGLOBALPADROW + row].clusters.resize(cluster_sector_counter[sec][row]);
    }
  }

  int total_clusters = 0;
  o2::dataformats::MCLabelContainer mcTruthBuffer;
  std::vector<o2::dataformats::MCLabelContainer> sorted_mc_labels(36);
  std::vector<int> sector_counter(36, 0);
  o2::MCCompLabel dummyMcLabel(0,0,0,true); // This is the dummy label for which the QA fails
  for (auto const cls : native_map) {
    int sec = cls.sector;
    int row = cls.row;
    // cont[sec*o2::tpc::constants::MAXGLOBALPADROW + row].clusters[cluster_sector_counter[sec][row]].setTime(cls[3]);
    cont[sec * o2::tpc::constants::MAXGLOBALPADROW + row].clusters[cluster_sector_counter[sec][row]].setTimeFlags(cls.cog_time, cls.flag);
    cont[sec * o2::tpc::constants::MAXGLOBALPADROW + row].clusters[cluster_sector_counter[sec][row]].setPad(cls.cog_pad);
    cont[sec * o2::tpc::constants::MAXGLOBALPADROW + row].clusters[cluster_sector_counter[sec][row]].setSigmaTime(cls.sigmaTime);
    cont[sec * o2::tpc::constants::MAXGLOBALPADROW + row].clusters[cluster_sector_counter[sec][row]].setSigmaPad(cls.sigmaPad);
    cont[sec * o2::tpc::constants::MAXGLOBALPADROW + row].clusters[cluster_sector_counter[sec][row]].qMax = cls.qMax;
    cont[sec * o2::tpc::constants::MAXGLOBALPADROW + row].clusters[cluster_sector_counter[sec][row]].qTot = cls.qTot;
    if(cls.mcTrkId != -1 && (total_clusters % 10 != 0)){
      mcTruth[sec * o2::tpc::constants::MAXGLOBALPADROW + row].addElement(cluster_sector_counter[sec][row], o2::MCCompLabel(cls.mcTrkId, cls.mcEvId, cls.mcSrcId, false));
      sorted_mc_labels[sec].addElement(sector_counter[sec], o2::MCCompLabel(cls.mcTrkId, cls.mcEvId, cls.mcSrcId, false));
      // mcTruthBuffer.addElement(total_clusters, o2::MCCompLabel(cls.mcTrkId, cls.mcEvId, cls.mcSrcId, false));
    } else {
      mcTruth[sec * o2::tpc::constants::MAXGLOBALPADROW + row].addElement(cluster_sector_counter[sec][row], dummyMcLabel);
      sorted_mc_labels[sec].addElement(sector_counter[sec], dummyMcLabel);
      // mcTruthBuffer.addElement(total_clusters, dummyMcLabel);
    }
    cluster_sector_counter[sec][row]++;
    sector_counter[sec]++;
    total_clusters++;
  }

  std::unique_ptr<ClusterNative[]> clusterBuffer;
  std::unique_ptr<ClusterNativeAccess> clusters = ClusterNativeHelper::createClusterNativeIndex(clusterBuffer, cont, &mcTruthBuffer, &mcTruth);

  LOG(info) << "ClusterNativeAccess structure created.";

  o2::tpc::ClusterNativeAccess const& clusterIndex = *(clusters.get());

  if(perSector){
    // Clusters are shipped by sector, we are copying into per-sector buffers (anyway only for ROOT output)
    o2::tpc::TPCSectorHeader clusterOutputSectorHeader{0};
    for (unsigned int i : tpcSectors) {
      unsigned int subspec = i;
      clusterOutputSectorHeader.sectorBits = (1ul << i);
      char* buffer = pc.outputs().make<char>({o2::header::gDataOriginTPC, "CLUSTERNATIVE", subspec, {clusterOutputSectorHeader}}, clusterIndex.nClustersSector[i] * sizeof(*clusterIndex.clustersLinear) + sizeof(o2::tpc::ClusterCountIndex)).data();
      o2::tpc::ClusterCountIndex* outIndex = reinterpret_cast<o2::tpc::ClusterCountIndex*>(buffer);
      memset(outIndex, 0, sizeof(*outIndex));
      for (int j = 0; j < o2::tpc::constants::MAXGLOBALPADROW; j++) {
        outIndex->nClusters[i][j] = clusterIndex.nClusters[i][j];
      }
      memcpy(buffer + sizeof(*outIndex), clusterIndex.clusters[i][0], clusterIndex.nClustersSector[i] * sizeof(*clusterIndex.clustersLinear));

      o2::dataformats::ConstMCLabelContainer contflat;
      sorted_mc_labels[i].flatten_to(contflat);
      pc.outputs().snapshot({o2::header::gDataOriginTPC, "CLNATIVEMCLBL", subspec, {clusterOutputSectorHeader}}, contflat);
    }
  }

  LOG(info) << "------- Native clusters structure written -------";
}

// ---------------------------------
void repro::run(ProcessingContext& pc)
{

  std::vector<customCluster> native_map;
	for(int s : tpcSectors){
		read_native(s, native_map);
    int native_writer_map_size = native_writer_map.size();
    int total_counter = 0;
    for(auto const cls : native_map){
      total_counter++;
    }
    native_writer_map.resize(native_writer_map_size + total_counter);
    total_counter = 0;
    for(auto const cls : native_map){
      native_writer_map[native_writer_map_size + total_counter] = cls;
      total_counter++;
    }
    native_map.clear();
	}

	write_custom_native(pc, native_writer_map);

	pc.services().get<ControlService>().endOfStream();
	pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
}


void customize(std::vector<o2::framework::CompletionPolicy>& policies)
{
  policies.push_back(o2::framework::CompletionPolicyHelpers::consumeWhenAllOrdered(".*(?:TPC|tpc).*[w,W]riter.*"));
}

void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  std::vector<ConfigParamSpec> options{
    {"tpc-sectors", VariantType::String, "0-35", {"TPC sector range, e.g. 5-7,8,9"}},
    {"outfile-native", VariantType::String, "tpc-native-clusters-custom.root", {"Path to native file"}}
  };
  std::swap(workflowOptions, options);
}

// ---------------------------------
#include "Framework/runDataProcessing.h"

DataProcessorSpec processRepro(ConfigContext const& cfgc, std::vector<InputSpec>& inputs, std::vector<OutputSpec>& outputs)
{
  // setOutputAllocator("CLUSTERNATIVE", true, outputRegions.clustersNative, std::make_tuple(gDataOriginTPC, mSpecConfig.sendClustersPerSector ? (DataDescription) "CLUSTERNATIVETMP" : (DataDescription) "CLUSTERNATIVE", NSectors, clusterOutputSectorHeader), sizeof(o2::tpc::ClusterCountIndex));
  for (int i : o2::RangeTokenizer::tokenize<int>(cfgc.options().get<std::string>("tpc-sectors"))) {
    outputs.emplace_back(o2::header::gDataOriginTPC, "CLUSTERNATIVE", i, Lifetime::Timeframe); // Dropping incomplete Lifetime::Transient?
    outputs.emplace_back(o2::header::gDataOriginTPC, "CLNATIVEMCLBL", i, Lifetime::Timeframe); // Dropping incomplete Lifetime::Transient?
  }

  return DataProcessorSpec{
    "tpc-reproducer",
    inputs,
    outputs,
    adaptFromTask<repro>(o2::RangeTokenizer::tokenize<int>(cfgc.options().get<std::string>("tpc-sectors"))),
    Options{
      {"infile-native", VariantType::String, "tpc-native-clusters.root", {"Path to native file"}},
    }};
}

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{

  WorkflowSpec specs;

  static o2::framework::Output gDispatchTrigger{"", ""};
  static std::vector<InputSpec> inputs;
  static std::vector<OutputSpec> outputs;

  gDispatchTrigger = o2::framework::Output{"TPC", "CLUSTERNATIVE"};

  // --- Functions writing to the WorkflowSpec ---

  // QA task
  specs.push_back(processRepro(cfgc, inputs, outputs));

  // Native writer

  std::vector<int> tpcSectors = o2::RangeTokenizer::tokenize<int>(cfgc.options().get<std::string>("tpc-sectors"));
  std::vector<int> laneConfiguration = tpcSectors;

  auto getIndex = [tpcSectors](o2::framework::DataRef const& ref) {
    auto const* tpcSectorHeader = o2::framework::DataRefUtils::getHeader<o2::tpc::TPCSectorHeader*>(ref);
    if (!tpcSectorHeader) {
      throw std::runtime_error("TPC sector header missing in header stack");
    }
    if (tpcSectorHeader->sector() < 0) {
      // special data sets, don't write
      return ~(size_t)0;
    }
    size_t index = 0;
    for (auto const& sector : tpcSectors) {
      if (sector == tpcSectorHeader->sector()) {
        return index;
      }
      index += 1;
    }
    throw std::runtime_error("sector " + std::to_string(tpcSectorHeader->sector()) + " not configured for writing");
  };

  auto getName = [tpcSectors](std::string base, size_t index) {
    // LOG(info) << "Writer publishing for sector " << tpcSectors.at(index);
    return base + "_" + std::to_string(tpcSectors.at(index));
  };

  auto fillLabels = [](TBranch& branch, std::vector<char> const& labelbuffer, DataRef const& /*ref*/) {
    o2::dataformats::ConstMCTruthContainerView<o2::MCCompLabel> labels(labelbuffer);
    o2::dataformats::IOMCTruthContainerView outputcontainer;
    auto ptr = &outputcontainer;
    auto br = o2::framework::RootTreeWriter::remapBranch(branch, &ptr);
    outputcontainer.adopt(labelbuffer);
    br->Fill();
    br->ResetAddress();
  };

  auto makeWriterSpec = [tpcSectors, laneConfiguration, getIndex, getName, fillLabels](const char* processName,
                                                                                        const char* defaultFileName,
                                                                                        const char* defaultTreeName,
                                                                                        auto&& databranch,
                                                                                        auto&& mcbranch,
                                                                                        bool singleBranch = false) {
    if (tpcSectors.size() == 0) {
      throw std::invalid_argument(std::string("writer process configuration needs list of TPC sectors"));
    }

    auto amendInput = [tpcSectors, laneConfiguration](InputSpec& input, size_t index) {
      input.binding += std::to_string(laneConfiguration[index]);
      DataSpecUtils::updateMatchingSubspec(input, laneConfiguration[index]);
    };
    auto amendBranchDef = [laneConfiguration, amendInput, tpcSectors, getIndex, getName, singleBranch](auto&& def, bool enableMC = true) {
      if (!singleBranch) {
        def.keys = mergeInputs(def.keys, laneConfiguration.size(), amendInput);
        // the branch is disabled if set to 0
        def.nofBranches = tpcSectors.size();
        def.getIndex = getIndex;
        def.getName = getName;
        return std::move(def);
      } else {
        // instead of the separate sector branches only one is going to be written
        def.nofBranches = enableMC ? 1 : 0;
      }
    };

    return std::move(MakeRootTreeWriterSpec(processName, defaultFileName, defaultTreeName,
                                            std::move(amendBranchDef(databranch)),
                                            std::move(amendBranchDef(mcbranch)))());
  };
  // ---

  specs.push_back(makeWriterSpec("tpc-custom-native-writer",
                                (cfgc.options().get<std::string>("outfile-native")).c_str(),
                                  "tpcrec",
                                  BranchDefinition<const char*>{InputSpec{"data", ConcreteDataTypeMatcher{"TPC", o2::header::DataDescription("CLUSTERNATIVE")}},
                                                                "TPCClusterNative",
                                                                "databranch"},
                                  BranchDefinition<std::vector<char>>{InputSpec{"mc", ConcreteDataTypeMatcher{"TPC", o2::header::DataDescription("CLNATIVEMCLBL")}},
                                                                      "TPCClusterNativeMCTruth",
                                                                      "mcbranch", fillLabels}));

  return specs;
}