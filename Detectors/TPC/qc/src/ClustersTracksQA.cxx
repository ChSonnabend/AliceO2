#include "DataFormatsTPC/ClusterNativeHelper.h"
#include "DataFormatsTPC/ClusterNative.h"
#include "DataFormatsTPC/Constants.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "TPCBase/Painter.h"
#include "TPCBase/Mapper.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "TPCQC/ClustersTracksQA.h"

ClassImp(o2::tpc::qc::ClustersTracksQa);

using namespace o2::tpc;
using namespace o2::tpc::qc;

template<class T>
void ClustersTracksQa::ReadClustersFromRootFile(std::string fileName, std::vector<T>& clusters) {
    // Open the ROOT file
    TFile *file = new TFile(fileName.c_str());
    if (!file || file->IsZombie()) {
        printf("Error: Cannot open file %s\n", fileName);
        return;
    }

    // Get the TTree from the file
    TTree *tree;
    file->GetObject("customClusters", tree);
    if (!tree) {
        printf("Error: Cannot find TTree in file %s\n", fileName);
        return;
    }

    // Associate the branch with the struct
    T data;
    TBranch *branch = tree->GetBranch("customClusters");
    if (!branch) {
        printf("Error: Cannot find TBranch in TTree\n");
        return;
    }

    branch->SetAddress(&data);

    // Read and print the data
    int nEntries = tree->GetEntries();
    clusters.resize(nEntries);
    for (int i = 0; i < nEntries; ++i) {
        branch->GetEntry(i);
        clusters[i] = data;
    }

    // Clean up
    file->Close();
};

std::vector<TrackTPC> ClustersTracksQa::loadTracks(std::string_view inputFileName){
    // ===| track file and tree |=================================================
  auto file = TFile::Open(inputFileName.data());
  auto tree = (TTree*)file->Get("tpcrec");
  if (tree == nullptr) {
    std::cout << "Error getting tree\n";
    return {};
  }

  // ===| branch setup |==========================================================
  std::vector<TrackTPC>* tpcTracks = nullptr;
  tree->SetBranchAddress("TPCTracks", &tpcTracks);

  return *tpcTracks;
}

ClusterNativeAccess ClustersTracksQa::loadClusters(std::string_view fileName)
{
  ClusterNativeHelper::Reader tpcClusterReader;
  tpcClusterReader.init(fileName.data());

  ClusterNativeAccess clusterIndex;
  std::unique_ptr<ClusterNative[]> clusterBuffer;
  memset(&clusterIndex, 0, sizeof(clusterIndex));
  o2::tpc::ClusterNativeHelper::ConstMCLabelContainerViewWithBuffer clusterMCBuffer;

  for (int i = 0; i < tpcClusterReader.getTreeSize(); ++i) {
    std::cout << "Event " << i << "\n";
    tpcClusterReader.read(i);
    tpcClusterReader.fillIndex(clusterIndex, clusterBuffer, clusterMCBuffer);
    size_t iClusters = 0;
    for (int isector = 0; isector < constants::MAXSECTOR; ++isector) {
      for (int irow = 0; irow < constants::MAXGLOBALPADROW; ++irow) {
        const int nClusters = clusterIndex.nClusters[isector][irow];
        if (!nClusters) {
          continue;
        }
      }
    }
  }

  return clusterIndex;
}

GlobalPosition2D ClustersTracksQa::convertSecRowPadToXY(int sector, int row, float pad){

    const auto& mapper = Mapper::instance();

  int firstRegion = 0, lastRegion = 10;
  if (row < 63) {
    firstRegion = 0;
    lastRegion = 4;
  } else {
    firstRegion = 4;
    lastRegion = 10;
  }

  GlobalPosition2D pos = mapper.getPadCentre(PadSecPos(sector, row, pad));
  float fractionalPad = 0;
  if(int(pad) != pad){
    fractionalPad = mapper.getPadRegionInfo(firstRegion).getPadWidth()*(pad - int(pad));
  }
  return GlobalPosition2D(pos.X() + fractionalPad, pos.Y());
}