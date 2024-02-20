#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "TFile.h"
#include "TTree.h"
#include "TStyle.h"
#include "TCanvas.h"
#include "TH1.h"
#include "TH2.h"
#include "TPCQC/ClustersTracksQA.h"
#endif

using namespace o2::tpc;
using namespace o2::tpc::qc;

void runClusterTracksQa()
{

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

  ~customCluster(){}
};

struct tabular_data {
  int sector = 0;
  float X_max = 0.f;
  float Y_max = 0.f;
  float T_max = 0.f;
  float X_cog = 0.f;
  float Y_cog = 0.f;
  float T_cog = 0.f;
  float label = -1;

  ~tabular_data(){}
};

  qc::ClustersTracksQa ClusterQA;

  std::cout << "Creating tabular data vector..." << std::endl;
  int tabular_data_counter = 0;
  std::vector<tabular_data> tabular_data_vec;

  std::cout << "Loading clusters..." << std::endl;
  ClusterNativeAccess clusters = ClusterQA.loadClusters("tpc-native-clusters.root");

  std::cout << "Loading tracks..." << std::endl;
  std::vector<TrackTPC> tpc_tracks = ClusterQA.loadTracks("tpctracks.root");

  std::cout << "Setting B-field..." << std::endl;
  float B_field = -5.f; // kiloGauss
  float y, z;
  int tracks_counter = 0;

  // tabular_data_vec.resize(tpc_tracks.size()*o2::tpc::constants::MAXGLOBALPADROW);
  // for(auto trk : tpc_tracks){
  //   for(int row = 0; row < o2::tpc::constants::MAXGLOBALPADROW; row++){ // this needs to be defined -> x = x_coordinate of row.
  //     trk.getYZAt(GPUTPCGeometry::Row2X(row), B_field, y, z);
  //     tabular_data_vec[tabular_data_counter] = {0, GPUTPCGeometry::Row2X(row), y, z, -1};
  //     tabular_data_counter++;
  //   }
  //   tracks_counter++;
  // }

  std::cout << "Reading custom clusters from file..." << std::endl;
  std::vector<customCluster> custom_clusters;
  ClusterQA.ReadClustersFromRootFile("custom_clusters.root", custom_clusters);
  tabular_data_vec.resize(tabular_data_vec.size() + custom_clusters.size());
  for(auto const cls : custom_clusters){
    GlobalPosition2D posMax = ClusterQA.convertSecRowPadToXY(cls.sector, cls.row, cls.max_pad);
    GlobalPosition2D posCog = ClusterQA.convertSecRowPadToXY(cls.sector, cls.row, cls.cog_pad);
    tabular_data_vec[tabular_data_counter] = {cls.sector, posMax.X(), posMax.Y(), (float)cls.max_time, posCog.X(), posCog.Y(), cls.cog_time, cls.label};
  }

  std::cout << "Writing tabular data..." << std::endl;
  std::stringstream file_in;
  file_in << "tabular_clusters.root";
  TFile* outputFileTabularClusters = new TFile(file_in.str().c_str(), "RECREATE");
  TTree* tabular_clusters = new TTree("clusters", "tree");

  tabular_data tmp_tab_data;
  tabular_clusters->Branch("sector", &tmp_tab_data.sector);
  tabular_clusters->Branch("X_max", &tmp_tab_data.X_max);
  tabular_clusters->Branch("Y_max", &tmp_tab_data.Y_max);
  tabular_clusters->Branch("T_max", &tmp_tab_data.T_max);
  tabular_clusters->Branch("X_cog", &tmp_tab_data.X_cog);
  tabular_clusters->Branch("Y_cog", &tmp_tab_data.Y_cog);
  tabular_clusters->Branch("T_cog", &tmp_tab_data.T_cog);
  tabular_clusters->Branch("flag", &tmp_tab_data.label);

  int elem_counter = 0;
  for (auto const elem : tabular_data_vec) {
    tmp_tab_data = elem;
    tabular_clusters->Fill();
  }

  tabular_clusters->Write();
  outputFileTabularClusters->Close();
}
