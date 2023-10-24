#include <cmath>
#include <boost/thread.hpp>
#include <stdlib.h>

#include "ML/onnx_interface.h"

#include "DetectorsRaw/HBFUtils.h"

#include "DataFormatsTPC/WorkflowHelper.h"
#include "DataFormatsTPC/ClusterNativeHelper.h"
#include "DataFormatsTPC/ClusterNative.h"
#include "DataFormatsTPC/ClusterGroupAttribute.h"
#include "DataFormatsTPC/Constants.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "DataFormatsGlobalTracking/TrackTuneParams.h"

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

#include "Headers/DataHeader.h"

#include "Steer/MCKinematicsReader.h"

#include "TPCWorkflow/ProcessingHelpers.h"
#include "TPCQC/Clusters.h"
#include "TPCBase/Painter.h"
#include "TPCBase/CalDet.h"

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
using namespace boost;

class qaIdeal : public Task
{
 public:
  template <typename T>
  int sign(T);

  void init(InitContext&) final;
  void setGeomFromTxt(std::string, std::vector<int> = {152, 14});
  int padOffset(int);
  int rowOffset(int);
  bool isBoundary(int, int);
  bool checkIdx(int);
  void read_digits(int, std::vector<std::array<int, 3>>&, std::vector<float>&);
  void read_ideal(int, std::vector<std::array<int, 3>>&, std::vector<float>&, std::vector<std::array<float, 3>>&, std::vector<std::array<float, 2>>&, std::vector<float>&);
  void read_native(int, std::vector<std::array<int, 3>>&, std::vector<std::array<float, 3>>&, std::vector<float>&);
  void read_network(int, std::vector<std::array<int, 3>>&, std::vector<float>&);

  template <class T>
  T init_map2d(int);

  template <class T>
  void fill_map2d(int, T&, std::vector<std::array<int, 3>>&, std::vector<std::array<int, 3>>&, std::vector<float>&, std::vector<std::array<float, 3>>&, std::vector<float>&, int = 0);

  template <class T>
  void find_maxima(int, T&, std::vector<int>&, std::vector<float>&);

  template <class T>
  bool is_local_minimum(T&, std::array<int, 3>&, std::vector<float>&);

  template <class T>
  int local_saddlepoint(T&, std::array<int, 3>&, std::vector<float>&);

  template <class T>
  void native_clusterizer(T&, std::vector<std::array<int, 3>>&, std::vector<int>&, std::vector<float>&, std::vector<std::array<float, 3>>&, std::vector<float>&);

  template <class T>
  std::vector<std::vector<std::vector<int>>> looper_tagger(int, T&, std::vector<float>&, std::vector<int>&, std::string);

  template <class T>
  void remove_loopers(int, std::vector<std::vector<std::vector<int>>>&, T&, std::vector<int>&);

  template <class T>
  void run_network(int, T&, std::vector<int>&, std::vector<std::array<int, 3>>&, std::vector<float>&, std::vector<std::array<float, 3>>&, int = 0);

  template <class T>
  void overwrite_map2d(int, T&, std::vector<std::array<int, 3>>&, std::vector<int>&, int = 0);

  template <class T>
  int test_neighbour(std::array<int, 3>, std::array<int, 2>, T&, int = 1);

  void runQa(int);
  void run(ProcessingContext&) final;

 private:
  std::vector<int> global_shift = {5, 5, 0};  // shifting digits to select windows easier, (pad, time, row)
  int charge_limits[2] = {2, 1024};           // upper and lower charge limits
  int verbose = 0;                            // chunk_size in time direction
  int create_output = 1;                      // Create output files specific for any mode
  int dim = 2;                                // Dimensionality of the training data
  int networkInputSize = 1000;                // vector input size for neural network
  float networkClassThres = 0.5f;             // Threshold where network decides to keep / reject digit maximum
  bool networkOptimizations = true;           // ONNX session optimizations
  int networkNumThreads = 1;                  // Future: Add Cuda and CoreML Execution providers to run on CPU
  int numThreads = 1;                         // Number of cores for multithreading
  int use_max_cog = 1;                        // 0 = use ideal maxima position; 1 = use ideal CoG position (rounded) for assignment
  int normalization_mode = 1;                 // Normalization of the charge: 0 = divide by 1024; 1 = divide by central charge
  int looper_tagger_granularity = 5;          // Granularity of looper tagger (time bins in which loopers are excluded in rectangular areas)
  int looper_tagger_timewindow = 20;          // Total time-window size of the looper tagger for evaluating if a region is looper or not
  int looper_tagger_padwindow = 3;            // Total pad-window size of the looper tagger for evaluating if a region is looper or not
  int looper_tagger_threshold_num = 5;        // Threshold of number of clusters over which rejection takes place
  float looper_tagger_threshold_q = 700.f;    // Threshold of charge-per-cluster that should be rejected
  std::string looper_tagger_opmode = "digit"; // Operational mode of the looper tagger

  std::array<int, o2::tpc::constants::MAXSECTOR> max_time, max_pad;
  std::string mode = "training_data";
  std::string inFileDigits = "tpcdigits.root";
  std::string inFileNative = "tpc-cluster-native.root";
  std::string inFileDigitizer = "mclabels_digitizer.root";
  std::string networkDataOutput = "./network_out.root";
  std::string networkClassification = "./net_classification.onnx";
  std::string networkRegression = "./net_regression.onnx";

  std::vector<std::vector<std::array<int, 2>>> adj_mat = {{{0, 0}}, {{1, 0}, {0, 1}, {-1, 0}, {0, -1}}, {{1, 1}, {-1, 1}, {-1, -1}, {1, -1}}, {{2, 0}, {0, -2}, {-2, 0}, {0, 2}}, {{2, 1}, {1, 2}, {-1, 2}, {-2, 1}, {-2, -1}, {-1, -2}, {1, -2}, {2, -1}}, {{2, 2}, {-2, 2}, {-2, -2}, {2, -2}}};

  std::array<std::array<unsigned int, 25>, o2::tpc::constants::MAXSECTOR> assignments_ideal, assignments_digit, assignments_ideal_findable, assignments_digit_findable;
  std::array<unsigned int, o2::tpc::constants::MAXSECTOR> number_of_ideal_max, number_of_digit_max, number_of_ideal_max_findable, clones;
  std::array<float, o2::tpc::constants::MAXSECTOR> fractional_clones;

  std::vector<std::vector<float>> TPC_GEOM;

  OnnxModel network_classification, network_regression;

  float landau_approx(float x)
  {
    return (1.f / TMath::Sqrt(2.f * (float)TMath::Pi())) * TMath::Exp(-(x + TMath::Exp(-x)) / 2.f) + 0.005; // +0.005 for regularization
  }
};

// ---------------------------------
template <typename T>
int qaIdeal::sign(T val)
{
  return (T(0) < val) - (val < T(0));
}

// ---------------------------------
void qaIdeal::setGeomFromTxt(std::string inputFile, std::vector<int> size)
{

  std::ifstream infile(inputFile.c_str(), std::ifstream::in);
  if (!infile.is_open()) {
    LOGP(fatal, "Could not open file {}!", inputFile);
  }

  // Col 0 -> INDEX (0 - 5279)
  // Col 1 -> PADROW (0 - 62)
  // Col 2 -> PAD (Np-1)
  // Col 3 -> X coordinate (mm)
  // Col 4 -> y coordinate (mm)
  // Col 5 -> Connector (1 - 132)
  // Col 6 -> Pin (1 - 40)
  // Col 7 -> Partion (0 - 1)
  // Col 8 -> Region (0 - 3)
  // Col 9 -> FEC (0 - 32)
  // Col 10 -> FEC Connector (0 - 3)
  // Col 11 -> FEC Channel (0 - 159)
  // Col 12 -> SAMPA Chip (0 - 4)
  // Col 13 -> SAMPA Channel (0 - 31)

  TPC_GEOM.resize(size[0]);
  std::fill(TPC_GEOM.begin(), TPC_GEOM.end(), std::vector<float>(size[1], 0));
  int trace = 0;

  std::string line;
  while (std::getline(infile, line)) {
    std::stringstream streamLine(line);
    streamLine >> TPC_GEOM[trace][0] >> TPC_GEOM[trace][1] >> TPC_GEOM[trace][2] >> TPC_GEOM[trace][3] >> TPC_GEOM[trace][4] >> TPC_GEOM[trace][5] >> TPC_GEOM[trace][6] >> TPC_GEOM[trace][7] >> TPC_GEOM[trace][8] >> TPC_GEOM[trace][9] >> TPC_GEOM[trace][10] >> TPC_GEOM[trace][11] >> TPC_GEOM[trace][12] >> TPC_GEOM[trace][13];
    trace++;
  }
}

// ---------------------------------
int qaIdeal::padOffset(int row)
{
  return (int)((TPC_GEOM[151][2] - TPC_GEOM[row][2]) / 2);
}

// ---------------------------------
int qaIdeal::rowOffset(int row)
{
  if (row <= 62) {
    return 0;
  } else {
    return global_shift[2];
  }
}

// ---------------------------------
bool qaIdeal::isBoundary(int row, int pad)
{
  if (row < 0 || pad < 0) {
    return true;
  } else if (row <= 62) {
    if (pad < (TPC_GEOM[151][2] - TPC_GEOM[row][2]) / 2 || pad > (TPC_GEOM[151][2] + TPC_GEOM[row][2]) / 2) {
      return true;
    } else {
      return false;
    }
  } else if (row <= 62 + global_shift[2]) {
    return true;
  } else if (row <= 151 + global_shift[2]) {
    if (pad < (TPC_GEOM[151][2] - TPC_GEOM[row - global_shift[2]][2]) / 2 || pad > (TPC_GEOM[151][2] + TPC_GEOM[row - global_shift[2]][2]) / 2) {
      return true;
    } else {
      return false;
    }
  } else if (row > 151 + global_shift[2]) {
    return true;
  } else {
    return false;
  }
}

// ---------------------------------
void qaIdeal::init(InitContext& ic)
{
  verbose = ic.options().get<int>("verbose");
  mode = ic.options().get<std::string>("mode");
  create_output = ic.options().get<int>("create-output");
  use_max_cog = ic.options().get<int>("use-max-cog");
  global_shift[0] = (int)((ic.options().get<int>("size-pad") - 1.f) / 2.f);
  global_shift[1] = (int)((ic.options().get<int>("size-time") - 1.f) / 2.f);
  global_shift[2] = (int)((ic.options().get<int>("size-row") - 1.f) / 2.f);
  numThreads = ic.options().get<int>("threads");
  inFileDigits = ic.options().get<std::string>("infile-digits");
  inFileNative = ic.options().get<std::string>("infile-native");
  networkDataOutput = ic.options().get<std::string>("network-data-output");
  networkClassification = ic.options().get<std::string>("network-classification-path");
  networkRegression = ic.options().get<std::string>("network-regression-path");
  networkInputSize = ic.options().get<int>("network-input-size");
  networkClassThres = ic.options().get<float>("network-class-threshold");
  networkOptimizations = ic.options().get<bool>("enable-network-optimizations");
  networkNumThreads = ic.options().get<int>("network-num-threads");
  normalization_mode = ic.options().get<int>("normalization-mode");
  looper_tagger_granularity = ic.options().get<int>("looper-tagger-granularity");
  looper_tagger_timewindow = ic.options().get<int>("looper-tagger-timewindow");
  looper_tagger_padwindow = ic.options().get<int>("looper-tagger-padwindow");
  looper_tagger_threshold_num = ic.options().get<int>("looper-tagger-threshold-num");
  looper_tagger_threshold_q = ic.options().get<float>("looper-tagger-threshold-q");
  looper_tagger_opmode = ic.options().get<std::string>("looper-tagger-opmode");

  ROOT::EnableThreadSafety();

  // LOG(info) << "Testing networks!";
  // std::vector<float> temp_input(10 * (2 * global_shift[0] + 1) * (2 * global_shift[1] + 1), 0);
  // float* out_net_class = network_classification.inference(temp_input, 10);
  //
  // for(int i = 0; i < 10; i++){
  //   LOG(info) << "Output of classification network (" << "): " << out_net_class[i];
  // }

  if (mode.find(std::string("network_class")) != std::string::npos || mode.find(std::string("network_full")) != std::string::npos) {
    network_classification.init(networkClassification, networkOptimizations, networkNumThreads);
  }
  if (mode.find(std::string("network_reg")) != std::string::npos || mode.find(std::string("network_full")) != std::string::npos) {
    network_regression.init(networkRegression, networkOptimizations, networkNumThreads);
  }

  const char* aliceO2env = std::getenv("O2_ROOT");
  if (aliceO2env) {
    std::string geom_file = aliceO2env;
    geom_file += "/share/Detectors/TPC/files/PAD_ROW_MAX.txt";
    setGeomFromTxt(geom_file);
    LOG(info) << "Geometry set!";
  } else {
    LOG(fatal) << "ALICE O2 environment not found!";
  }
  if (verbose >= 1)
    LOG(info) << "Initialized QA macro, ready to go!";
}

// ---------------------------------
bool qaIdeal::checkIdx(int idx)
{
  return (idx > -1);
}

// ---------------------------------
void qaIdeal::read_digits(int sector, std::vector<std::array<int, 3>>& digit_map, std::vector<float>& digit_q)
{

  if (verbose >= 1)
    LOG(info) << "Reading the digits...";

  // reading in the raw digit information
  TFile* digitFile = TFile::Open(inFileDigits.c_str());
  TTree* digitTree = (TTree*)digitFile->Get("o2sim");

  std::vector<o2::tpc::Digit>* digits = nullptr;
  int current_time = 0, current_pad = 0, current_row = 0;

  std::string branch_name = fmt::format("TPCDigit_{:d}", sector).c_str();
  digitTree->SetBranchAddress(branch_name.c_str(), &digits);

  int counter = 0;
  digitTree->GetEntry(0);

  digit_map.resize(digits->size());
  digit_q.resize(digits->size());
  // digit_isNoise.resize(digits->size());
  // digit_isQED.resize(digits->size());
  // digit_isValid.resize(digits->size());

  for (unsigned int i_digit = 0; i_digit < digits->size(); i_digit++) {
    const auto& digit = (*digits)[i_digit];
    current_time = digit.getTimeStamp();
    current_pad = digit.getPad();
    current_row = digit.getRow();

    if (current_time >= max_time[sector])
      max_time[sector] = current_time + 1;
    if (current_pad >= max_pad[sector])
      max_pad[sector] = current_pad + 1;

    digit_map[i_digit][0] = current_row;
    digit_map[i_digit][1] = current_pad;
    digit_map[i_digit][2] = current_time;
    digit_q[i_digit] = digit.getChargeFloat();
    // digit_isNoise[i_digit] = digit.isNoise();
    // digit_isQED[i_digit] = digit.isQED();
    // digit_isValid[i_digit] = digit.isValid();
    counter++;
  }
  (*digits).clear();

  digitFile->Close();

  if (verbose >= 1)
    LOG(info) << "Done reading digits!";
}

// ---------------------------------
void qaIdeal::read_native(int sector, std::vector<std::array<int, 3>>& digit_map, std::vector<std::array<float, 3>>& native_map, std::vector<float>& digit_q)
{

  if (verbose >= 1)
    LOG(info) << "Reading native clusters...";

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
    if (verbose >= 3) {
      LOG(info) << "Native clusters in sector " << sector << ": " << nClustersSec;
    }
    digit_map.resize(nClustersSec);
    digit_q.resize(nClustersSec);
    native_map.resize(nClustersSec);
    int count_clusters = 0;
    for (int irow = 0; irow < o2::tpc::constants::MAXGLOBALPADROW; ++irow) {
      const unsigned long nClusters = clusterIndex.nClusters[sector][irow];
      if (!nClusters) {
        continue;
      }
      for (int icl = 0; icl < nClusters; ++icl) {
        const auto& cl = *(clusterIndex.clusters[sector][irow] + icl);
        clusters.processCluster(cl, Sector(sector), irow);
        current_pad = cl.getPad();
        current_time = cl.getTime();

        native_map[count_clusters][0] = (float)irow;
        native_map[count_clusters][1] = current_pad;
        native_map[count_clusters][2] = current_time;
        digit_map[count_clusters][0] = irow;
        digit_map[count_clusters][1] = static_cast<int>(round(current_pad));
        digit_map[count_clusters][2] = static_cast<int>(round(current_time));
        digit_q[count_clusters] = cl.getQtot();

        if (current_time >= max_time[sector])
          max_time[sector] = current_time + 1;
        if (current_pad >= max_pad[sector])
          max_pad[sector] = current_pad + 1;
        count_clusters++;
      }
    }
  }

  if (verbose >= 1)
    LOG(info) << "Done reading native clusters!";
}

// ---------------------------------
void qaIdeal::read_network(int sector, std::vector<std::array<int, 3>>& digit_map, std::vector<float>& digit_q)
{

  //////// Deprecated ////////

  if (verbose >= 1)
    LOG(info) << "Reading network output...";

  // reading in the raw digit information
  TFile* networkFile = TFile::Open(networkDataOutput.c_str());
  TTree* networkTree = (TTree*)networkFile->Get("data_tree");

  double sec, row, pad, time, reg_pad, reg_time, reg_qRatio;
  std::vector<int> sizes(o2::tpc::constants::MAXSECTOR, 0);

  networkTree->SetBranchAddress("sector", &sec);
  networkTree->SetBranchAddress("row", &row);
  networkTree->SetBranchAddress("pad", &pad);
  networkTree->SetBranchAddress("time", &time);
  networkTree->SetBranchAddress("reg_pad", &reg_pad);
  networkTree->SetBranchAddress("reg_time", &reg_time);
  networkTree->SetBranchAddress("reg_qRatio", &reg_qRatio);

  for (unsigned int j = 0; j < networkTree->GetEntries(); j++) {
    try {
      networkTree->GetEntry(j);
      sizes[sector]++;
      if (round(time + reg_time) > max_time[sector])
        max_time[sector] = round(time + reg_time);
      if (round(pad + reg_pad) > max_pad[sector])
        max_pad[sector] = round(pad + reg_pad);
    } catch (...) {
      LOG(info) << "(Digitizer) Problem occured in sector " << sector;
    }
  }

  digit_map.resize(sizes[sector]);
  digit_q.resize(sizes[sector]);

  std::fill(sizes.begin(), sizes.end(), 0);

  for (unsigned int j = 0; j < networkTree->GetEntries(); j++) {
    try {
      networkTree->GetEntry(j);
      digit_map[sizes[sector]] = std::array<int, 3>{(int)row, static_cast<int>(round(pad + reg_pad)), static_cast<int>(round(time + reg_time))};
      digit_q[sizes[sector]] = 1000;
      sizes[sector]++;
    } catch (...) {
      LOG(info) << "(Digitizer) Problem occured in sector " << sector;
    }
  }

  networkFile->Close();

  if (verbose >= 1)
    LOG(info) << "Done reading digits!";
}

// ---------------------------------
void qaIdeal::read_ideal(int sector, std::vector<std::array<int, 3>>& ideal_max_map, std::vector<float>& ideal_max_q, std::vector<std::array<float, 3>>& ideal_cog_map, std::vector<std::array<float, 2>>& ideal_sigma_map, std::vector<float>& ideal_cog_q)
{

  int sec, row, maxp, maxt, pcount, lab;
  float cogp, cogt, cogq, maxq, sigmap, sigmat;
  int elements = 0;

  if (verbose > 0)
    LOG(info) << "Reading ideal clusters, sector " << sector << " ...";
  std::stringstream tmp_file;
  tmp_file << "mclabels_digitizer_" << sector << ".root";
  auto inputFile = TFile::Open(tmp_file.str().c_str());
  std::stringstream tmp_sec;
  tmp_sec << "sector_" << sector;
  auto digitizerSector = (TTree*)inputFile->Get(tmp_sec.str().c_str());

  // digitizerSector->SetBranchAddress("cluster_sector", &sec);
  digitizerSector->SetBranchAddress("cluster_row", &row);
  digitizerSector->SetBranchAddress("cluster_cog_pad", &cogp);
  digitizerSector->SetBranchAddress("cluster_cog_time", &cogt);
  digitizerSector->SetBranchAddress("cluster_cog_q", &cogq);
  digitizerSector->SetBranchAddress("cluster_max_pad", &maxp);
  digitizerSector->SetBranchAddress("cluster_max_time", &maxt);
  digitizerSector->SetBranchAddress("cluster_sigma_pad", &sigmap);
  digitizerSector->SetBranchAddress("cluster_sigma_time", &sigmat);
  digitizerSector->SetBranchAddress("cluster_max_q", &maxq);
  // digitizerSector->SetBranchAddress("cluster_points", &pcount);

  ideal_max_map.resize(digitizerSector->GetEntries());
  ideal_max_q.resize(digitizerSector->GetEntries());
  ideal_cog_map.resize(digitizerSector->GetEntries());
  ideal_sigma_map.resize(digitizerSector->GetEntries());
  ideal_cog_q.resize(digitizerSector->GetEntries());

  if (verbose >= 3)
    LOG(info) << "Trying to read " << digitizerSector->GetEntries() << " ideal digits";
  for (unsigned int j = 0; j < digitizerSector->GetEntries(); j++) {
    try {
      digitizerSector->GetEntry(j);
      // ideal_point_count.push_back(pcount);

      ideal_max_map[j] = std::array<int, 3>{row, maxp, maxt};
      ideal_max_q[j] = maxq;
      ideal_cog_map[j] = std::array<float, 3>{(float)row, cogp, cogt};
      ideal_sigma_map[j] = std::array<float, 2>{sigmap, sigmat};
      ideal_cog_q[j] = cogq;
      elements++;

      if (maxt >= max_time[sector])
        max_time[sector] = maxt + 1;
      if (maxp >= max_pad[sector])
        max_pad[sector] = maxp + 1;
      if (std::ceil(cogt) >= max_time[sector])
        max_time[sector] = std::ceil(cogt) + 1;
      if (std::ceil(cogp) >= max_pad[sector])
        max_pad[sector] = std::ceil(cogp) + 1;

    } catch (...) {
      LOG(info) << "(Digitizer) Problem occured in sector " << sector;
    }
  }
  inputFile->Close();
}

template <class T>
T qaIdeal::init_map2d(int sector)
{
  T map2d;
  for (int i = 0; i < 2; i++) {
    map2d[i].resize(max_time[sector] + (2 * global_shift[1]));
    for (int time_size = 0; time_size < max_time[sector] + (2 * global_shift[1]); time_size++) {
      map2d[i][time_size].resize(o2::tpc::constants::MAXGLOBALPADROW + (3 * global_shift[2]));
      for (int row = 0; row < o2::tpc::constants::MAXGLOBALPADROW + (3 * global_shift[2]); row++) {
        // Support for IROC - OROC1 transition: Add rows after row 62 + global_shift[2]
        map2d[i][time_size][row].resize(138 + 2 * global_shift[0]);
        for (int pad = 0; pad < 138 + 2 * global_shift[0]; pad++) {
          map2d[i][time_size][row][pad] = -1;
        }
      }
    };
  };

  if (verbose >= 1)
    LOG(info) << "Initialized 2D map! Time size is " << map2d[0].size();

  return map2d;
}

// ---------------------------------
template <class T>
void qaIdeal::fill_map2d(int sector, T& map2d, std::vector<std::array<int, 3>>& digit_map, std::vector<std::array<int, 3>>& ideal_max_map, std::vector<float>& ideal_max_q, std::vector<std::array<float, 3>>& ideal_cog_map, std::vector<float>& ideal_cog_q, int fillmode)
{

  int* map_ptr = nullptr;
  if (use_max_cog == 0) {
    // Storing the indices
    if (fillmode == 0 || fillmode == -1) {
      for (unsigned int ind = 0; ind < digit_map.size(); ind++) {
        map2d[1][digit_map[ind][2] + global_shift[1]][digit_map[ind][0] + rowOffset(digit_map[ind][0]) + global_shift[2]][digit_map[ind][1] + global_shift[0] + padOffset(digit_map[ind][0])] = ind;
      }
    }
    if (fillmode == 1 || fillmode == -1) {
      for (unsigned int ind = 0; ind < ideal_max_map.size(); ind++) {
        map_ptr = &map2d[0][ideal_max_map[ind][2] + global_shift[1]][ideal_max_map[ind][0] + rowOffset(ideal_max_map[ind][0]) + global_shift[2]][ideal_max_map[ind][1] + global_shift[0] + padOffset(ideal_max_map[ind][0])];
        if (*map_ptr == -1 || ideal_max_q[ind] > ideal_max_q[*map_ptr]) { // Using short-circuiting for second expression
          if (*map_ptr != -1 && verbose >= 4) {
            LOG(warning) << "Conflict detected! Current MaxQ : " << ideal_max_q[*map_ptr] << "; New MaxQ: " << ideal_max_q[ind] << "; Index " << ind << "/" << ideal_max_map.size();
          }
          *map_ptr = ind;
        }
      }
    }
    if (fillmode < -1 || fillmode > 1) {
      LOG(info) << "Fillmode unknown! No fill performed!";
    }
  } else if (use_max_cog == 1) {
    // Storing the indices
    if (fillmode == 0 || fillmode == -1) {
      for (unsigned int ind = 0; ind < digit_map.size(); ind++) {
        map2d[1][digit_map[ind][2] + global_shift[1]][digit_map[ind][0] + rowOffset(digit_map[ind][0]) + global_shift[2]][digit_map[ind][1] + global_shift[0] + padOffset(digit_map[ind][0])] = ind;
      }
    }
    if (fillmode == 1 || fillmode == -1) {
      for (unsigned int ind = 0; ind < ideal_cog_map.size(); ind++) {
        map_ptr = &map2d[0][round(ideal_cog_map[ind][2]) + global_shift[1]][round(ideal_cog_map[ind][0]) + rowOffset(round(ideal_cog_map[ind][0])) + global_shift[2]][round(ideal_cog_map[ind][1]) + global_shift[0] + padOffset((int)ideal_cog_map[ind][0])];
        if (*map_ptr == -1 || ideal_cog_q[ind] > ideal_cog_q[*map_ptr]) {
          if (*map_ptr != -1 && verbose >= 4) {
            LOG(warning) << "Conflict detected! Current CoGQ : " << ideal_cog_q[*map_ptr] << "; New CoGQ: " << ideal_cog_q[ind] << "; Index " << ind << "/" << ideal_cog_map.size();
          }
          *map_ptr = ind;
        }
      }
    }
    if (fillmode < -1 || fillmode > 1) {
      LOG(info) << "Fillmode unknown! No fill performed!";
    }
  }
}

// ---------------------------------
template <class T>
void qaIdeal::find_maxima(int sector, T& map2d, std::vector<int>& maxima_digits, std::vector<float>& digit_q)
{

  if (verbose >= 1) {
    LOG(info) << "Finding local maxima";
  }

  bool is_max = true;
  float current_charge = 0;
  int row_offset = 0, pad_offset = 0;
  for (int row = 0; row < o2::tpc::constants::MAXGLOBALPADROW; row++) {
    if (verbose >= 3)
      LOG(info) << "Finding maxima in row " << row;
    for (int pad = 0; pad < TPC_GEOM[row][2] + 1; pad++) { // -> Needs fixing
      row_offset = rowOffset(row);
      pad_offset = padOffset(row);
      for (int time = 0; time < max_time[sector]; time++) {
        if (checkIdx(map2d[1][time + global_shift[1]][row + row_offset + global_shift[2]][pad + global_shift[0] + pad_offset])) {

          current_charge = digit_q[map2d[1][time + global_shift[1]][row + row_offset + global_shift[2]][pad + global_shift[0] + pad_offset]];

          if (map2d[1][time + global_shift[1]][row + row_offset + global_shift[2]][pad + global_shift[0] + pad_offset + 1] != -1) {
            is_max = (current_charge >= digit_q[map2d[1][time + global_shift[1]][row + row_offset + global_shift[2]][pad + global_shift[0] + pad_offset + 1]]);
          }

          if (is_max && map2d[1][time + global_shift[1] + 1][row + row_offset + global_shift[2]][pad + global_shift[0] + pad_offset] != -1) {
            is_max = (current_charge >= digit_q[map2d[1][time + global_shift[1] + 1][row + row_offset + global_shift[2]][pad + global_shift[0] + pad_offset]]);
          }

          if (is_max && map2d[1][time + global_shift[1]][row + row_offset + global_shift[2]][pad + global_shift[0] + pad_offset - 1] != -1) {
            is_max = (current_charge >= digit_q[map2d[1][time + global_shift[1]][row + row_offset + global_shift[2]][pad + global_shift[0] + pad_offset - 1]]);
          }

          if (is_max && map2d[1][time + global_shift[1] - 1][row + row_offset + global_shift[2]][pad + global_shift[0] + pad_offset] != -1) {
            is_max = (current_charge >= digit_q[map2d[1][time + global_shift[1] - 1][row + row_offset + global_shift[2]][pad + global_shift[0] + pad_offset]]);
          }

          if (is_max && map2d[1][time + global_shift[1] + 1][row + row_offset + global_shift[2]][pad + global_shift[0] + pad_offset + 1] != -1) {
            is_max = (current_charge >= digit_q[map2d[1][time + global_shift[1] + 1][row + row_offset + global_shift[2]][pad + global_shift[0] + pad_offset + 1]]);
          }

          if (is_max && map2d[1][time + global_shift[1] - 1][row + row_offset + global_shift[2]][pad + global_shift[0] + pad_offset + 1] != -1) {
            is_max = (current_charge >= digit_q[map2d[1][time + global_shift[1] - 1][row + row_offset + global_shift[2]][pad + global_shift[0] + pad_offset + 1]]);
          }

          if (is_max && map2d[1][time + global_shift[1] + 1][row + row_offset + global_shift[2]][pad + global_shift[0] + pad_offset - 1] != -1) {
            is_max = (current_charge >= digit_q[map2d[1][time + global_shift[1] + 1][row + row_offset + global_shift[2]][pad + global_shift[0] + pad_offset - 1]]);
          }

          if (is_max && map2d[1][time + global_shift[1] - 1][row + row_offset + global_shift[2]][pad + global_shift[0] + pad_offset - 1] != -1) {
            is_max = (current_charge >= digit_q[map2d[1][time + global_shift[1] - 1][row + row_offset + global_shift[2]][pad + global_shift[0] + pad_offset - 1]]);
          }

          if (is_max) {
            maxima_digits.push_back(map2d[1][time + global_shift[1]][row + row_offset + global_shift[2]][pad + global_shift[0] + pad_offset]);
          }
          is_max = true;
        }
      }
    }
    if (verbose >= 3)
      LOG(info) << "Found " << maxima_digits.size() << " maxima in row " << row;
  }

  if (verbose >= 1)
    LOG(info) << "Found " << maxima_digits.size() << " maxima. Done!";
}

// ---------------------------------
template <class T>
bool qaIdeal::is_local_minimum(T& map2d, std::array<int, 3>& current_position, std::vector<float>& digit_q)
{

  bool is_min = false;
  float current_charge = 0;
  int row = current_position[0], pad = current_position[1], time = current_position[2];
  int row_offset = rowOffset(current_position[0]);
  int pad_offset = padOffset(current_position[0]);
  if (checkIdx(map2d[1][time + global_shift[1]][row + row_offset + global_shift[2]][pad + global_shift[0] + pad_offset])) {

    current_charge = digit_q[map2d[1][time + global_shift[1]][row + row_offset + global_shift[2]][pad + global_shift[0] + pad_offset]];

    if (map2d[1][time + global_shift[1]][row + row_offset + global_shift[2]][pad + global_shift[0] + pad_offset + 1] != -1) {
      is_min = (current_charge < digit_q[map2d[1][time + global_shift[1]][row + row_offset + global_shift[2]][pad + global_shift[0] + pad_offset + 1]]);
    }

    if (is_min && map2d[1][time + global_shift[1] + 1][row + row_offset + global_shift[2]][pad + global_shift[0] + pad_offset] != -1) {
      is_min = (current_charge < digit_q[map2d[1][time + global_shift[1] + 1][row + row_offset + global_shift[2]][pad + global_shift[0] + pad_offset]]);
    }

    if (is_min && map2d[1][time + global_shift[1]][row + row_offset + global_shift[2]][pad + global_shift[0] + pad_offset - 1] != -1) {
      is_min = (current_charge < digit_q[map2d[1][time + global_shift[1]][row + row_offset + global_shift[2]][pad + global_shift[0] + pad_offset - 1]]);
    }

    if (is_min && map2d[1][time + global_shift[1] - 1][row + row_offset + global_shift[2]][pad + global_shift[0] + pad_offset] != -1) {
      is_min = (current_charge < digit_q[map2d[1][time + global_shift[1] - 1][row + row_offset + global_shift[2]][pad + global_shift[0] + pad_offset]]);
    }

    if (is_min && map2d[1][time + global_shift[1] + 1][row + row_offset + global_shift[2]][pad + global_shift[0] + pad_offset + 1] != -1) {
      is_min = (current_charge < digit_q[map2d[1][time + global_shift[1] + 1][row + row_offset + global_shift[2]][pad + global_shift[0] + pad_offset + 1]]);
    }

    if (is_min && map2d[1][time + global_shift[1] - 1][row + row_offset + global_shift[2]][pad + global_shift[0] + pad_offset + 1] != -1) {
      is_min = (current_charge < digit_q[map2d[1][time + global_shift[1] - 1][row + row_offset + global_shift[2]][pad + global_shift[0] + pad_offset + 1]]);
    }

    if (is_min && map2d[1][time + global_shift[1] + 1][row + row_offset + global_shift[2]][pad + global_shift[0] + pad_offset - 1] != -1) {
      is_min = (current_charge < digit_q[map2d[1][time + global_shift[1] + 1][row + row_offset + global_shift[2]][pad + global_shift[0] + pad_offset - 1]]);
    }

    if (is_min && map2d[1][time + global_shift[1] - 1][row + row_offset + global_shift[2]][pad + global_shift[0] + pad_offset - 1] != -1) {
      is_min = (current_charge < digit_q[map2d[1][time + global_shift[1] - 1][row + row_offset + global_shift[2]][pad + global_shift[0] + pad_offset - 1]]);
    }
  }

  return is_min;
}

// ---------------------------------
template <class T>
int qaIdeal::local_saddlepoint(T& map2d, std::array<int, 3>& current_position, std::vector<float>& digit_q)
{

  // returns 0 if no saddlepoint, returns 1 if saddlepoint rising in pad direction, returns 2 if saddlepoint rising in time directino

  bool saddlepoint = false;
  int saddlepoint_mode = 0;
  float current_charge = 0;
  int row = current_position[0], pad = current_position[1], time = current_position[2];
  int row_offset = rowOffset(row);
  int pad_offset = padOffset(row);
  if (checkIdx(map2d[1][time + global_shift[1]][row + row_offset + global_shift[2]][pad + global_shift[0] + pad_offset])) {

    current_charge = digit_q[map2d[1][time + global_shift[1]][row + row_offset + global_shift[2]][pad + global_shift[0] + pad_offset]];

    if (map2d[1][time + global_shift[1]][row + row_offset + global_shift[2]][pad + global_shift[0] + pad_offset + 1] != -1) {
      saddlepoint = (current_charge < digit_q[map2d[1][time + global_shift[1]][row + row_offset + global_shift[2]][pad + global_shift[0] + pad_offset + 1]]);
      if (map2d[1][time + global_shift[1]][row + row_offset + global_shift[2]][pad + global_shift[0] + pad_offset - 1] != -1)
        saddlepoint = saddlepoint && (current_charge < digit_q[map2d[1][time + global_shift[1]][row + row_offset + global_shift[2]][pad + global_shift[0] + pad_offset - 1]]);
      if (map2d[1][time + global_shift[1] + 1][row + row_offset + global_shift[2]][pad + global_shift[0] + pad_offset] != -1)
        saddlepoint = saddlepoint && (current_charge > digit_q[map2d[1][time + global_shift[1] + 1][row + row_offset + global_shift[2]][pad + global_shift[0] + pad_offset]]);
      if (map2d[1][time + global_shift[1] - 1][row + row_offset + global_shift[2]][pad + global_shift[0] + pad_offset] != -1)
        saddlepoint = saddlepoint && (current_charge > digit_q[map2d[1][time + global_shift[1] - 1][row + row_offset + global_shift[2]][pad + global_shift[0] + pad_offset]]);
      if (saddlepoint)
        saddlepoint_mode = 1;
    }

    if (!saddlepoint && map2d[1][time + global_shift[1] + 1][row + row_offset + global_shift[2]][pad + global_shift[0] + pad_offset] != -1) {
      saddlepoint = (current_charge < digit_q[map2d[1][time + global_shift[1] + 1][row + row_offset + global_shift[2]][pad + global_shift[0] + pad_offset]]);
      if (map2d[1][time + global_shift[1] - 1][row + row_offset + global_shift[2]][pad + global_shift[0] + pad_offset] != -1)
        saddlepoint = saddlepoint && (current_charge < digit_q[map2d[1][time + global_shift[1] - 1][row + row_offset + global_shift[2]][pad + global_shift[0] + pad_offset]]);
      if (map2d[1][time + global_shift[1]][row + row_offset + global_shift[2]][pad + global_shift[0] + pad_offset + 1] != -1)
        saddlepoint = saddlepoint && (current_charge > digit_q[map2d[1][time + global_shift[1]][row + row_offset + global_shift[2]][pad + global_shift[0] + pad_offset + 1]]);
      if (map2d[1][time + global_shift[1]][row + row_offset + global_shift[2]][pad + global_shift[0] + pad_offset - 1] != -1)
        saddlepoint = saddlepoint && (current_charge > digit_q[map2d[1][time + global_shift[1]][row + row_offset + global_shift[2]][pad + global_shift[0] + pad_offset - 1]]);
      if (saddlepoint)
        saddlepoint_mode = 2;
    }
  }
  return saddlepoint_mode;
}

// ---------------------------------
template <class T>
void qaIdeal::native_clusterizer(T& map2d, std::vector<std::array<int, 3>>& digit_map, std::vector<int>& maxima_digits, std::vector<float>& digit_q, std::vector<std::array<float, 3>>& digit_clusterizer_map, std::vector<float>& digit_clusterizer_q)
{

  digit_clusterizer_map.clear();
  digit_clusterizer_q.clear();

  std::vector<std::array<int, 3>> digit_map_tmp = digit_map;
  digit_map.clear();
  std::vector<float> digit_q_tmp = digit_q;
  digit_q.clear();

  digit_clusterizer_map.resize(maxima_digits.size());
  digit_clusterizer_q.resize(maxima_digits.size());
  digit_map.resize(maxima_digits.size());
  digit_q.resize(maxima_digits.size());

  int row, pad, time, row_offset, pad_offset;
  float cog_charge = 0, current_charge = 0;
  std::array<float, 3> cog_position{}; // pad, time
  std::array<int, 3> current_pos{};

  std::array<std::array<int, 3>, 3> found_min_saddle{};
  std::array<std::array<int, 5>, 5> investigate{};
  std::array<int, 2> adjusted_elem;

  for (int max_pos = 0; max_pos < maxima_digits.size(); max_pos++) {

    row = digit_map_tmp[maxima_digits[max_pos]][0];
    pad = digit_map_tmp[maxima_digits[max_pos]][1];
    time = digit_map_tmp[maxima_digits[max_pos]][2];
    row_offset = rowOffset(row);
    pad_offset = padOffset(row);

    for (int layer = 0; layer < adj_mat.size(); layer++) {
      for (auto elem : adj_mat[layer]) {
        if (map2d[1][time + global_shift[1] + elem[1]][row + row_offset + global_shift[2]][pad + global_shift[0] + pad_offset + elem[0]] != -1) {
          current_charge = digit_q_tmp[map2d[1][time + global_shift[1] + elem[1]][row + row_offset + global_shift[2]][pad + global_shift[0] + pad_offset + elem[0]]];
        } else {
          current_charge = 0;
        }
        current_pos = {row, pad + elem[0], time + elem[1]};
        cog_position = {(float)row, 0.f, 0.f};

        adjusted_elem[0] = elem[0] + 2;
        adjusted_elem[1] = elem[1] + 2;

        if (layer <= 2) {
          found_min_saddle[adjusted_elem[0]][adjusted_elem[1]] = is_local_minimum(map2d, current_pos, digit_q_tmp) + 2 * local_saddlepoint(map2d, current_pos, digit_q_tmp); // 1 == min, 2 == saddlepoint, rising in pad, 4 = saddlepoint, rising in time
          if (layer == 0 || !found_min_saddle[adjusted_elem[0]][adjusted_elem[1]]) {
            cog_charge += current_charge;
            cog_position[1] += elem[0] * current_charge;
            cog_position[2] += elem[1] * current_charge;
          } else {
            if (std::abs(elem[0]) && std::abs(elem[1]) && current_charge > 0) {
              if (found_min_saddle[adjusted_elem[0]][adjusted_elem[1]] == 1) {
                investigate[sign(elem[0]) * (std::abs(elem[0]) + 1) + 2][adjusted_elem[1]] = 1;
                investigate[elem[0] + 2][sign(elem[1]) * (std::abs(elem[1]) + 1) + 2] = 1;
                investigate[sign(elem[0]) * (std::abs(elem[0]) + 1) + 2][sign(elem[1]) * (std::abs(elem[1]) + 1) + 2] = 1;
              } else if (found_min_saddle[adjusted_elem[0]][adjusted_elem[1]] == 2) {
                investigate[adjusted_elem[0]][sign(elem[1]) * (std::abs(elem[1]) + 1) + 2] = 2;
                investigate[sign(elem[0]) * (std::abs(elem[0]) + 1) + 2][sign(elem[1]) * (std::abs(elem[1]) + 1) + 2] = 2;
              } else if (found_min_saddle[adjusted_elem[0]][adjusted_elem[1]] == 4) {
                investigate[sign(elem[0]) * (std::abs(elem[0]) + 1) + 2][adjusted_elem[1]] = 4;
                investigate[sign(elem[0]) * (std::abs(elem[0]) + 1) + 2][sign(elem[1]) * (std::abs(elem[1]) + 1) + 2] = 4;
              }
            } else {
              if (std::abs(elem[0]) && current_charge > 0) {
                if (found_min_saddle[adjusted_elem[0]][adjusted_elem[1]] == 1 || found_min_saddle[adjusted_elem[0]][adjusted_elem[1]] == 2) {
                  investigate[sign(elem[0]) * (std::abs(elem[0]) + 1) + 2][sign(elem[1]) * (std::abs(elem[1]) - 1) + 2] = 1;
                  investigate[sign(elem[0]) * (std::abs(elem[0]) + 1) + 2][adjusted_elem[1]] = 1;
                  investigate[sign(elem[0]) * (std::abs(elem[0]) + 1) + 2][sign(elem[1]) * (std::abs(elem[1]) + 1) + 2] = 1;
                } else if (found_min_saddle[adjusted_elem[0]][adjusted_elem[1]] == 4) {
                  investigate[sign(elem[0]) * (std::abs(elem[0]) + 1) + 2][sign(elem[1]) * (std::abs(elem[1]) - 1) + 2] = 4;
                  investigate[sign(elem[0]) * (std::abs(elem[0]) + 1) + 2][adjusted_elem[1]] = 4;
                  investigate[sign(elem[0]) * (std::abs(elem[0]) + 1) + 2][sign(elem[1]) * (std::abs(elem[1]) + 1) + 2] = 4;
                }
              } else if (std::abs(elem[1]) && current_charge > 0) {
                if (found_min_saddle[adjusted_elem[0]][adjusted_elem[1]] == 1 || found_min_saddle[adjusted_elem[0]][adjusted_elem[1]] == 2) {
                  investigate[sign(elem[0]) * (std::abs(elem[0]) + 1) + 2][sign(elem[1]) * (std::abs(elem[1]) + 1) + 2] = 1;
                  investigate[adjusted_elem[0]][sign(elem[1]) * (std::abs(elem[1]) + 1) + 2] = 1;
                  investigate[sign(elem[0]) * (std::abs(elem[0]) - 1) + 2][sign(elem[1]) * (std::abs(elem[1]) + 1) + 2] = 1;
                } else if (found_min_saddle[adjusted_elem[0]][adjusted_elem[1]] == 4) {
                  investigate[sign(elem[0]) * (std::abs(elem[0]) + 1) + 2][sign(elem[1]) * (std::abs(elem[1]) + 1) + 2] = 4;
                  investigate[adjusted_elem[0]][sign(elem[1]) * (std::abs(elem[1]) + 1) + 2] = 4;
                  investigate[sign(elem[0]) * (std::abs(elem[0]) - 1) + 2][sign(elem[1]) * (std::abs(elem[1]) + 1) + 2] = 4;
                }
              }
            }
            cog_charge += current_charge / 2.f;
            cog_position[1] += elem[0] * (current_charge / 2.f);
            cog_position[2] += elem[1] * (current_charge / 2.f);
          }
        } else {
          if ((investigate[adjusted_elem[0]][adjusted_elem[1]] == 0) && !(is_local_minimum(map2d, current_pos, digit_q_tmp) || local_saddlepoint(map2d, current_pos, digit_q_tmp)) && current_charge > 0) {
            cog_charge += current_charge;
            cog_position[1] += elem[0] * current_charge;
            cog_position[2] += elem[1] * current_charge;
          } else if (investigate[adjusted_elem[0]][adjusted_elem[1]] > 1) {
            cog_charge += current_charge / 2.f;
            cog_position[1] += elem[0] * (current_charge / 2.f);
            cog_position[2] += elem[1] * (current_charge / 2.f);
          }
        }
      }
    }

    cog_position[1] /= cog_charge;
    cog_position[2] /= cog_charge;
    cog_position[1] += pad;
    cog_position[2] += time;

    digit_clusterizer_map[max_pos] = cog_position;
    digit_clusterizer_q[max_pos] = cog_charge;
    digit_map[max_pos] = {(int)cog_position[0], static_cast<int>(std::round(cog_position[1])), static_cast<int>(std::round(cog_position[2]))};
    digit_q[max_pos] = cog_charge;

    digit_map_tmp.clear();
    digit_q_tmp.clear();

    cog_charge = 0;
    std::fill(cog_position.begin(), cog_position.end(), 0);
    std::fill(investigate.begin(), investigate.end(), std::array<int, 5>{0});
    std::fill(found_min_saddle.begin(), found_min_saddle.end(), std::array<int, 3>{0});
  }

  std::iota(maxima_digits.begin(), maxima_digits.end(), 0);
}

// ---------------------------------
template <class T>
std::vector<std::vector<std::vector<int>>> qaIdeal::looper_tagger(int sector, T& index_map, std::vector<float>& array_q, std::vector<int>& index_array, std::string op_mode)
{
  int looper_detector_timesize = std::ceil((float)max_time[sector] / (float)looper_tagger_granularity);

  std::vector<std::vector<std::vector<float>>> tagger(looper_detector_timesize, std::vector<std::vector<float>>(o2::tpc::constants::MAXGLOBALPADROW)); // time_slice (=std::floor(time/looper_tagger_granularity)), row, pad array -> looper_tagged = 1, else 0
  std::vector<std::vector<std::vector<int>>> tagger_counter(looper_detector_timesize, std::vector<std::vector<int>>(o2::tpc::constants::MAXGLOBALPADROW));
  std::vector<std::vector<std::vector<int>>> looper_tagged_region(looper_detector_timesize, std::vector<std::vector<int>>(o2::tpc::constants::MAXGLOBALPADROW)); // accumulates all the regions that should be tagged: looper_tagged_region[time_slice][row] = (pad_low, pad_high)
  // std::vector<std::vector<std::vector<std::vector<float>>>> indv_charges(looper_detector_timesize, std::vector<std::vector<std::vector<float>>>(o2::tpc::constants::MAXGLOBALPADROW));

  for (int t = 0; t < looper_detector_timesize; t++) {
    for (int r = 0; r < o2::tpc::constants::MAXGLOBALPADROW; r++) {
      tagger[t][r].resize(TPC_GEOM[r][2] + looper_tagger_padwindow);
      tagger_counter[t][r].resize(TPC_GEOM[r][2] + looper_tagger_padwindow);
      looper_tagged_region[t][r].resize(TPC_GEOM[r][2] + 1);
      // indv_charges.resize(TPC_GEOM[r][2] + looper_tagger_padwindow);
    }
  }

  int operation_mode = 0;
  op_mode.find(std::string("digit")) != std::string::npos ? operation_mode = 1 : operation_mode = operation_mode;
  op_mode.find(std::string("ideal")) != std::string::npos ? operation_mode = 2 : operation_mode = operation_mode;

  // Improvements:
  // - Check the charge sigma -> Looper should have narrow sigma in charge
  // - Check width between clusters -> Looper should have regular distance -> peak in distribution of distance
  // - Check for gaussian distribution of charge: D'Agostino-Pearson

  int row = 0, pad = 0, time_slice = 0;
  for (int idx = 0; idx < index_array.size(); idx++) {
    row = std::round(index_map[index_array[idx]][0]);
    pad = std::round(index_map[index_array[idx]][1]);
    time_slice = std::floor(index_map[index_array[idx]][2] / (float)looper_tagger_granularity);

    // tagger[time_slice][row][pad]++;
    tagger_counter[time_slice][row][pad]++;
    tagger[time_slice][row][pad] += array_q[index_array[idx]] / landau_approx((array_q[index_array[idx]] - 25.f) / 17.f);
    // indv_charges[time_slice][row][pad].push_back(array_q[index_array[idx]]);

    // Approximate Landau and scale for the width:
    // Lindhards theory: L(x, mu=0, c=pi/2) ~ exp(-1/x)/(x*(x+1))
    // Estimation of peak-charge: Scale by landau((charge-25)/17)
  }

  if (verbose > 2)
    LOG(info) << "Tagger done. Building tagged regions.";

  // int unit_volume = looper_tagger_timewindow * 3;
  float avg_charge = 0;
  int num_elements = 0;
  for (int t = 0; t < (looper_detector_timesize - std::ceil(looper_tagger_timewindow / looper_tagger_granularity)); t++) {
    for (int r = 0; r < o2::tpc::constants::MAXGLOBALPADROW; r++) {
      for (int p = 0; p < TPC_GEOM[r][2] + 1; p++) {
        for (int t_acc = 0; t_acc < std::ceil(looper_tagger_timewindow / looper_tagger_granularity); t_acc++) {
          if (p == 0) {
            if (operation_mode == 1) {
              if (tagger_counter[t + t_acc][r][p] > 0)
                avg_charge += tagger[t + t_acc][r][p] / tagger_counter[t + t_acc][r][p];
              if (tagger_counter[t + t_acc][r][p + 1] > 0)
                avg_charge += tagger[t + t_acc][r][p + 1] / tagger_counter[t + t_acc][r][p + 1];
            }
            num_elements += tagger_counter[t + t_acc][r][p] + tagger_counter[t + t_acc][r][p + 1];
          } else if (p == TPC_GEOM[r][2]) {
            if (operation_mode == 1) {
              if (tagger_counter[t + t_acc][r][p] > 0)
                avg_charge += tagger[t + t_acc][r][p] / tagger_counter[t + t_acc][r][p];
              if (tagger_counter[t + t_acc][r][p - 1])
                avg_charge += tagger[t + t_acc][r][p - 1] / tagger_counter[t + t_acc][r][p - 1];
            }
            num_elements += tagger_counter[t + t_acc][r][p] + tagger_counter[t + t_acc][r][p - 1];
          } else {
            if (operation_mode == 1) {
              if (tagger_counter[t + t_acc][r][p + 1] > 0)
                avg_charge += tagger[t + t_acc][r][p + 1] / tagger_counter[t + t_acc][r][p + 1];
              if (tagger_counter[t + t_acc][r][p] > 0)
                avg_charge += tagger[t + t_acc][r][p] / tagger_counter[t + t_acc][r][p];
              if (tagger_counter[t + t_acc][r][p - 1] > 0)
                avg_charge += tagger[t + t_acc][r][p - 1] / tagger_counter[t + t_acc][r][p - 1];
            }
            num_elements += tagger_counter[t + t_acc][r][p - 1] + tagger_counter[t + t_acc][r][p] + tagger_counter[t + t_acc][r][p + 1];
          }
        }

        if (operation_mode == 1 && avg_charge >= looper_tagger_threshold_q && num_elements >= looper_tagger_threshold_num) {
          for (int t_tag = 0; t_tag < std::ceil(looper_tagger_timewindow / looper_tagger_granularity); t_tag++) {
            looper_tagged_region[t + t_tag][r][p] = 1;
          }
        } else if (operation_mode == 2 && num_elements >= looper_tagger_threshold_num) {
          for (int t_tag = 0; t_tag < std::ceil(looper_tagger_timewindow / looper_tagger_granularity); t_tag++) {
            looper_tagged_region[t + t_tag][r][p] = 1;
          }
        }
        avg_charge = 0;
        num_elements = 0;
      }
    }
  }

  if (verbose > 2)
    LOG(info) << "Looper tagging complete.";

  tagger_counter.clear();
  tagger.clear();

  return looper_tagged_region;
}

// ---------------------------------
template <class T>
void qaIdeal::remove_loopers(int sector, std::vector<std::vector<std::vector<int>>>& looper_map, T& map, std::vector<int>& index_array)
{
  std::vector<int> new_index_array;

  for (int m = 0; m < index_array.size(); m++) {
    if (looper_map[std::floor(map[index_array[m]][2] / (float)looper_tagger_granularity)][std::round(map[index_array[m]][0])][std::round(map[index_array[m]][1])] == 0) {
      new_index_array.push_back(index_array[m]);
    }
  }

  if (verbose > 2)
    LOG(info) << "Old size of maxima index array: " << index_array.size() << "; New size: " << new_index_array.size();

  index_array = new_index_array;
}

// ---------------------------------
template <class T>
void qaIdeal::run_network(int sector, T& map2d, std::vector<int>& maxima_digits, std::vector<std::array<int, 3>>& digit_map, std::vector<float>& digit_q, std::vector<std::array<float, 3>>& network_map, int eval_mode)
{

  // Loading the data
  std::vector<float> input_vector(maxima_digits.size() * (2 * global_shift[0] + 1) * (2 * global_shift[1] + 1) * (2 * global_shift[2] + 1), 0.f);
  std::vector<float> central_charges(maxima_digits.size(), 0.f);
  std::vector<int> new_max_dig = maxima_digits, index_pass(maxima_digits.size(), 0);
  std::iota(index_pass.begin(), index_pass.end(), 0);
  network_map.resize(maxima_digits.size());

  int index_shift_global = (2 * global_shift[0] + 1) * (2 * global_shift[1] + 1) * (2 * global_shift[2] + 1), index_shift_row = (2 * global_shift[0] + 1) * (2 * global_shift[1] + 1), index_shift_pad = (2 * global_shift[1] + 1), row_offset = 0, pad_offset = 0;

  int network_reg_size = maxima_digits.size(), counter_max_dig = 0, array_idx;
  std::vector<float> output_network_class(maxima_digits.size(), 0.f), output_network_reg;
  std::vector<float> temp_input(networkInputSize * (2 * global_shift[0] + 1) * (2 * global_shift[1] + 1) * (2 * global_shift[2] + 1), 0.f);

  for (unsigned int max = 0; max < maxima_digits.size(); max++) {
    row_offset = rowOffset(digit_map[maxima_digits[max]][0]);
    pad_offset = padOffset(digit_map[maxima_digits[max]][0]);
    central_charges[max] = digit_q[map2d[1][digit_map[maxima_digits[max]][2] + global_shift[1]][digit_map[maxima_digits[max]][0] + row_offset + global_shift[2]][digit_map[maxima_digits[max]][1] + global_shift[0] + pad_offset]];
    for (int row = 0; row < 2 * global_shift[2] + 1; row++) {
      for (int pad = 0; pad < 2 * global_shift[0] + 1; pad++) {
        for (int time = 0; time < 2 * global_shift[1] + 1; time++) {
          // (?) array_idx = map2d[1][digit_map[maxima_digits[max]][2] + 2 * global_shift[1] + 1 - time][digit_map[maxima_digits[max]][0]][digit_map[maxima_digits[max]][1] + pad + pad_offset];
          array_idx = map2d[1][digit_map[maxima_digits[max]][2] + time][digit_map[maxima_digits[max]][0] + row + row_offset][digit_map[maxima_digits[max]][1] + pad + pad_offset];
          if (array_idx > -1) {
            if (normalization_mode == 0) {
              input_vector[max * index_shift_global + row * index_shift_row + pad * index_shift_pad + time] = digit_q[array_idx] / 1024.f;
            } else if (normalization_mode == 1) {
              input_vector[max * index_shift_global + row * index_shift_row + pad * index_shift_pad + time] = digit_q[array_idx] / central_charges[max];
            }
          } else if (isBoundary(digit_map[maxima_digits[max]][0] + row + row_offset, digit_map[maxima_digits[max]][1] + pad + pad_offset)) {
            input_vector[max * index_shift_global + row * index_shift_row + pad * index_shift_pad + time] = -1;
          }
        }
      }
    }
  }

  if (eval_mode == 0 || eval_mode == 2) {

    network_reg_size = 0;
    for (int max = 0; max < maxima_digits.size(); max++) {
      for (int idx = 0; idx < index_shift_global; idx++) {
        temp_input[(max % networkInputSize) * index_shift_global + idx] = input_vector[max * index_shift_global + idx];
      }
      // if (verbose >= 5 && max == 10) {
      //   LOG(info) << "Size of the input vector: " << temp_input.size();
      //   LOG(info) << "Example input for neural network";
      //   for (int i = 0; i < 11; i++) {
      //     LOG(info) << "[ " << temp_input[11 * i + 0] << " " << temp_input[11 * i + 1] << " " << temp_input[11 * i + 2] << " " << temp_input[11 * i + 3] << " " << temp_input[11 * i + 4] << " " << temp_input[11 * i + 5] << " " << temp_input[11 * i + 6] << " " << temp_input[11 * i + 7] << " " << temp_input[11 * i + 8] << " " << temp_input[11 * i + 9] << " " << temp_input[11 * i + 10] << " ]";
      //   }
      //   LOG(info) << "Example output (classification): " << network_classification.inference(temp_input, networkInputSize)[0];
      //   LOG(info) << "Example output (regression): " << network_regression.inference(temp_input, networkInputSize)[0] << ", " << network_regression.inference(temp_input, networkInputSize)[1] << ", " << network_regression.inference(temp_input, networkInputSize)[2];
      // }
      if ((max + 1) % networkInputSize == 0 || max + 1 == maxima_digits.size()) {
        float* out_net = network_classification.inference(temp_input, networkInputSize);
        for (int idx = 0; idx < networkInputSize; idx++) {
          if (max + 1 == maxima_digits.size() && idx > (max % networkInputSize))
            break;
          else {
            output_network_class[int((max + 1) - networkInputSize) + idx] = out_net[idx];
            if (out_net[idx] > networkClassThres) {
              network_reg_size++;
            } else {
              new_max_dig[int((max + 1) - networkInputSize) + idx] = -1;
            }
          }
        }
      }
      if (max + 1 == maxima_digits.size()) {
        break;
      }
    }
    maxima_digits.clear();
    network_map.clear();
    index_pass.clear();
    maxima_digits.resize(network_reg_size);
    network_map.resize(network_reg_size);

    for (int max = 0; max < new_max_dig.size(); max++) {
      if (new_max_dig[max] > -1) {
        maxima_digits[counter_max_dig] = new_max_dig[max];
        index_pass.push_back(max);
        network_map[max][0] = digit_map[new_max_dig[max]][0];
        network_map[max][1] = digit_map[new_max_dig[max]][1];
        network_map[max][2] = digit_map[new_max_dig[max]][2];
        counter_max_dig++;
      }
    }

    LOG(info) << "Classification network done!";
  }

  if (eval_mode == 1 || eval_mode == 2) {

    output_network_reg.resize(3 * network_reg_size);
    int count_num_maxima = 0, count_reg = 0;
    std::vector<int> max_pos(networkInputSize);
    for (int max = 0; max < maxima_digits.size(); max++) {
      for (int idx = 0; idx < index_shift_row; idx++) {
        temp_input[(count_num_maxima % networkInputSize) * index_shift_global + idx] = input_vector[index_pass[max] * index_shift_global + idx];
      }
      max_pos[count_num_maxima] = max;
      count_num_maxima++;
      if ((count_num_maxima % networkInputSize == 0 && count_num_maxima > 0) || max + 1 == maxima_digits.size()) {
        float* out_net = network_regression.inference(temp_input, networkInputSize);
        for (int idx = 0; idx < networkInputSize; idx++) {
          if (max + 1 == maxima_digits.size() && idx >= (count_num_maxima % networkInputSize)) {
            break;
          } else {
            if ((out_net[2 * idx + 1] * 2.5 + digit_map[new_max_dig[index_pass[max_pos[idx]]]][2] < max_time[sector]) &&
                (out_net[2 * idx] * 2.5 + digit_map[new_max_dig[index_pass[max_pos[idx]]]][1] < max_pad[sector])) {
              output_network_reg[count_reg] = out_net[2 * idx + 1] * 2.5 + digit_map[new_max_dig[index_pass[max_pos[idx]]]][2]; // time
              output_network_reg[count_reg + 1] = out_net[2 * idx] * 2.5 + digit_map[new_max_dig[index_pass[max_pos[idx]]]][1]; // pad
              output_network_reg[count_reg + 2] = 100.f;                                                                        // out_net[3 * idx + 2]*10.f * central_charges[index_pass[max_pos[idx]]];       // charge
            } else {
              output_network_reg[count_reg] = 0;         // time
              output_network_reg[count_reg + 1] = 0;     // pad
              output_network_reg[count_reg + 2] = 100.f; // out_net[3 * idx + 2]*10.f * central_charges[index_pass[max_pos[idx]]];       // charge
            }
            count_reg += 3;
          }
        }
        count_num_maxima = 0;
      }
      if (max + 1 == maxima_digits.size()) {
        break;
      }
    }
    LOG(info) << "Regression network done!";

    std::vector<int> rows_max(maxima_digits.size());
    for (unsigned int max = 0; max < maxima_digits.size(); max++) {
      rows_max[max] = digit_map[maxima_digits[max]][0];
    }
    std::iota(maxima_digits.begin(), maxima_digits.end(), 0);
    digit_map.clear();
    digit_q.clear();
    digit_map.resize(maxima_digits.size());
    digit_q.resize(maxima_digits.size());

    for (int i = 0; i < maxima_digits.size(); i++) {
      digit_map[i] = std::array<int, 3>{rows_max[i], static_cast<int>(round(output_network_reg[3 * i + 1])), static_cast<int>(round(output_network_reg[3 * i]))};
      digit_q[i] = output_network_reg[3 * i + 2];
      network_map[i][0] = rows_max[i];
      network_map[i][1] = output_network_reg[3 * i + 1]; // pad
      network_map[i][2] = output_network_reg[3 * i];     // time
    }
    LOG(info) << "Network map written.";
  }

  input_vector.clear();
  temp_input.clear();
  central_charges.clear();
  new_max_dig.clear();
  index_pass.clear();
  output_network_class.clear();
  output_network_reg.clear();
}

// ---------------------------------
template <class T>
void qaIdeal::overwrite_map2d(int sector, T& map2d, std::vector<std::array<int, 3>>& element_map, std::vector<int>& element_idx, int mode)
{

  for (int row = 0; row < o2::tpc::constants::MAXGLOBALPADROW + 3 * global_shift[2]; row++) {
    for (int pad = 0; pad < 138 + 2 * global_shift[0]; pad++) {
      for (int time = 0; time < (max_time[sector] + 2 * global_shift[1]); time++) {
        map2d[mode][time][row][pad] = -1;
      }
    }
  }
  for (unsigned int id = 0; id < element_idx.size(); id++) {
    // LOG(info) << "Seg fault: time: " << element_map[element_idx[id]][2] << ", row: " << element_map[element_idx[id]][0] << ", pad: " << element_map[element_idx[id]][1];
    // LOG(info) << "Offset: " << padOffset(element_map[element_idx[id]][0]);
    map2d[mode][element_map[element_idx[id]][2] + global_shift[1]][element_map[element_idx[id]][0] + global_shift[2] + rowOffset(element_map[element_idx[id]][0])][element_map[element_idx[id]][1] + global_shift[0] + padOffset(element_map[element_idx[id]][0])] = id;
  }
}

// ---------------------------------
template <class T>
int qaIdeal::test_neighbour(std::array<int, 3> index, std::array<int, 2> nn, T& map2d, int mode)
{
  return map2d[mode][index[2] + global_shift[1] + nn[1]][index[0] + rowOffset(index[0]) + global_shift[2]][index[1] + padOffset(index[0]) + global_shift[0] + nn[0]];
}

// ---------------------------------
void qaIdeal::runQa(int loop_sectors)
{

  typedef std::array<std::vector<std::vector<std::vector<int>>>, 2> qa_t; // local 2D charge map, 0 - digits; 1 - ideal

  std::vector<int> maxima_digits; // , digit_isNoise, digit_isQED, digit_isValid;
  std::vector<std::array<int, 3>> digit_map, ideal_max_map;
  std::vector<std::array<float, 3>> ideal_cog_map, native_map, network_map, digit_clusterizer_map;
  std::vector<std::array<float, 2>> ideal_sigma_map;
  std::vector<float> ideal_max_q, ideal_cog_q, digit_q, digit_clusterizer_q;
  std::vector<std::vector<std::vector<int>>> tagger_map;

  if (mode.find(std::string("native")) != std::string::npos) {
    read_native(loop_sectors, digit_map, native_map, digit_q);
  } else if (mode.find(std::string("network")) != std::string::npos) {
    // read_network(loop_sectors, digit_map, digit_q);
    read_digits(loop_sectors, digit_map, digit_q);
  } else {
    read_digits(loop_sectors, digit_map, digit_q);
  }

  read_ideal(loop_sectors, ideal_max_map, ideal_max_q, ideal_cog_map, ideal_sigma_map, ideal_cog_q);

  LOG(info) << "Starting process for sector " << loop_sectors;

  qa_t map2d = init_map2d<qa_t>(loop_sectors);
  fill_map2d<qa_t>(loop_sectors, map2d, digit_map, ideal_max_map, ideal_max_q, ideal_cog_map, ideal_cog_q, -1);

  std::vector<int> dummy_counter(ideal_max_map.size());
  std::iota(dummy_counter.begin(), dummy_counter.end(), 0);
  if (mode.find(std::string("looper_tagger")) != std::string::npos) {
    tagger_map = looper_tagger(loop_sectors, ideal_max_map, ideal_max_q, dummy_counter, looper_tagger_opmode);
  }

  if ((mode.find(std::string("network")) == std::string::npos) && (mode.find(std::string("native")) == std::string::npos)) {
    find_maxima<qa_t>(loop_sectors, map2d, maxima_digits, digit_q);
    if (mode.find(std::string("looper_tagger")) != std::string::npos) {
      remove_loopers(loop_sectors, tagger_map, digit_map, maxima_digits);
    }
    if (mode.find(std::string("clusterizer")) != std::string::npos) {
      native_clusterizer(map2d, digit_map, maxima_digits, digit_q, digit_clusterizer_map, digit_clusterizer_q);
    }
    overwrite_map2d<qa_t>(loop_sectors, map2d, digit_map, maxima_digits, 1);
  } else {
    if (mode.find(std::string("native")) == std::string::npos) {
      find_maxima<qa_t>(loop_sectors, map2d, maxima_digits, digit_q);
      if (mode.find(std::string("looper_tagger")) != std::string::npos) {
        remove_loopers(loop_sectors, tagger_map, digit_map, maxima_digits);
      }
      if (mode.find(std::string("network_class")) != std::string::npos && mode.find(std::string("network_reg")) == std::string::npos) {
        run_network<qa_t>(loop_sectors, map2d, maxima_digits, digit_map, digit_q, network_map, 0); // classification
        if (mode.find(std::string("clusterizer")) != std::string::npos) {
          native_clusterizer(map2d, digit_map, maxima_digits, digit_q, digit_clusterizer_map, digit_clusterizer_q);
        }
      } else if (mode.find(std::string("network_reg")) != std::string::npos) {
        run_network<qa_t>(loop_sectors, map2d, maxima_digits, digit_map, digit_q, network_map, 1); // regression
      } else if (mode.find(std::string("network_full")) != std::string::npos) {
        run_network<qa_t>(loop_sectors, map2d, maxima_digits, digit_map, digit_q, network_map, 2); // classification + regression
      }
      overwrite_map2d<qa_t>(loop_sectors, map2d, digit_map, maxima_digits, 1);
    } else {
      maxima_digits.resize(digit_q.size());
      std::iota(std::begin(maxima_digits), std::end(maxima_digits), 0);
      if (mode.find(std::string("looper_tagger")) != std::string::npos) {
        remove_loopers(loop_sectors, tagger_map, digit_map, maxima_digits);
        remove_loopers(loop_sectors, tagger_map, native_map, maxima_digits);
      }
    }
  }

  // Assignment at d=1
  LOG(info) << "Maxima found in digits (before): " << maxima_digits.size() << "; Maxima found in ideal clusters (before): " << ideal_max_map.size();

  std::vector<int> assigned_ideal(ideal_max_map.size(), 0), clone_order(maxima_digits.size(), 0);
  std::vector<std::array<int, 25>> assignments_dig_to_id(ideal_max_map.size());
  std::vector<int> assigned_digit(maxima_digits.size(), 0);
  std::vector<std::array<int, 25>> assignments_id_to_dig(maxima_digits.size());
  int current_neighbour;
  std::vector<float> fractional_clones_vector(maxima_digits.size(), 0);

  for (int i = 0; i < ideal_max_map.size(); i++) {
    for (int j = 0; j < 25; j++) {
      assignments_dig_to_id[i][j] = -1;
    }
  }
  for (int i = 0; i < maxima_digits.size(); i++) {
    for (int j = 0; j < 25; j++) {
      assignments_id_to_dig[i][j] = -1;
    }
  }

  std::fill(assigned_digit.begin(), assigned_digit.end(), 0);
  std::fill(assigned_ideal.begin(), assigned_ideal.end(), 0);
  std::fill(clone_order.begin(), clone_order.end(), 0);

  number_of_digit_max[loop_sectors] += maxima_digits.size();
  number_of_ideal_max[loop_sectors] += ideal_max_map.size();

  for (int max = 0; max < ideal_max_map.size(); max++) {
    if (ideal_cog_q[max] >= 5 && ideal_max_q[max] >= 3) {
      number_of_ideal_max_findable[loop_sectors]++;
    }
  }

  // Level-1 loop: Goes through the layers specified by the adjacency matrix <-> Loop of possible distances
  int layer_count = 0;
  for (int layer = 0; layer < adj_mat.size(); layer++) {

    // Level-2 loop: Goes through the elements of the adjacency matrix at distance d, n times to assign neighbours iteratively
    for (int nn = 0; nn < adj_mat[layer].size(); nn++) {

      // Level-3 loop: Goes through all digit maxima and checks neighbourhood for potential ideal maxima
      for (unsigned int locdigit = 0; locdigit < maxima_digits.size(); locdigit++) {
        current_neighbour = test_neighbour(digit_map[maxima_digits[locdigit]], adj_mat[layer][nn], map2d, 0);
        // if (verbose >= 5) LOG(info) << "current_neighbour: " << current_neighbour;
        // if (verbose >= 5) LOG(info) << "Maximum digit " << maxima_digits[locdigit];
        // if (verbose >= 5) LOG(info) << "Digit max index: " << digit_map[maxima_digits[locdigit]][0] << " " << digit_map[maxima_digits[locdigit]][1] << " " << digit_map[maxima_digits[locdigit]][2];
        if (current_neighbour >= -1) {
          assignments_id_to_dig[locdigit][layer_count + nn] = ((current_neighbour != -1 && assigned_digit[locdigit] == 0) ? (assigned_ideal[current_neighbour] == 0 ? current_neighbour : -1) : -1);
        }
      }
      if (verbose >= 4)
        LOG(info) << "Done with assignment for digit maxima, layer " << layer;

      // Level-3 loop: Goes through all ideal maxima and checks neighbourhood for potential digit maxima
      std::array<int, 3> rounded_cog;
      for (unsigned int locideal = 0; locideal < ideal_max_map.size(); locideal++) {
        for (int i = 0; i < 3; i++) {
          rounded_cog[i] = round(ideal_cog_map[locideal][i]);
        }
        current_neighbour = test_neighbour(rounded_cog, adj_mat[layer][nn], map2d, 1);
        // if (verbose >= 5) LOG(info) << "current_neighbour: " << current_neighbour;
        // if (verbose >= 5) LOG(info) << "Maximum ideal " << locideal;
        // if (verbose >= 5) LOG(info) << "Ideal max index: " << ideal_max_map[locideal][0] << " " << ideal_max_map[locideal][1] << " " << ideal_max_map[locideal][2];
        if (current_neighbour >= -1) {
          assignments_dig_to_id[locideal][layer_count + nn] = ((current_neighbour != -1 && assigned_ideal[locideal] == 0) ? (assigned_digit[current_neighbour] == 0 ? current_neighbour : -1) : -1);
        }
      }
      if (verbose >= 4)
        LOG(info) << "Done with assignment for ideal maxima, layer " << layer;
    }

    // Level-2 loop: Checks all digit maxima and how many ideal maxima neighbours have been found in the current layer
    if ((mode.find(std::string("training_data")) != std::string::npos && layer >= 2) || mode.find(std::string("training_data")) == std::string::npos) {
      for (unsigned int locdigit = 0; locdigit < maxima_digits.size(); locdigit++) {
        assigned_digit[locdigit] = 0;
        for (int counter_max = 0; counter_max < 25; counter_max++) {
          if (checkIdx(assignments_id_to_dig[locdigit][counter_max])) {
            assigned_digit[locdigit] += 1;
          }
        }
      }
    }

    // Level-2 loop: Checks all ideal maxima and how many digit maxima neighbours have been found in the current layer
    for (unsigned int locideal = 0; locideal < ideal_max_map.size(); locideal++) {
      assigned_ideal[locideal] = 0;
      for (int counter_max = 0; counter_max < 25; counter_max++) {
        if (checkIdx(assignments_dig_to_id[locideal][counter_max])) {
          assigned_ideal[locideal] += 1;
        }
      }
    }

    // Assign all possible digit maxima once you are above a certain distance away from the current maximum (here: At layer with distance greater than sqrt(2))

    if (verbose >= 3)
      LOG(info) << "Removed maxima for layer " << layer;

    layer_count += adj_mat[layer].size();

    if (verbose >= 3)
      LOG(info) << "Layer count is now " << layer_count;
  }

  // Checks the number of assignments that have been made with the above loops
  int count_elements_findable = 0, count_elements_dig = 0, count_elements_id = 0;
  for (unsigned int ass_id = 0; ass_id < ideal_max_map.size(); ass_id++) {
    count_elements_id = 0;
    count_elements_findable = 0;
    for (auto elem : assignments_dig_to_id[ass_id]) {
      if (checkIdx(elem)) {
        count_elements_id += 1;
        if (ideal_cog_q[ass_id] >= 5 && ideal_max_q[ass_id] >= 3) {
          count_elements_findable += 1;
        }
      }
    }
    // if (verbose >= 5 && (ass_id%10000)==0) LOG(info) << "Count elements: " << count_elements_id << " ass_id: " << ass_id << " assignments_ideal: " << assignments_ideal[count_elements_id];
    assignments_ideal[loop_sectors][count_elements_id] += 1;
    assignments_ideal_findable[loop_sectors][count_elements_findable] += 1;
  }
  for (unsigned int ass_dig = 0; ass_dig < maxima_digits.size(); ass_dig++) {
    count_elements_dig = 0;
    count_elements_findable = 0;
    for (auto elem : assignments_id_to_dig[ass_dig]) {
      if (checkIdx(elem)) {
        count_elements_dig += 1;
        if (ideal_cog_q[elem] >= 5 && ideal_max_q[elem] >= 3) {
          count_elements_findable += 1;
        }
      }
    }
    // if (verbose >= 5 && (ass_dig%10000)==0) LOG(info) << "Count elements: " << count_elements_dig << " ass_dig: " << ass_dig << " assignments_digit: " << assignments_digit[count_elements_dig];
    assignments_digit[loop_sectors][count_elements_dig] += 1;
    assignments_digit_findable[loop_sectors][count_elements_findable] += 1;
  }

  if (verbose >= 3)
    LOG(info) << "Done checking the number of assignments";

  // Clone-rate
  for (unsigned int locideal = 0; locideal < assignments_dig_to_id.size(); locideal++) {
    for (auto elem_id : assignments_dig_to_id[locideal]) {
      if (checkIdx(elem_id)) {
        int count_elements_clone = 0;
        for (auto elem_dig : assignments_id_to_dig[elem_id]) {
          if (checkIdx(elem_dig))
            count_elements_clone += 1;
        }
        if (count_elements_clone == 1)
          clone_order[elem_id] += 1;
      }
    }
  }
  for (unsigned int locdigit = 0; locdigit < maxima_digits.size(); locdigit++) {
    if (clone_order[locdigit] > 1) {
      clones[loop_sectors] += 1;
    }
  }

  // Fractional clone rate
  for (unsigned int locideal = 0; locideal < assignments_dig_to_id.size(); locideal++) {
    int count_links = 0;
    for (auto elem_id : assignments_dig_to_id[locideal]) {
      if (checkIdx(elem_id)) {
        count_links += 1;
      }
    }
    if (count_links > 1) {
      for (auto elem_id : assignments_dig_to_id[locideal]) {
        if (checkIdx(elem_id)) {
          fractional_clones_vector[elem_id] += 1.f / (float)count_links;
        }
      }
    }
  }
  for (auto elem_frac : fractional_clones_vector) {
    fractional_clones[loop_sectors] += elem_frac;
  }

  if (verbose >= 3)
    LOG(info) << "Done determining the clone rate";

  if (verbose >= 4) {
    for (int ass = 0; ass < 25; ass++) {
      LOG(info) << "Number of assignments to one digit maximum (#assignments " << ass << "): " << assignments_digit[loop_sectors][ass];
      LOG(info) << "Number of assignments to one ideal maximum (#assignments " << ass << "): " << assignments_ideal[loop_sectors][ass] << "\n";
    }
  }

  if (mode.find(std::string("native")) != std::string::npos && create_output == 1) {

    if (verbose >= 3)
      LOG(info) << "Native-Ideal assignment...";

    // creating training data for the neural network
    int data_size = maxima_digits.size();

    std::vector<std::array<float, 6>> native_ideal_assignemnt;
    std::array<float, 6> current_element;

    std::fill(assigned_ideal.begin(), assigned_ideal.end(), 0);
    std::fill(assigned_digit.begin(), assigned_digit.end(), 0);

    // Some useful variables
    int map_dig_idx = 0, map_q_idx = 0, check_assignment = 0, index_assignment = -1, current_idx_id = -1, current_idx_dig = -1, row_offset = 0, pad_offset = 0;
    float distance_assignment = 100000.f, current_distance_dig_to_id = 0, current_distance_id_to_dig = 0;
    bool is_min_dist = true;

    for (int max_point = 0; max_point < data_size; max_point++) {
      row_offset = rowOffset(digit_map[maxima_digits[max_point]][0]);
      pad_offset = padOffset(digit_map[maxima_digits[max_point]][0]);
      map_dig_idx = map2d[1][digit_map[maxima_digits[max_point]][2] + global_shift[1]][digit_map[maxima_digits[max_point]][0] + row_offset + global_shift[2]][digit_map[maxima_digits[max_point]][1] + global_shift[0] + pad_offset];
      if (checkIdx(map_dig_idx)) {
        check_assignment = 0;
        index_assignment = -1;
        distance_assignment = 100000.f;
        is_min_dist = true;
        for (int i = 0; i < 25; i++) {
          // Checks all ideal maxima assigned to one digit maximum by calculating mutual distance
          current_idx_id = assignments_id_to_dig[max_point][i];
          if (checkIdx(current_idx_id)) {
            if ((ideal_cog_q[current_idx_id] < 5 && ideal_max_q[current_idx_id] < 3) || (assigned_ideal[current_idx_id] != 0)) {
              is_min_dist = false;
              break;
            } else {
              current_distance_dig_to_id = std::pow((native_map[maxima_digits[max_point]][2] - ideal_cog_map[current_idx_id][2]), 2) + std::pow((native_map[maxima_digits[max_point]][1] - ideal_cog_map[current_idx_id][1]), 2);
              // if the distance is less than the previous one check if update should be made
              if (current_distance_dig_to_id < distance_assignment) {
                for (int j = 0; j < 25; j++) {
                  current_idx_dig = assignments_dig_to_id[current_idx_id][j];
                  if (checkIdx(current_idx_dig)) {
                    if (assigned_digit[current_idx_dig] == 0) {
                      // calculate mutual distance from current ideal CoG to all assigned digit maxima. Update if and only if distance is minimal. Else do not assign.
                      current_distance_id_to_dig = std::pow((native_map[maxima_digits[current_idx_dig]][2] - ideal_cog_map[current_idx_id][2]), 2) + std::pow((native_map[maxima_digits[current_idx_dig]][1] - ideal_cog_map[current_idx_id][1]), 2);
                      if (current_distance_id_to_dig < current_distance_dig_to_id) {
                        is_min_dist = false;
                        break;
                      }
                      ////
                      // Potential improvement: Weight the distance calculation by the qTot/qMax such that they are not too far apart
                      ////
                    }
                  }
                }
                if (is_min_dist) {
                  check_assignment++;
                  distance_assignment = current_distance_dig_to_id;
                  index_assignment = current_idx_id;
                  // Adding an assignment in order to avoid duplication
                  assigned_digit[max_point]++;
                  assigned_ideal[current_idx_id]++;
                  current_element[0] = ideal_cog_map[maxima_digits[max_point]][0];
                  current_element[1] = ideal_cog_map[current_idx_id][1];
                  current_element[2] = ideal_cog_map[current_idx_id][2];
                  current_element[3] = native_map[maxima_digits[max_point]][0];
                  current_element[4] = native_map[maxima_digits[max_point]][1];
                  current_element[5] = native_map[maxima_digits[max_point]][2];
                }
                // At least check if assigned, and put classification label to 1, no regression
                // else {
                //   check_assignment++;
                //   distance_assignment = current_distance_dig_to_id;
                //   index_assignment = current_idx_id;
                //   // Adding an assignment in order to avoid duplication
                //   assigned_digit[max_point]++;
                //   assigned_ideal[current_idx_id]++;
                //   current_element[0] = native_map[maxima_digits[max_point]][2];
                //   current_element[1] = native_map[maxima_digits[max_point]][1];
                //   current_element[2] = ideal_cog_map[current_idx_id][2];
                //   current_element[3] = ideal_cog_map[current_idx_id][1];
                // }
              }
            }
          }
        }

        if (check_assignment > 0 && is_min_dist) {
          native_ideal_assignemnt.push_back(current_element);
        }
      }
    }
    if (verbose >= 3)
      LOG(info) << "Done performing native-ideal assignment. Writing to file...";

    std::stringstream file_in;
    file_in << "native_ideal_" << loop_sectors << ".root";
    TFile* outputFileNativeIdeal = new TFile(file_in.str().c_str(), "RECREATE");
    TTree* native_ideal = new TTree("native_ideal", "tree");

    float nat_row = 0, nat_time = 0, nat_pad = 0, id_row = 0, id_time = 0, id_pad = 0, native_minus_ideal_time = 0, native_minus_ideal_pad = 0;
    native_ideal->Branch("sector", &loop_sectors);
    native_ideal->Branch("native_row", &nat_row);
    native_ideal->Branch("native_cog_time", &nat_time);
    native_ideal->Branch("native_cog_pad", &nat_pad);
    native_ideal->Branch("ideal_row", &id_row);
    native_ideal->Branch("ideal_cog_time", &id_time);
    native_ideal->Branch("ideal_cog_pad", &id_pad);
    native_ideal->Branch("native_minus_ideal_time", &native_minus_ideal_time);
    native_ideal->Branch("native_minus_ideal_pad", &native_minus_ideal_pad);

    for (int elem = 0; elem < native_ideal_assignemnt.size(); elem++) {
      id_row = native_ideal_assignemnt[elem][0];
      id_pad = native_ideal_assignemnt[elem][1];
      id_time = native_ideal_assignemnt[elem][2];
      nat_row = native_ideal_assignemnt[elem][3];
      nat_pad = native_ideal_assignemnt[elem][4];
      nat_time = native_ideal_assignemnt[elem][5];
      native_minus_ideal_time = nat_time - id_time;
      native_minus_ideal_pad = nat_pad - id_pad;
      native_ideal->Fill();
    }

    native_ideal->Write();
    outputFileNativeIdeal->Close();

    native_ideal_assignemnt.clear();
  }

  if (mode.find(std::string("network")) != std::string::npos && create_output == 1) {

    if (verbose >= 3)
      LOG(info) << "Network-Ideal assignment...";

    // creating training data for the neural network
    int data_size = maxima_digits.size();

    std::vector<std::array<float, 5>> network_ideal_assignemnt;
    std::array<float, 5> current_element;

    std::fill(assigned_ideal.begin(), assigned_ideal.end(), 0);
    std::fill(assigned_digit.begin(), assigned_digit.end(), 0);

    // Some useful variables
    int map_dig_idx = 0, map_q_idx = 0, check_assignment = 0, index_assignment = -1, current_idx_id = -1, current_idx_dig = -1;
    float distance_assignment = 100000.f, current_distance_dig_to_id = 0, current_distance_id_to_dig = 0;
    bool is_min_dist = true;

    for (int max_point = 0; max_point < data_size; max_point++) {
      map_dig_idx = map2d[1][digit_map[maxima_digits[max_point]][2] + global_shift[1]][digit_map[maxima_digits[max_point]][0] + rowOffset(digit_map[maxima_digits[max_point]][0]) + global_shift[2]][digit_map[maxima_digits[max_point]][1] + global_shift[0] + padOffset(digit_map[maxima_digits[max_point]][0])];
      if (checkIdx(map_dig_idx)) {
        check_assignment = 0;
        index_assignment = -1;
        distance_assignment = 100000.f;
        is_min_dist = true;
        for (int i = 0; i < 25; i++) {
          // Checks all ideal maxima assigned to one digit maximum by calculating mutual distance
          current_idx_id = assignments_id_to_dig[max_point][i];
          if (checkIdx(current_idx_id)) {
            if ((ideal_cog_q[current_idx_id] < 5 && ideal_max_q[current_idx_id] < 3) || (assigned_ideal[current_idx_id] != 0)) {
              is_min_dist = false;
              break;
            } else {
              current_distance_dig_to_id = std::pow((network_map[maxima_digits[max_point]][2] - ideal_cog_map[current_idx_id][2]), 2) + std::pow((network_map[maxima_digits[max_point]][1] - ideal_cog_map[current_idx_id][1]), 2);
              // if the distance is less than the previous one check if update should be made
              if (current_distance_dig_to_id < distance_assignment) {
                for (int j = 0; j < 25; j++) {
                  current_idx_dig = assignments_dig_to_id[current_idx_id][j];
                  if (checkIdx(current_idx_dig)) {
                    if (assigned_digit[current_idx_dig] == 0) {
                      // calculate mutual distance from current ideal CoG to all assigned digit maxima. Update if and only if distance is minimal. Else do not assign.
                      current_distance_id_to_dig = std::pow((network_map[maxima_digits[current_idx_dig]][2] - ideal_cog_map[current_idx_id][2]), 2) + std::pow((network_map[maxima_digits[current_idx_dig]][1] - ideal_cog_map[current_idx_id][1]), 2);
                      if (current_distance_id_to_dig < current_distance_dig_to_id) {
                        is_min_dist = false;
                        break;
                      }
                      ////
                      // Potential improvement: Weight the distance calculation by the qTot/qMax such that they are not too far apart
                      ////
                    }
                  }
                }
                if (is_min_dist) {
                  check_assignment++;
                  distance_assignment = current_distance_dig_to_id;
                  index_assignment = current_idx_id;
                  // Adding an assignment in order to avoid duplication
                  assigned_digit[max_point]++;
                  assigned_ideal[current_idx_id]++;
                  current_element[0] = network_map[maxima_digits[max_point]][2];
                  current_element[1] = network_map[maxima_digits[max_point]][1];
                  current_element[2] = ideal_cog_map[current_idx_id][2];
                  current_element[3] = ideal_cog_map[current_idx_id][1];
                  if (normalization_mode == 0) {
                    current_element[4] = ideal_cog_q[current_idx_id] / 1024.f;
                  } else if (normalization_mode == 1) {
                    current_element[4] = ideal_cog_q[current_idx_id] / digit_q[maxima_digits[max_point]];
                  }
                }
              }
            }
          }
        }

        if (check_assignment > 0 && is_min_dist) {
          network_ideal_assignemnt.push_back(current_element);
        }
      }
    }
    if (verbose >= 3)
      LOG(info) << "Done performing network-ideal assignment. Writing to file...";

    std::stringstream file_in;
    file_in << "network_ideal_" << loop_sectors << ".root";
    TFile* outputFileNetworkIdeal = new TFile(file_in.str().c_str(), "RECREATE");
    TTree* network_ideal = new TTree("network_ideal", "tree");

    float net_time = 0, net_pad = 0, id_time = 0, id_pad = 0, net_minus_ideal_time = 0, net_minus_ideal_pad = 0, charge_ratio;
    network_ideal->Branch("network_cog_time", &net_time);
    network_ideal->Branch("network_cog_pad", &net_pad);
    network_ideal->Branch("ideal_cog_time", &id_time);
    network_ideal->Branch("ideal_cog_pad", &id_pad);
    network_ideal->Branch("net_minus_ideal_time", &net_minus_ideal_time);
    network_ideal->Branch("net_minus_ideal_pad", &net_minus_ideal_pad);
    network_ideal->Branch("charge_ideal_over_network", &charge_ratio);

    LOG(info) << "Network map size: " << network_map.size();
    LOG(info) << "Network-ideal size: " << network_ideal_assignemnt.size();

    for (int elem = 0; elem < network_ideal_assignemnt.size(); elem++) {
      net_time = network_ideal_assignemnt[elem][0];
      net_pad = network_ideal_assignemnt[elem][1];
      id_time = network_ideal_assignemnt[elem][2];
      id_pad = network_ideal_assignemnt[elem][3];
      net_minus_ideal_time = net_time - id_time;
      net_minus_ideal_pad = net_pad - id_pad;
      charge_ratio = network_ideal_assignemnt[elem][4];
      network_ideal->Fill();
    }

    network_ideal->Write();
    outputFileNetworkIdeal->Close();

    network_ideal_assignemnt.clear();
  }

  if (mode.find(std::string("training_data")) != std::string::npos && create_output == 1) {

    std::vector<int> index_digits(digit_map.size());
    std::iota(index_digits.begin(), index_digits.end(), 0);
    overwrite_map2d(loop_sectors, map2d, digit_map, index_digits, 0);

    if (verbose >= 3)
      LOG(info) << "Creating training data...";

    // creating training data for the neural network
    int mat_size_time = (global_shift[1] * 2 + 1), mat_size_pad = (global_shift[0] * 2 + 1), mat_size_row = (global_shift[2] * 2 + 1), data_size = maxima_digits.size();
    std::vector<std::vector<std::vector<std::vector<float>>>> tr_data_X(data_size);
    std::vector<std::vector<std::vector<float>>> atomic_unit;

    atomic_unit.resize(mat_size_row);
    for (int row = 0; row < mat_size_row; row++) {
      atomic_unit[row].resize(mat_size_pad);
      for (int pad = 0; pad < mat_size_pad; pad++) {
        atomic_unit[row][pad].resize(mat_size_time);
        for (int time = 0; time < mat_size_time; time++) {
          atomic_unit[row][pad][time] = 0;
        }
      }
    }

    std::fill(tr_data_X.begin(), tr_data_X.end(), atomic_unit);

    std::vector<int> tr_data_Y_class(data_size, -1);
    std::array<std::vector<float>, 5> tr_data_Y_reg;
    std::fill(tr_data_Y_reg.begin(), tr_data_Y_reg.end(), std::vector<float>(data_size, -1));

    std::fill(assigned_ideal.begin(), assigned_ideal.end(), 0);
    std::fill(assigned_digit.begin(), assigned_digit.end(), 0);

    if (verbose >= 3)
      LOG(info) << "Initialized arrays...";

    // Some useful variables
    int map_dig_idx = 0, map_q_idx = 0, check_assignment = 0, index_assignment = -1, current_idx_id = -1, current_idx_dig = -1, row_offset = 0, pad_offset = 0, class_label = 0;
    float distance_assignment = 100000.f, current_distance_dig_to_id = 0, current_distance_id_to_dig = 0;
    bool is_min_dist = true;

    for (int max_point = 0; max_point < data_size; max_point++) {
      row_offset = rowOffset(digit_map[maxima_digits[max_point]][0]);
      pad_offset = padOffset(digit_map[maxima_digits[max_point]][0]);
      map_dig_idx = map2d[1][digit_map[maxima_digits[max_point]][2] + global_shift[1]][digit_map[maxima_digits[max_point]][0] + row_offset + global_shift[2]][digit_map[maxima_digits[max_point]][1] + pad_offset + global_shift[0]];
      if (checkIdx(map_dig_idx)) {
        float q_max = digit_q[maxima_digits[map_dig_idx]];
        for (int row = 0; row < mat_size_row; row++) {
          for (int pad = 0; pad < mat_size_pad; pad++) {
            for (int time = 0; time < mat_size_time; time++) {
              map_q_idx = map2d[0][digit_map[maxima_digits[max_point]][2] + time][digit_map[maxima_digits[max_point]][0] + row + row_offset][digit_map[maxima_digits[max_point]][1] + pad + pad_offset];
              if (map_q_idx == -1) {
                if (isBoundary(digit_map[maxima_digits[max_point]][0] + row + row_offset - global_shift[2], digit_map[maxima_digits[max_point]][1] + pad + pad_offset - global_shift[0])) {
                  tr_data_X[max_point][row][pad][time] = -1;
                } else {
                  tr_data_X[max_point][row][pad][time] = 0;
                }
              } else {
                if (normalization_mode == 0) {
                  tr_data_X[max_point][row][pad][time] = digit_q[map_q_idx] / 1024.f;
                } else if (normalization_mode == 1) {
                  tr_data_X[max_point][row][pad][time] = digit_q[map_q_idx] / q_max;
                }
              }
            }
          }
        }
        check_assignment = 0;
        index_assignment = -1;
        distance_assignment = 100000.f;
        is_min_dist = true;
        class_label = 0;
        for (int i = 0; i < 25; i++) {
          // Checks all ideal maxima assigned to one digit maximum by calculating mutual distance
          current_idx_id = assignments_id_to_dig[max_point][i];
          if (checkIdx(current_idx_id)) {
            if ((ideal_cog_q[current_idx_id] < 5 && ideal_max_q[current_idx_id] < 3) || (assigned_ideal[current_idx_id] != 0)) {
              is_min_dist = false;
              break;
            } else {
              class_label++;
              if (use_max_cog == 0) {
                current_distance_dig_to_id = std::pow((digit_map[maxima_digits[max_point]][2] - ideal_max_map[current_idx_id][2]), 2) + std::pow((digit_map[maxima_digits[max_point]][1] - ideal_max_map[current_idx_id][1]), 2);
              } else if (use_max_cog == 1) {
                current_distance_dig_to_id = std::pow((digit_map[maxima_digits[max_point]][2] - ideal_cog_map[current_idx_id][2]), 2) + std::pow((digit_map[maxima_digits[max_point]][1] - ideal_cog_map[current_idx_id][1]), 2);
              }
              // if the distance is less than the previous one check if update should be made
              if (current_distance_dig_to_id < distance_assignment) {
                for (int j = 0; j < 25; j++) {
                  current_idx_dig = assignments_dig_to_id[current_idx_id][j];
                  if (checkIdx(current_idx_dig)) {
                    if (assigned_digit[current_idx_dig] == 0) {
                      // calculate mutual distance from current ideal CoG to all assigned digit maxima. Update if and only if distance is minimal. Else do not assign.
                      if (use_max_cog == 0) {
                        current_distance_id_to_dig = std::pow((digit_map[maxima_digits[max_point]][2] - ideal_max_map[current_idx_id][2]), 2) + std::pow((digit_map[maxima_digits[max_point]][1] - ideal_max_map[current_idx_id][1]), 2);
                      } else if (use_max_cog == 1) {
                        current_distance_id_to_dig = std::pow((digit_map[maxima_digits[max_point]][2] - ideal_cog_map[current_idx_id][2]), 2) + std::pow((digit_map[maxima_digits[max_point]][1] - ideal_cog_map[current_idx_id][1]), 2);
                      }
                      if (current_distance_id_to_dig < current_distance_dig_to_id) {
                        is_min_dist = false;
                        break;
                      }
                      ////
                      // Potential improvement: Weight the distance calculation by the qTot/qMax such that they are not too far apart
                      ////
                    }
                  }
                }
                if (is_min_dist) { // && (digit_q[maxima_digits[max_point]] < ideal_cog_q[current_idx_id])) {
                  check_assignment++;
                  distance_assignment = current_distance_dig_to_id;
                  index_assignment = current_idx_id;
                  // Adding an assignment in order to avoid duplication
                  assigned_digit[max_point]++;
                  assigned_ideal[current_idx_id]++;
                }
              }
            }
          }
        }

        // tr_data_Y_class[max_point] = check_assignment;
        tr_data_Y_class[max_point] = class_label;
        if (check_assignment > 0 && is_min_dist) {
          tr_data_Y_reg[0][max_point] = ideal_cog_map[index_assignment][2] - digit_map[maxima_digits[max_point]][2]; // time
          tr_data_Y_reg[1][max_point] = ideal_cog_map[index_assignment][1] - digit_map[maxima_digits[max_point]][1]; // pad
          tr_data_Y_reg[2][max_point] = ideal_sigma_map[index_assignment][0];                                        // sigma pad
          tr_data_Y_reg[3][max_point] = ideal_sigma_map[index_assignment][1];                                        // sigma time
          if (normalization_mode == 0) {
            tr_data_Y_reg[4][max_point] = ideal_cog_q[index_assignment] / 1024.f;
          } else if (normalization_mode == 1) {
            tr_data_Y_reg[4][max_point] = ideal_cog_q[index_assignment] / q_max;
          }
          // if(std::abs(digit_map[maxima_digits[max_point]][2] - ideal_cog_map[index_assignment][2]) > 3 || std::abs(digit_map[maxima_digits[max_point]][1] - ideal_cog_map[index_assignment][1]) > 3){
          //   LOG(info) << "#Maxima: " << maxima_digits.size() << ", Index (point) " << max_point << " & (max) " << maxima_digits[max_point] << " & (ideal) " << index_assignment << ", ideal_cog_map.size(): " <<  ideal_cog_map.size() << ", index_assignment: " << index_assignment;
          // }
        } else {
          tr_data_Y_reg[0][max_point] = -1.f;
          tr_data_Y_reg[1][max_point] = -1.f;
          tr_data_Y_reg[2][max_point] = -1.f;
          tr_data_Y_reg[3][max_point] = -1.f;
          tr_data_Y_reg[4][max_point] = -1.f;
        }
      }
    }
    if (verbose >= 3)
      LOG(info) << "Done creating training data. Writing to file...";

    std::stringstream file_in;
    file_in << "training_data_" << loop_sectors << ".root";
    TFile* outputFileTrData = new TFile(file_in.str().c_str(), "RECREATE");
    TTree* tr_data = new TTree("tr_data", "tree");

    // Defining the branches
    for (int row = 0; row < mat_size_row; row++) {
      for (int pad = 0; pad < mat_size_pad; pad++) {
        for (int time = 0; time < mat_size_time; time++) {
          std::stringstream branch_name;
          branch_name << "in_row_" << row << "_pad_" << pad << "_time_" << time;
          tr_data->Branch(branch_name.str().c_str(), &atomic_unit[row][pad][time]); // Switching pad and time here makes the transformatio in python easier
        }
      }
    }

    int class_val = 0, idx_sector = 0, idx_row = 0, idx_pad = 0, idx_time = 0;
    float trY_time = 0, trY_pad = 0, trY_sigma_pad = 0, trY_sigma_time = 0, trY_q = 0;
    tr_data->Branch("out_class", &class_val);
    tr_data->Branch("out_idx_sector", &idx_sector);
    tr_data->Branch("out_idx_row", &idx_row);
    tr_data->Branch("out_idx_pad", &idx_pad);
    tr_data->Branch("out_idx_time", &idx_time);
    tr_data->Branch("out_reg_pad", &trY_pad);
    tr_data->Branch("out_reg_time", &trY_time);
    tr_data->Branch("out_sigma_pad", &trY_sigma_pad);
    tr_data->Branch("out_sigma_time", &trY_sigma_time);
    tr_data->Branch("out_reg_qTotOverqMax", &trY_q);

    // Filling elements
    for (int element = 0; element < data_size; element++) {
      atomic_unit = tr_data_X[element];
      class_val = tr_data_Y_class[element];
      trY_time = tr_data_Y_reg[0][element];
      trY_pad = tr_data_Y_reg[1][element];
      trY_sigma_time = tr_data_Y_reg[2][element];
      trY_sigma_pad = tr_data_Y_reg[3][element];
      trY_q = tr_data_Y_reg[4][element];
      idx_sector = loop_sectors;
      idx_row = digit_map[maxima_digits[element]][0];
      idx_pad = digit_map[maxima_digits[element]][1];
      idx_time = digit_map[maxima_digits[element]][2];
      tr_data->Fill();
    }
    tr_data->Write();
    outputFileTrData->Close();

    tr_data_X.clear();
    for (int i = 0; i < 5; i++) {
      tr_data_Y_reg[i].clear();
    }
    tr_data_Y_class.clear();
  }

  map2d[0].clear();
  map2d[1].clear();

  maxima_digits.clear();
  digit_map.clear();
  ideal_max_map.clear();
  ideal_cog_map.clear();
  native_map.clear();
  network_map.clear();
  digit_clusterizer_map.clear();
  ideal_sigma_map.clear();
  ideal_max_q.clear();
  ideal_cog_q.clear();
  digit_q.clear();
  digit_clusterizer_q.clear();

  assigned_ideal.clear();
  clone_order.clear();
  assignments_dig_to_id.clear();
  assigned_digit.clear();
  assignments_id_to_dig.clear();
  fractional_clones_vector.clear();

  if (verbose >= 2)
    LOG(info) << "Done with sector " << loop_sectors << "!";
}

// ---------------------------------
void qaIdeal::run(ProcessingContext& pc)
{

  number_of_ideal_max.fill(0);
  number_of_digit_max.fill(0);
  number_of_ideal_max_findable.fill(0);
  clones.fill(0);
  fractional_clones.fill(0.f);

  // init array
  for (int i = 0; i < o2::tpc::constants::MAXSECTOR; i++) {
    for (int j = 0; j < 25; j++) {
      assignments_ideal[i][j] = 0;
      assignments_digit[i][j] = 0;
      assignments_ideal_findable[i][j] = 0;
      assignments_digit_findable[i][j] = 0;
    }
  }

  numThreads = std::min(numThreads, 36);

  thread_group group;
  if (numThreads > 1) {
    for (int loop_sectors = 0; loop_sectors < o2::tpc::constants::MAXSECTOR; loop_sectors++) {
      group.create_thread(boost::bind(&qaIdeal::runQa, this, loop_sectors));
      if ((loop_sectors + 1) % numThreads == 0 || loop_sectors + 1 == o2::tpc::constants::MAXSECTOR) {
        group.join_all();
      }
    }
  } else {
    for (int loop_sectors = 0; loop_sectors < o2::tpc::constants::MAXSECTOR; loop_sectors++) {
      runQa(loop_sectors);
    }
  }

  unsigned int number_of_ideal_max_sum = std::accumulate(number_of_ideal_max.begin(), number_of_ideal_max.end(), 0), number_of_digit_max_sum = std::accumulate(number_of_digit_max.begin(), number_of_digit_max.end(), 0), number_of_ideal_max_findable_sum = std::accumulate(number_of_ideal_max_findable.begin(), number_of_ideal_max_findable.end(), 0), clones_sum = std::accumulate(clones.begin(), clones.end(), 0);
  float fractional_clones_sum = std::accumulate(fractional_clones.begin(), fractional_clones.end(), 0);

  LOG(info) << "------- RESULTS -------\n";
  LOG(info) << "Number of digit maxima: " << number_of_digit_max_sum;
  LOG(info) << "Number of ideal maxima (total): " << number_of_ideal_max_sum;
  LOG(info) << "Number of ideal maxima (findable): " << number_of_ideal_max_findable_sum << "\n";

  unsigned int efficiency_normal = 0;
  unsigned int efficiency_findable = 0;
  for (int ass = 0; ass < 10; ass++) {
    int ass_dig = 0, ass_id = 0;
    for (int s = 0; s < o2::tpc::constants::MAXSECTOR; s++) {
      ass_dig += assignments_digit[s][ass];
      ass_id += assignments_ideal[s][ass];
      if (ass > 0) {
        efficiency_normal += assignments_ideal[s][ass];
      }
    }
    LOG(info) << "Number of assigned digit maxima (#assignments " << ass << "): " << ass_dig;
    LOG(info) << "Number of assigned ideal maxima (#assignments " << ass << "): " << ass_id << "\n";
  }

  for (int ass = 0; ass < 10; ass++) {
    int ass_dig = 0, ass_id = 0;
    for (int s = 0; s < o2::tpc::constants::MAXSECTOR; s++) {
      ass_dig += assignments_digit_findable[s][ass];
      ass_id += assignments_ideal_findable[s][ass];
    }
    if (ass == 0) {
      ass_id -= (number_of_ideal_max_sum - number_of_ideal_max_findable_sum);
    }
    LOG(info) << "Number of finable assigned digit maxima (#assignments " << ass << "): " << ass_dig;
    LOG(info) << "Number of finable assigned ideal maxima (#assignments " << ass << "): " << ass_id << "\n";
    if (ass > 0) {
      for (int s = 0; s < o2::tpc::constants::MAXSECTOR; s++) {
        efficiency_findable += assignments_ideal_findable[s][ass];
      }
    }
  }

  int ass_dig = 0;
  for (int s = 0; s < o2::tpc::constants::MAXSECTOR; s++) {
    ass_dig += assignments_digit[s][0];
  }
  int ass_id = 0;
  for (int s = 0; s < o2::tpc::constants::MAXSECTOR; s++) {
    ass_id += assignments_ideal[s][0];
  }

  LOG(info) << "Efficiency - Number of assigned (ideal -> digit) clusters: " << efficiency_normal << " (" << (float)efficiency_normal * 100 / (float)number_of_ideal_max_sum << "% of ideal maxima)";
  LOG(info) << "Efficiency (findable) - Number of assigned (ideal -> digit) clusters: " << efficiency_findable << " (" << (float)efficiency_findable * 100 / (float)number_of_ideal_max_findable_sum << "% of ideal maxima)";
  LOG(info) << "Clones (Int, clone-order >= 2 for ideal cluster): " << clones_sum << " (" << (float)clones_sum * 100 / (float)number_of_digit_max_sum << "% of digit maxima)";
  LOG(info) << "Clones (Float, fractional clone-order): " << fractional_clones_sum << " (" << (float)fractional_clones_sum * 100 / (float)number_of_digit_max_sum << "% of digit maxima)";
  LOG(info) << "Fakes for digits (number of digit hits that can't be assigned to any ideal hit): " << ass_dig << " (" << (float)ass_dig * 100 / (float)number_of_digit_max_sum << "% of digit maxima)";
  LOG(info) << "Fakes for ideal (number of ideal hits that can't be assigned to any digit hit): " << ass_id << " (" << (float)ass_id * 100 / (float)number_of_ideal_max_sum << "% of ideal maxima)";

  if (mode.find(std::string("training_data")) != std::string::npos && create_output == 1) {
    LOG(info) << "------- Merging training data -------";
    gSystem->Exec("hadd -k -f ./training_data.root ./training_data_*.root");
  }

  if (mode.find(std::string("native")) != std::string::npos && create_output == 1) {
    LOG(info) << "------- Merging native-ideal assignments -------";
    gSystem->Exec("hadd -k -f ./native_ideal.root ./native_ideal_*.root");
  }

  if (mode.find(std::string("network")) != std::string::npos && create_output == 1) {
    LOG(info) << "------- Merging native-ideal assignments -------";
    gSystem->Exec("hadd -k -f ./network_ideal.root ./network_ideal_*.root");
  }

  pc.services().get<ControlService>().endOfStream();
  pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
}

// ---------------------------------
#include "Framework/runDataProcessing.h"

DataProcessorSpec processIdealClusterizer()
{
  std::vector<InputSpec> inputs;
  std::vector<OutputSpec> outputs;

  return DataProcessorSpec{
    "tpc-qa-ideal",
    inputs,
    outputs,
    adaptFromTask<qaIdeal>(),
    Options{
      {"verbose", VariantType::Int, 0, {"Verbosity level"}},
      {"mode", VariantType::String, "training_data", {"Enables different settings (e.g. creation of training data for NN, running with tpc-native clusters). Options are: training_data, native, network_classification, network_regression, network_full, clusterizer"}},
      {"normalization-mode", VariantType::Int, 1, {"Normalization: 0 = normalization by 1024.f; 1 = normalization by q_center "}},
      {"create-output", VariantType::Int, 1, {"Create output, specific to any given mode."}},
      {"use-max-cog", VariantType::Int, 1, {"Use maxima for assignment = 0, use CoG's = 1"}},
      {"size-pad", VariantType::Int, 11, {"Training data selection size: Images are (size-pad, size-time, size-row)."}},
      {"size-time", VariantType::Int, 11, {"Training data selection size: Images are (size-pad, size-time, size-row)."}},
      {"size-row", VariantType::Int, 1, {"Training data selection size: Images are (size-pad, size-time, size-row)."}},
      {"threads", VariantType::Int, 1, {"Number of CPU threads to be used."}},
      {"looper-tagger-opmode", VariantType::String, "digit", {"Mode in which the looper tagger is run: ideal or digit."}},
      {"looper-tagger-granularity", VariantType::Int, 5, {"Granularity of looper tagger (time bins in which loopers are excluded in rectangular areas)."}},
      {"looper-tagger-padwindow", VariantType::Int, 3, {"Total pad-window size of the looper tagger for evaluating if a region is looper or not."}},
      {"looper-tagger-timewindow", VariantType::Int, 20, {"Total time-window size of the looper tagger for evaluating if a region is looper or not."}},
      {"looper-tagger-threshold-num", VariantType::Int, 5, {"Threshold of number of clusters over which rejection takes place."}},
      {"looper-tagger-threshold-q", VariantType::Float, 600.f, {"Threshold of charge-per-cluster that should be rejected."}},
      {"infile-digits", VariantType::String, "tpcdigits.root", {"Input file name (digits)"}},
      {"infile-native", VariantType::String, "tpc-native-clusters.root", {"Input file name (native)"}},
      {"network-data-output", VariantType::String, "network_out.root", {"Input file for the network output"}},
      {"network-classification-path", VariantType::String, "./net_classification.onnx", {"Absolute path to the network file (classification)"}},
      {"network-regression-path", VariantType::String, "./net_regression.onnx", {"Absolute path to the network file (regression)"}},
      {"network-input-size", VariantType::Int, 1000, {"Size of the vector to be fed through the neural network"}},
      {"network-class-threshold", VariantType::Float, 0.5f, {"Threshold for classification network: Keep or reject maximum (default: 0.5)"}},
      {"enable-network-optimizations", VariantType::Bool, true, {"Enable ONNX network optimizations"}},
      {"network-num-threads", VariantType::Int, 1, {"Set the number of CPU threads for network execution"}}}};
}

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{processIdealClusterizer()};
}