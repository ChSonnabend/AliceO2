#include <cmath>
#include <boost/thread.hpp>
#include <boost/bind.hpp>

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

using namespace o2;
using namespace GPUCA_NAMESPACE::gpu;
using namespace o2::tpc;
using namespace o2::framework;
using namespace o2::ml;
using namespace boost;

class qaIdeal : public Task
{
 public:
  void init(InitContext&) final;

  bool checkIdx(int);

  void read_digits(int);

  void read_ideal(int);

  void read_native(int);

  void read_network(int);

  template <class T>
  T init_map2d(int, std::vector<int>);

  template <class T>
  void fill_map2d(int, std::vector<int>, T, int = 0, int = 0);

  void clear_memory(int, int, int);

  template <class T>
  void find_maxima(int, int, int, T);

  template <class T>
  void run_network(int, int, int, T, int = 0);

  template <class T>
  void overwrite_map2d(int, int, int, T, int = 0);

  template <class T>
  int test_neighbour(int, std::array<int, 3>, std::array<int, 2>, T, int = 1);

  template <class T>
  void runQaSingle(int, int, int, T);

  template <class T>
  void runQa(int, std::vector<int>);

  void run(ProcessingContext&) final;

 private:
  int global_shift[2] = {5, 5};     // shifting digits to select windows easier, (pad, time)
  int charge_limits[2] = {2, 1024}; // upper and lower charge limits
  int verbose = 0;                  // chunk_size in time direction
  int networkInputSize = 100;       // vector input size for neural network
  bool networkOptimizations = true; // ONNX session optimizations
  int networkNumThreads = 10;       // Future: Add Cuda and CoreML Execution providers to run on CPU
  int numThreads = 1;               // Number of threads for multithreading
  std::array<int, o2::tpc::constants::MAXSECTOR> max_time, max_pad;
  std::string mode = "training_data";
  std::string inFileDigits = "tpcdigits.root";
  std::string inFileNative = "tpc-cluster-native.root";
  std::string inFileDigitizer = "mclabels_digitizer.root";
  std::string networkDataOutput = "./network_out.root";
  std::string networkClassification = "./net_classification.onnx";
  std::string networkRegression = "./net_regression.onnx";

  std::array<std::array<std::vector<int>, o2::tpc::constants::MAXGLOBALPADROW>, o2::tpc::constants::MAXSECTOR> maxima_digits;
  std::array<std::array<std::vector<std::array<int, 3>>, o2::tpc::constants::MAXGLOBALPADROW>, o2::tpc::constants::MAXSECTOR> digit_map, ideal_max_map;
  std::array<std::array<std::vector<std::array<float, 3>>, o2::tpc::constants::MAXGLOBALPADROW>, o2::tpc::constants::MAXSECTOR> ideal_cog_map;
  std::array<std::array<std::vector<std::array<float, 2>>, o2::tpc::constants::MAXGLOBALPADROW>, o2::tpc::constants::MAXSECTOR> ideal_sigma_map;
  std::array<std::array<std::vector<float>, o2::tpc::constants::MAXGLOBALPADROW>, o2::tpc::constants::MAXSECTOR> ideal_max_q, ideal_cog_q, digit_q;

  std::vector<std::vector<std::array<int, 2>>> adj_mat = {{{0, 0}}, {{1, 0}, {0, 1}, {-1, 0}, {0, -1}}, {{1, 1}, {-1, 1}, {-1, -1}, {1, -1}}, {{2, 0}, {0, -2}, {-2, 0}, {0, 2}}, {{2, 1}, {1, 2}, {-1, 2}, {-2, 1}, {-2, -1}, {-1, -2}, {1, -2}, {2, -1}}, {{2, 2}, {-2, 2}, {-2, -2}, {2, -2}}};

  std::array<unsigned int, 25> assignments_ideal, assignments_digit, assignments_ideal_findable, assignments_digit_findable;
  unsigned int number_of_ideal_max = 0, number_of_digit_max = 0, number_of_ideal_max_findable = 0;
  unsigned int clones = 0;
  float fractional_clones = 0;
};

// ---------------------------------
void qaIdeal::init(InitContext& ic)
{
  verbose = ic.options().get<int>("verbose");
  mode = ic.options().get<std::string>("mode");
  numThreads = ic.options().get<int>("threads");
  inFileDigits = ic.options().get<std::string>("infile-digits");
  inFileNative = ic.options().get<std::string>("infile-native");
  networkDataOutput = ic.options().get<std::string>("network-data-output");
  networkClassification = ic.options().get<std::string>("network-classification-path");
  networkRegression = ic.options().get<std::string>("network-regression-path");
  networkInputSize = ic.options().get<int>("network-input-size");
  networkOptimizations = ic.options().get<bool>("enable-network-optimizations");
  networkNumThreads = ic.options().get<int>("network-num-threads");

  if (verbose >= 1)
    LOG(info) << "Initialized QA macro!";
}

// ---------------------------------
bool qaIdeal::checkIdx(int idx)
{
  return ((idx > -1) && (idx < 20000000));
}

// ---------------------------------
void qaIdeal::clear_memory(int sector, int globalrow, int localrow)
{
  maxima_digits[sector][globalrow].clear();
  digit_map[sector][globalrow].clear();
  ideal_max_map[sector][globalrow].clear();
  ideal_cog_map[sector][globalrow].clear();
  ideal_sigma_map[sector][globalrow].clear();
  ideal_max_q[sector][globalrow].clear();
  ideal_cog_q[sector][globalrow].clear();
  digit_q[sector][globalrow].clear();

  if (verbose >= 1)
    LOG(info) << "Cleared the 2D charge map!";
}

// ---------------------------------
void qaIdeal::read_digits(int sector)
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

  digitTree->GetEntry(0);
  if (verbose >= 1)
    LOG(info) << "Trying to read " << digits->size() << " digits";

  std::vector<int> sizes(o2::tpc::constants::MAXGLOBALPADROW, 0);

  for (unsigned int i_digit = 0; i_digit < digits->size(); i_digit++) {
    const auto& digit = (*digits)[i_digit];
    current_time = digit.getTimeStamp();
    current_pad = digit.getPad();
    if (current_time > max_time[sector])
      max_time[sector] = current_time;
    if (current_pad > max_pad[sector])
      max_pad[sector] = current_pad;
    sizes[digit.getRow()]++;
  }

  for (int row = 0; row < o2::tpc::constants::MAXGLOBALPADROW; row++) {
    digit_map[sector][row].resize(sizes[row]);
    digit_q[sector][row].resize(sizes[row]);
  }

  std::fill(sizes.begin(), sizes.end(), 0);
  for (unsigned int i_digit = 0; i_digit < digits->size(); i_digit++) {
    const auto& digit = (*digits)[i_digit];
    current_row = digit.getRow();
    digit_map[sector][current_row][sizes[current_row]] = std::array<int, 3>{current_row, digit.getPad(), current_time};
    digit_q[sector][current_row][sizes[current_row]] = digit.getChargeFloat();
    sizes[current_row]++;
  }
  (*digits).clear();

  digitFile->Close();

  if (verbose >= 1)
    LOG(info) << "Done reading digits!";
}

// ---------------------------------
void qaIdeal::read_native(int sector)
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

  for (int i = 0; i < tpcClusterReader.getTreeSize(); ++i) {
    tpcClusterReader.read(i);
    tpcClusterReader.fillIndex(clusterIndex, clusterBuffer, clusterMCBuffer);

    int nClustersSec = 0;
    for (int row = 0; row < o2::tpc::constants::MAXGLOBALPADROW; ++row) {
      digit_map[sector][row].resize(clusterIndex.nClusters[sector][row]);
      digit_q[sector][row].resize(clusterIndex.nClusters[sector][row]);
      nClustersSec += clusterIndex.nClusters[sector][row];
    }
    if (verbose >= 2) {
      LOG(info) << "Native clusters in sector " << sector << ": " << nClustersSec;
    }
    int count_clusters = 0;
    for (int row = 0; row < o2::tpc::constants::MAXGLOBALPADROW; ++row) {
      const int nClusters = clusterIndex.nClusters[sector][row];
      if (!nClusters) {
        continue;
      }
      for (int icl = 0; icl < nClusters; ++icl) {
        const auto& cl = *(clusterIndex.clusters[sector][row] + icl);
        clusters.processCluster(cl, Sector(sector), row);
        digit_map[sector][row][count_clusters] = std::array<int, 3>{row, static_cast<int>(round(cl.getPad())), static_cast<int>(round(cl.getTime()))};
        digit_q[sector][row][count_clusters] = cl.getQtot();
        if (cl.getTime() > max_time[sector])
          max_time[sector] = cl.getTime();
        if (cl.getPad() > max_pad[sector])
          max_pad[sector] = cl.getPad();
        count_clusters++;
      }
    }
  }

  if (verbose >= 1)
    LOG(info) << "Done reading native clusters!";
}

// ---------------------------------
void qaIdeal::read_network(int sector)
{

  if (verbose >= 1)
    LOG(info) << "Reading network output...";

  // reading in the raw digit information
  TFile* networkFile = TFile::Open(networkDataOutput.c_str());
  TTree* networkTree = (TTree*)networkFile->Get("data_tree");

  double row, pad, time, reg_pad, reg_time, reg_qRatio;
  int size = 0;

  networkTree->SetBranchAddress("sector", &sector);
  networkTree->SetBranchAddress("row", &row);
  networkTree->SetBranchAddress("pad", &pad);
  networkTree->SetBranchAddress("time", &time);
  networkTree->SetBranchAddress("reg_pad", &reg_pad);
  networkTree->SetBranchAddress("reg_time", &reg_time);
  networkTree->SetBranchAddress("reg_qRatio", &reg_qRatio);

  for (unsigned int j = 0; j < networkTree->GetEntries(); j++) {
    try {
      networkTree->GetEntry(j);
      size++;
      if (round(time + reg_time) > max_time[sector])
        max_time[sector] = round(time + reg_time);
      if (round(pad + reg_pad) > max_pad[sector])
        max_pad[sector] = round(pad + reg_pad);
    } catch (...) {
      LOG(info) << "(Digitizer) Problem occured in sector " << sector;
    }
  }

  for (int padrow = 0; padrow < o2::tpc::constants::MAXGLOBALPADROW; padrow++) {
    digit_map[sector][padrow].resize(size);
    digit_q[sector][padrow].resize(size);
  }

  for (unsigned int j = 0; j < networkTree->GetEntries(); j++) {
    try {
      networkTree->GetEntry(j);
      digit_map[sector][row][size] = std::array<int, 3>{(int)row, static_cast<int>(round(pad + reg_pad)), static_cast<int>(round(time + reg_time))};
      digit_q[sector][row][size] = 1000;
      size++;
    } catch (...) {
      LOG(info) << "(Digitizer) Problem occured in sector " << sector;
    }
  }

  networkFile->Close();

  if (verbose >= 1)
    LOG(info) << "Done reading digits!";
}

// ---------------------------------
void qaIdeal::read_ideal(int sector)
{

  int sec, row, maxp, maxt, pcount, lab;
  float cogp, cogt, cogq, maxq, sigmap, sigmat;

  if (verbose > 0)
    LOG(info) << "Reading ideal clusterizer, sector " << sector << " ...";
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

  std::vector<int> sizes(o2::tpc::constants::MAXGLOBALPADROW, 0);
  for (unsigned int j = 0; j < digitizerSector->GetEntries(); j++) {
    digitizerSector->GetEntry(j);
    sizes[row]++;
  }
  for (int padrow = 0; padrow < o2::tpc::constants::MAXGLOBALPADROW; padrow++) {
    ideal_max_map[sector][padrow].resize(sizes[padrow]);
    ideal_max_q[sector][padrow].resize(sizes[padrow]);
    ideal_cog_map[sector][padrow].resize(sizes[padrow]);
    ideal_sigma_map[sector][padrow].resize(sizes[padrow]);
    ideal_cog_q[sector][padrow].resize(sizes[padrow]);
  }
  std::fill(sizes.begin(), sizes.end(), 0);

  if (verbose >= 1)
    LOG(info) << "Trying to read " << digitizerSector->GetEntries() << " ideal digits";
  for (unsigned int j = 0; j < digitizerSector->GetEntries(); j++) {
    try {
      digitizerSector->GetEntry(j);
      // ideal_point_count.push_back(pcount);

      // LOG(info) << "Entry: " << j << "; Map sizes: " << ideal_max_map[sector][row].size() << "; row: " << row << "; sizes[row]: " << sizes[row];

      ideal_max_map[sector][row][sizes[row]] = std::array<int, 3>{row, maxp, maxt};
      ideal_max_q[sector][row][sizes[row]] = maxq;
      ideal_cog_map[sector][row][sizes[row]] = std::array<float, 3>{(float)row, cogp, cogt};
      ideal_sigma_map[sector][row][sizes[row]] = std::array<float, 2>{sigmap, sigmat};
      ideal_cog_q[sector][row][sizes[row]] = cogq;
      sizes[row]++;

    } catch (...) {
      LOG(info) << "(Digitizer) Problem occured in sector " << sector;
    }
  }
  inputFile->Close();
}

template <class T>
T qaIdeal::init_map2d(int maxtime, std::vector<int> rows)
{
  // auto temp_arr = T{}[0][0];
  T map2d;
  for (int i = 0; i < 2; i++) {
    map2d[i].resize(maxtime + (2 * global_shift[1]) + 1);
    for (int time_size = 0; time_size < maxtime + (2 * global_shift[1]) + 1; time_size++) {
      // map2d[i].push_back(std::array<std::array<int, 170>, 10>{});
      for (int row = 0; row < rows[1] - rows[0]; row++) {
        for (int pad = 0; pad < 170; pad++) {
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
void qaIdeal::fill_map2d(int sector, std::vector<int> rows, T map2d, int fillmode, int use_max_cog)
{

  // assert(map2d[0][sector].size() == rows.size()); //s Add later
  for (int row = rows[0]; row < rows[1]; row++) {
    if (use_max_cog == 0) {
      // Storing the indices
      if (fillmode == 0) {
        for (unsigned int ind = 0; ind < digit_map[sector][row].size(); ind++) {
          map2d[1][digit_map[sector][row][ind][2] + global_shift[1]][row - rows[0]][digit_map[sector][row][ind][1] + global_shift[0]] = ind;
        }
      } else if (fillmode == 1) {
        for (unsigned int ind = 0; ind < ideal_max_map[sector][row].size(); ind++) {
          map2d[0][ideal_max_map[sector][row][ind][2] + global_shift[1]][row - rows[0]][ideal_max_map[sector][row][ind][1] + global_shift[0]] = ind;
        }
      } else if (fillmode == -1) {
        for (unsigned int ind = 0; ind < digit_map[sector][row].size(); ind++) {
          map2d[1][digit_map[sector][row][ind][2] + global_shift[1]][row - rows[0]][digit_map[sector][row][ind][1] + global_shift[0]] = ind;
        }
        for (unsigned int ind = 0; ind < ideal_max_map[sector][row].size(); ind++) {
          map2d[0][ideal_max_map[sector][row][ind][2] + global_shift[1]][row - rows[0]][ideal_max_map[sector][row][ind][1] + global_shift[0]] = ind;
        }
      } else {
        LOG(info) << "Fillmode unknown! No fill performed!";
      }
    } else if (use_max_cog == 1) {
      // Storing the indices
      if (fillmode == 0) {
        for (unsigned int ind = 0; ind < digit_map[sector][row].size(); ind++) {
          map2d[1][digit_map[sector][row][ind][2] + global_shift[1]][row - rows[0]][digit_map[sector][row][ind][1] + global_shift[0]] = ind;
        }
      } else if (fillmode == 1) {
        for (unsigned int ind = 0; ind < ideal_cog_map[sector][row].size(); ind++) {
          map2d[0][round(ideal_cog_map[sector][row][ind][2]) + global_shift[1]][row - rows[0]][round(ideal_cog_map[sector][row][ind][1]) + global_shift[0]] = ind;
        }
      } else if (fillmode == -1) {
        for (unsigned int ind = 0; ind < digit_map[sector][row].size(); ind++) {
          // LOG(info) << "Works for index " << ind << "/" << digit_map[sector][row].size() << " indexing " << sector << ", " << row << ", " << digit_map[sector][row][ind][1] + global_shift[0] << ", " << digit_map[sector][row][ind][2] + global_shift[1];
          // LOG(info) << "Max size is " << map2d[1].size() << ", " << map2d[1][0][0].size();
          map2d[1][digit_map[sector][row][ind][2] + global_shift[1]][row - rows[0]][digit_map[sector][row][ind][1] + global_shift[0]] = ind;
        }
        for (unsigned int ind = 0; ind < ideal_cog_map[sector][row].size(); ind++) {
          map2d[0][round(ideal_cog_map[sector][row][ind][2]) + global_shift[1]][row - rows[0]][round(ideal_cog_map[sector][row][ind][1]) + global_shift[0]] = ind;
        }
      } else {
        LOG(info) << "Fillmode unknown! No fill performed!";
      }
    }
  }
}

// ---------------------------------
template <class T>
void qaIdeal::find_maxima(int sector, int globalrow, int localrow, T map2d)
{

  if (verbose >= 1) {
    LOG(info) << "Finding local maxima";
  }

  bool is_max = true;
  float current_charge = 0;
  if (verbose >= 3)
    LOG(info) << "Finding maxima in row " << globalrow;
  for (int pad = 0; pad < 170; pad++) {
    for (int time = 0; time < max_time[sector]; time++) {
      if (checkIdx(map2d[1][time + global_shift[1]][localrow][pad + global_shift[0]])) {

        current_charge = digit_q[sector][globalrow][map2d[1][time + global_shift[1]][localrow][pad + global_shift[0]]];
        if (map2d[1][time + global_shift[1]][localrow][pad + global_shift[0] + 1] != -1) {
          is_max = (current_charge >= digit_q[sector][globalrow][map2d[1][time + global_shift[1]][localrow][pad + global_shift[0] + 1]]);
        }

        if (map2d[1][time + global_shift[1] + 1][localrow][pad + global_shift[0]] != -1 && is_max) {
          is_max = (current_charge >= digit_q[sector][globalrow][map2d[1][time + global_shift[1] + 1][localrow][pad + global_shift[0]]]);
        }

        if (map2d[1][time + global_shift[1]][localrow][pad + global_shift[0] - 1] != -1 && is_max) {
          is_max = (current_charge >= digit_q[sector][globalrow][map2d[1][time + global_shift[1]][localrow][pad + global_shift[0] - 1]]);
        }

        if (map2d[1][time + global_shift[1] - 1][localrow][pad + global_shift[0]] != -1 && is_max) {
          is_max = (current_charge >= digit_q[sector][globalrow][map2d[1][time + global_shift[1] - 1][localrow][pad + global_shift[0]]]);
        }

        if (map2d[1][time + global_shift[1] + 1][localrow][pad + global_shift[0] + 1] != -1 && is_max) {
          is_max = (current_charge >= digit_q[sector][globalrow][map2d[1][time + global_shift[1] + 1][localrow][pad + global_shift[0] + 1]]);
        }

        if (map2d[1][time + global_shift[1] - 1][localrow][pad + global_shift[0] + 1] != -1 && is_max) {
          is_max = (current_charge >= digit_q[sector][globalrow][map2d[1][time + global_shift[1] - 1][localrow][pad + global_shift[0] + 1]]);
        }

        if (map2d[1][time + global_shift[1] + 1][localrow][pad + global_shift[0] - 1] != -1 && is_max) {
          is_max = (current_charge >= digit_q[sector][globalrow][map2d[1][time + global_shift[1] + 1][localrow][pad + global_shift[0] - 1]]);
        }

        if (map2d[1][time + global_shift[1] - 1][localrow][pad + global_shift[0] - 1] != -1 && is_max) {
          is_max = (current_charge >= digit_q[sector][globalrow][map2d[1][time + global_shift[1] - 1][localrow][pad + global_shift[0] - 1]]);
        }

        if (is_max) {
          maxima_digits[sector][globalrow].push_back(map2d[1][time + global_shift[1]][localrow][pad + global_shift[0]]);
        }
        is_max = true;
      }
    }
    if (verbose >= 3)
      LOG(info) << "Found " << maxima_digits[sector][globalrow].size() << " maxima in row " << globalrow;
  }

  if (verbose >= 1)
    LOG(info) << "Found " << maxima_digits[sector][globalrow].size() << " maxima. Done!";
}

// ---------------------------------
template <class T>
void qaIdeal::run_network(int sector, int globalrow, int localrow, T map2d, int mode)
{

  OnnxModel network;

  // Loading the data
  // std::vector<std::vector<std::vector<std::vector<float>>>> input_vector(maxima_digits.size(), std::vector<std::vector<std::vector<float>>>(1, std::vector<std::vector<float>>(2*global_shift[0]+1, std::vector<float>(2*global_shift[1]+1,0)))); // 11x11 images for each maximum of the digits
  std::vector<float> input_vector(maxima_digits[sector][globalrow].size() * (2 * global_shift[0] + 1) * (2 * global_shift[1] + 1), 0.);
  std::vector<float> central_charges(maxima_digits[sector][globalrow].size(), 0.);

  std::vector<float> output_network_class(maxima_digits[sector][globalrow].size(), 0), output_network_reg;
  std::vector<float> temp_input(networkInputSize * (2 * global_shift[0] + 1) * (2 * global_shift[1] + 1), 0);

  for (unsigned int max = 0; max < maxima_digits[sector][globalrow].size(); max++) {
    central_charges[max] = digit_q[sector][globalrow][map2d[1][digit_map[sector][localrow][maxima_digits[sector][globalrow][max]][2] + global_shift[1]][localrow][digit_map[sector][globalrow][maxima_digits[sector][globalrow][max]][1] + global_shift[0]]];
    for (int pad = 0; pad < 2 * global_shift[0] + 1; pad++) {
      for (int time = 0; time < 2 * global_shift[1] + 1; time++) {
        // input_vector[(2*global_shift[0]+1)*pad + time][0][pad][time] = digit_q[sector][map2d[1][digit_map[sector][maxima_digits[max]][2] + time][digit_map[sector][maxima_digits[max]][0]][digit_map[sector][maxima_digits[max]][1] + pad]] / central_charges[max];
        input_vector[max * (2 * global_shift[0] + 1) * (2 * global_shift[1] + 1) + (2 * global_shift[1] + 1) * pad + time] = digit_q[sector][globalrow][map2d[1][digit_map[sector][globalrow][maxima_digits[sector][globalrow][max]][2] + time][localrow][digit_map[sector][globalrow][maxima_digits[sector][globalrow][max]][1] + pad]] / central_charges[max];
      }
    }
  }

  if (mode >= 0) {
    network.init(networkClassification, networkOptimizations, networkNumThreads);
    for (int max = 0; max < maxima_digits[sector][globalrow].size(); max++) {
      for (int idx = 0; idx < (2 * global_shift[0] + 1) * (2 * global_shift[1] + 1); idx++) {
        temp_input[(max % networkInputSize) * (2 * global_shift[0] + 1) * (2 * global_shift[1] + 1) + idx] = input_vector[max * (2 * global_shift[0] + 1) * (2 * global_shift[1] + 1) + idx];
      }
      if ((max + 1) % networkInputSize == 0) {
        float* out_net = network.inference(temp_input, temp_input.size());
        for (int idx = 0; idx < networkInputSize; idx++) {
          output_network_class[int(max / networkInputSize - 1) * (networkInputSize) + idx] = out_net[idx];
        }
      }
    }
    LOG(info) << "Classification network done!";
  }

  if (mode >= 1) {
    network.init(networkRegression, networkOptimizations, networkNumThreads);
    int count_num_maxima = 0;
    for (int max = 0; max < maxima_digits[sector][globalrow].size(); max++) {
      if (output_network_class[max] > 0.5) {
        for (int idx = 0; idx < (2 * global_shift[0] + 1) * (2 * global_shift[1] + 1); idx++) {
          temp_input[(max % networkInputSize) * (2 * global_shift[0] + 1) * (2 * global_shift[1] + 1) + idx] = input_vector[max * (2 * global_shift[0] + 1) * (2 * global_shift[1] + 1) + idx];
        }
      }
      if ((max + 1) % networkInputSize == 0) {
        float* out_net = network.inference(temp_input, temp_input.size());
        for (int idx = 0; idx < 3 * networkInputSize; idx++) {
          output_network_reg.push_back(out_net[idx]);
        }
      } else {
        maxima_digits[sector][globalrow].erase(maxima_digits[sector][globalrow].begin() + max);
      }
    }
    LOG(info) << "Regression network done!";

    std::vector<int> rows_max(maxima_digits[sector][globalrow].size());
    for (unsigned int max = 0; max < maxima_digits[sector][globalrow].size(); max++) {
      rows_max[max] = digit_map[sector][globalrow][maxima_digits[sector][globalrow][max]][0];
    }
    std::iota(std::begin(maxima_digits[sector][globalrow]), std::end(maxima_digits[sector][globalrow]), 0);
    digit_map[sector][globalrow].clear();
    digit_q[sector][globalrow].clear();
    digit_map[sector][globalrow].resize(maxima_digits[sector][globalrow].size());
    digit_q[sector][globalrow].resize(maxima_digits[sector][globalrow].size());

    for (int i = 0; i < maxima_digits[sector][globalrow].size(); i++) {
      digit_map[sector][globalrow][i] = std::array<int, 3>{rows_max[i], static_cast<int>(round(output_network_reg[3 * i])), static_cast<int>(round(output_network_reg[3 * i + 1]))};
      digit_q[sector][globalrow][i] = output_network_reg[3 * i + 2] * central_charges[i];
    }
    LOG(info) << "Digit map written.";
  } else {
    LOG(fatal) << "(Network evaluation error) Mode unknown!";
  }
}

// ---------------------------------
template <class T>
void qaIdeal::overwrite_map2d(int sector, int globalrow, int localrow, T map2d, int mode)
{

  if (mode == 0) {
    for (int pad = 0; pad < 170; pad++) {
      for (int time = 0; time < (max_time[sector] + 2 * global_shift[1] + 1); time++) {
        map2d[1][time][localrow][pad] = -1;
      }
    }
    for (unsigned int max = 0; max < maxima_digits[sector][globalrow].size(); max++) {
      map2d[1][digit_map[sector][globalrow][maxima_digits[sector][globalrow][max]][2] + global_shift[1]][localrow][digit_map[sector][globalrow][maxima_digits[sector][globalrow][max]][1] + global_shift[0]] = max;
    }
  } else if (mode == 1) {
    for (int pad = 0; pad < 170; pad++) {
      for (int time = 0; time < (max_time[sector] + 2 * global_shift[1]); time++) {
        map2d[0][time][localrow][pad] = -1;
      }
    }
    for (unsigned int dig = 0; dig < digit_q[sector][globalrow].size(); dig++) {
      map2d[0][digit_map[sector][globalrow][dig][2] + global_shift[1]][localrow][digit_map[sector][globalrow][dig][1] + global_shift[0]] = dig;
    }
  } else {
    LOG(fatal) << "Mode unknown!!!";
  }
}

// ---------------------------------
template <class T>
int qaIdeal::test_neighbour(int localrow, std::array<int, 3> index, std::array<int, 2> nn, T map2d, int mode)
{
  return map2d[mode][(int)index[2] + global_shift[1] + nn[1]][localrow][(int)index[1] + global_shift[0] + nn[0]];
}

// ---------------------------------
template <class T>
void qaIdeal::runQaSingle(int sector, int globalrow, int localrow, T map2d)
{

  LOG(info) << "Starting process for sector " << sector;

  if ((mode.find(std::string("network")) == std::string::npos) && (mode.find(std::string("native")) == std::string::npos)) {
    find_maxima<T>(sector, globalrow, localrow, map2d);
    overwrite_map2d<T>(sector, globalrow, localrow, map2d);
  } else {
    maxima_digits[sector][globalrow].resize(digit_q[sector][globalrow].size());
    std::iota(std::begin(maxima_digits[sector][globalrow]), std::end(maxima_digits[sector][globalrow]), 0);
  }

  // if(mode.find(std::string("native")) == std::string::npos){
  //   find_maxima(loop_sectors);
  //   if(mode.find(std::string("network")) != std::string::npos && mode.find(std::string("network_reg")) == std::string::npos){
  //     run_network(loop_sectors, map2d, 0); // classification of maxima
  //   }
  //   else if(mode.find(std::string("network_reg")) != std::string::npos){
  //     run_network(loop_sectors, map2d, 1); // classification + regression
  //   }
  //   overwrite_map2d(loop_sectors);
  // }
  // else{
  //   maxima_digits.resize(digit_q[loop_sectors].size());
  //   std::iota(std::begin(maxima_digits), std::end(maxima_digits), 0);
  // }

  // effCloneFake(0, loop_chunks*chunk_size);
  // Assignment at d=1
  LOG(info) << "Maxima found in digits (before): " << maxima_digits.size() << "; Maxima found in ideal clusters (before): " << ideal_max_map[sector][globalrow].size();

  std::vector<int> assigned_ideal(ideal_max_map[sector][globalrow].size(), 0), clone_order(maxima_digits[sector][globalrow].size(), 0);
  std::vector<std::array<int, 25>> assignments_dig_to_id(ideal_max_map[sector][globalrow].size());
  std::vector<int> assigned_digit(maxima_digits[sector][globalrow].size(), 0);
  std::vector<std::array<int, 25>> assignments_id_to_dig(maxima_digits[sector][globalrow].size());
  int current_neighbour;
  std::vector<float> fractional_clones_vector(maxima_digits[sector][globalrow].size(), 0);

  for (int i = 0; i < ideal_max_map[sector][globalrow].size(); i++) {
    for (int j = 0; j < 25; j++) {
      assignments_dig_to_id[i][j] = -1;
    }
  }
  for (int i = 0; i < maxima_digits[sector][globalrow].size(); i++) {
    for (int j = 0; j < 25; j++) {
      assignments_id_to_dig[i][j] = -1;
    }
  }

  std::fill(assigned_digit.begin(), assigned_digit.end(), 0);
  std::fill(assigned_ideal.begin(), assigned_ideal.end(), 0);
  std::fill(clone_order.begin(), clone_order.end(), 0);

  number_of_digit_max += maxima_digits[sector][globalrow].size();
  number_of_ideal_max += ideal_max_map[sector][globalrow].size();

  for (int max = 0; max < ideal_max_map[sector][globalrow].size(); max++) {
    if (ideal_cog_q[sector][globalrow][max] >= 5 && ideal_max_q[sector][globalrow][max] >= 3) {
      number_of_ideal_max_findable++;
    }
  }

  // Level-1 loop: Goes through the layers specified by the adjacency matrix <-> Loop of possible distances
  int layer_count = 0;
  for (int layer = 0; layer < adj_mat.size(); layer++) {

    if (verbose >= 3)
      LOG(info) << "Layer " << layer;

    // Level-2 loop: Goes through the elements of the adjacency matrix at distance d, n times to assign neighbours iteratively
    for (int nn = 0; nn < adj_mat[layer].size(); nn++) {

      // Level-3 loop: Goes through all digit maxima and checks neighbourhood for potential ideal maxima
      for (unsigned int locdigit = 0; locdigit < maxima_digits[sector][globalrow].size(); locdigit++) {
        // LOG(info) << "Failure? " << localrow;
        // LOG(info) << "Failure? " << maxima_digits[sector][globalrow][locdigit];
        // LOG(info) << "Failure? " << adj_mat[layer][nn][0];
        // LOG(info) << "Failure? " << adj_mat[layer][nn][1];
        // LOG(info) << "Failure? " << digit_map[sector][globalrow][maxima_digits[sector][globalrow][locdigit]][0];
        // LOG(info) << "Failure? " << digit_map[sector][globalrow][maxima_digits[sector][globalrow][locdigit]][1];
        // LOG(info) << "Failure? " << digit_map[sector][globalrow][maxima_digits[sector][globalrow][locdigit]][2];

        current_neighbour = test_neighbour(localrow, digit_map[sector][globalrow][maxima_digits[sector][globalrow][locdigit]], adj_mat[layer][nn], map2d, 0);
        // if (verbose >= 5) LOG(info) << "current_neighbour: " << current_neighbour;
        // if (verbose >= 5) LOG(info) << "Maximum digit " << maxima_digits[locdigit];
        // if (verbose >= 5) LOG(info) << "Digit max index: " << digit_map[loop_sectors][maxima_digits[locdigit]][0] << " " << digit_map[loop_sectors][maxima_digits[locdigit]][1] << " " << digit_map[loop_sectors][maxima_digits[locdigit]][2];
        if (current_neighbour >= -1 && current_neighbour <= 20000000) {
          assignments_id_to_dig[locdigit][layer_count + nn] = ((current_neighbour != -1 && assigned_digit[locdigit] == 0) ? (assigned_ideal[current_neighbour] == 0 ? current_neighbour : -1) : -1);
        } else {
          assignments_id_to_dig[locdigit][layer_count + nn] = -1;
          LOG(warning) << "Current neighbour: " << current_neighbour << "; Ideal max index: " << digit_map[sector][globalrow][maxima_digits[sector][globalrow][locdigit]][0] << " " << digit_map[sector][globalrow][maxima_digits[sector][globalrow][locdigit]][1] << " " << digit_map[sector][globalrow][maxima_digits[sector][globalrow][locdigit]][2];
        }
      }
      if (verbose >= 4)
        LOG(info) << "Done with assignment for digit maxima layer " << layer;

      // Level-3 loop: Goes through all ideal maxima and checks neighbourhood for potential digit maxima
      std::array<int, 3> rounded_cog;
      for (unsigned int locideal = 0; locideal < ideal_max_map[sector][globalrow].size(); locideal++) {
        for (int i = 0; i < 3; i++) {
          rounded_cog[i] = round(ideal_cog_map[sector][globalrow][locideal][i]);
        }
        current_neighbour = test_neighbour(localrow, rounded_cog, adj_mat[layer][nn], map2d, 1);
        // if (verbose >= 5) LOG(info) << "current_neighbour: " << current_neighbour;
        // if (verbose >= 5) LOG(info) << "Maximum ideal " << locideal;
        // if (verbose >= 5) LOG(info) << "Ideal max index: " << ideal_max_map[loop_sectors][locideal][0] << " " << ideal_max_map[loop_sectors][locideal][1] << " " << ideal_max_map[loop_sectors][locideal][2];
        if (current_neighbour >= -1 && current_neighbour <= 20000000) {
          assignments_dig_to_id[locideal][layer_count + nn] = ((current_neighbour != -1 && assigned_ideal[locideal] == 0) ? (assigned_digit[current_neighbour] == 0 ? current_neighbour : -1) : -1);
        } else {
          assignments_dig_to_id[locideal][layer_count + nn] = -1;
          LOG(warning) << "Current neighbour: " << current_neighbour << "; Ideal max index: " << ideal_cog_map[sector][globalrow][locideal][0] << " " << ideal_cog_map[sector][globalrow][locideal][1] << " " << ideal_cog_map[sector][globalrow][locideal][2];
        }
      }
      if (verbose >= 4)
        LOG(info) << "Done with assignment for ideal maxima layer " << layer;
    }

    // Level-2 loop: Checks all digit maxima and how many ideal maxima neighbours have been found in the current layer
    if (layer >= 2) {
      for (unsigned int locdigit = 0; locdigit < maxima_digits[sector][globalrow].size(); locdigit++) {
        assigned_digit[locdigit] = 0;
        for (int counter_max = 0; counter_max < 25; counter_max++) {
          if (checkIdx(assignments_id_to_dig[locdigit][counter_max])) {
            assigned_digit[locdigit] += 1;
          }
        }
      }
    }

    // Level-2 loop: Checks all ideal maxima and how many digit maxima neighbours have been found in the current layer
    for (unsigned int locideal = 0; locideal < ideal_max_map[sector][globalrow].size(); locideal++) {
      assigned_ideal[locideal] = 0;
      for (int counter_max = 0; counter_max < 25; counter_max++) {
        if (checkIdx(assignments_dig_to_id[locideal][counter_max])) {
          assigned_ideal[locideal] += 1;
        }
      }
    }

    // Assign all possible digit maxima once you are above a certain distance away from the current maximum (here: At layer with distance greater than sqrt(2))

    if (verbose >= 2)
      LOG(info) << "Removed maxima for layer " << layer;

    layer_count += adj_mat[layer].size();

    if (verbose >= 3)
      LOG(info) << "Layer count is now " << layer_count;
  }

  // Checks the number of assignments that have been made with the above loops
  int count_elements_findable = 0, count_elements_dig = 0, count_elements_id = 0;
  for (unsigned int ass_id = 0; ass_id < ideal_max_map[sector][globalrow].size(); ass_id++) {
    count_elements_id = 0;
    count_elements_findable = 0;
    for (auto elem : assignments_dig_to_id[ass_id]) {
      if (checkIdx(elem)) {
        count_elements_id += 1;
        if (ideal_cog_q[sector][globalrow][ass_id] >= 5 && ideal_max_q[sector][globalrow][ass_id] >= 3) {
          count_elements_findable += 1;
        }
      }
    }
    // if (verbose >= 5 && (ass_id%10000)==0) LOG(info) << "Count elements: " << count_elements_id << " ass_id: " << ass_id << " assignments_ideal: " << assignments_ideal[count_elements_id];
    assignments_ideal[count_elements_id] += 1;
    assignments_ideal_findable[count_elements_findable] += 1;
  }
  for (unsigned int ass_dig = 0; ass_dig < maxima_digits[sector][globalrow].size(); ass_dig++) {
    count_elements_dig = 0;
    count_elements_findable = 0;
    for (auto elem : assignments_id_to_dig[ass_dig]) {
      if (checkIdx(elem)) {
        count_elements_dig += 1;
        if (ideal_cog_q[sector][globalrow][elem] >= 5 && ideal_max_q[sector][globalrow][elem] >= 3) {
          count_elements_findable += 1;
        }
      }
    }
    // if (verbose >= 5 && (ass_dig%10000)==0) LOG(info) << "Count elements: " << count_elements_dig << " ass_dig: " << ass_dig << " assignments_digit: " << assignments_digit[count_elements_dig];
    assignments_digit[count_elements_dig] += 1;
    assignments_digit_findable[count_elements_findable] += 1;
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
  for (unsigned int locdigit = 0; locdigit < maxima_digits[sector][globalrow].size(); locdigit++) {
    if (clone_order[locdigit] > 1) {
      clones += 1;
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
    fractional_clones += elem_frac;
  }

  if (verbose >= 3)
    LOG(info) << "Done with determining the clone rate";

  if (verbose >= 4) {
    for (int ass = 0; ass < 25; ass++) {
      LOG(info) << "Number of assignments to one digit maximum (#assignments " << ass << "): " << assignments_digit[ass];
      LOG(info) << "Number of assignments to one ideal maximum (#assignments " << ass << "): " << assignments_ideal[ass] << "\n";
    }
  }

  if (mode.find(std::string("training_data")) != std::string::npos) {

    overwrite_map2d<T>(sector, globalrow, localrow, map2d, 1);

    if (verbose >= 3)
      LOG(info) << "Creating training data...";

    // creating training data for the neural network
    int mat_size_time = (global_shift[1] * 2 + 1), mat_size_pad = (global_shift[0] * 2 + 1), data_size = maxima_digits[sector][globalrow].size();

    std::vector<std::vector<float>> atomic_unit;
    atomic_unit.resize(mat_size_time, std::vector<float>(mat_size_pad, 0));
    std::vector<std::vector<std::vector<float>>> tr_data_X(data_size);
    std::fill(tr_data_X.begin(), tr_data_X.end(), atomic_unit);

    std::vector<int> tr_data_Y_class(data_size, -1);
    std::array<std::vector<float>, 5> tr_data_Y_reg;
    std::fill(tr_data_Y_reg.begin(), tr_data_Y_reg.end(), std::vector<float>(data_size, -1));

    std::fill(assigned_ideal.begin(), assigned_ideal.end(), 0);
    std::fill(assigned_digit.begin(), assigned_digit.end(), 0);

    if (verbose >= 3)
      LOG(info) << "Initialized arrays...";

    // Some useful variables
    int map_dig_idx = 0, map_q_idx = 0, check_assignment = 0, index_assignment = -1, current_idx_id = -1, current_idx_dig = -1;
    float distance_assignment = 100000.f, current_distance_dig_to_id = 0, current_distance_id_to_dig = 0;
    bool is_min_dist = true;

    for (int max_point = 0; max_point < data_size; max_point++) {
      map_dig_idx = map2d[1][digit_map[sector][globalrow][maxima_digits[sector][globalrow][max_point]][2] + global_shift[1]][localrow][digit_map[sector][globalrow][maxima_digits[sector][globalrow][max_point]][1] + global_shift[0]];
      if (checkIdx(map_dig_idx)) {
        // if (verbose >= 5) LOG(info) << "Current elem at index [" <<digit_map[loop_sectors][maxima_digits[max_point]][2] << " " << digit_map[loop_sectors][maxima_digits[max_point]][0] << " " << digit_map[loop_sectors][maxima_digits[max_point]][1] << "] has value " << map_dig_idx;
        float q_max = digit_q[sector][globalrow][maxima_digits[sector][globalrow][map_dig_idx]];
        for (int time = 0; time < mat_size_time; time++) {
          for (int pad = 0; pad < mat_size_pad; pad++) {
            map_q_idx = map2d[0][digit_map[sector][globalrow][maxima_digits[sector][globalrow][max_point]][2] + time][localrow][digit_map[sector][globalrow][maxima_digits[sector][globalrow][max_point]][1] + pad];
            map_q_idx == -1 ? tr_data_X[max_point][time][pad] = 0 : tr_data_X[max_point][time][pad] = digit_q[sector][globalrow][map_q_idx] / 100.f;
          }
        }
        check_assignment = 0;
        index_assignment = -1;
        distance_assignment = 100000.f;
        is_min_dist = true;
        for (int i = 0; i < 25; i++) {
          // Checks all ideal maxima assigned to one digit maximum by calculating mutual distance
          current_idx_id = assignments_id_to_dig[max_point][i];
          if (checkIdx(current_idx_id)) {
            if ((ideal_cog_q[sector][globalrow][current_idx_id] < 5 && ideal_max_q[sector][globalrow][current_idx_id] < 3) || (assigned_ideal[current_idx_id] != 0)) {
              is_min_dist = false;
              break;
            } else {
              current_distance_dig_to_id = std::pow((digit_map[sector][globalrow][maxima_digits[sector][globalrow][max_point]][2] - ideal_cog_map[sector][globalrow][current_idx_id][2]), 2) + std::pow((digit_map[sector][globalrow][maxima_digits[sector][globalrow][max_point]][1] - ideal_cog_map[sector][globalrow][current_idx_id][1]), 2);
              // if the distance is less than the previous one check if update should be made
              if (current_distance_dig_to_id < distance_assignment) {
                for (int j = 0; j < 25; j++) {
                  current_idx_dig = assignments_dig_to_id[current_idx_id][j];
                  if (checkIdx(current_idx_dig)) {
                    if (assigned_digit[current_idx_dig] == 0) {
                      // calculate mutual distance from current ideal CoG to all assigned digit maxima. Update if and only if distance is minimal. Else do not assign.
                      current_distance_id_to_dig = std::pow((digit_map[sector][globalrow][maxima_digits[sector][globalrow][current_idx_dig]][2] - ideal_cog_map[sector][globalrow][current_idx_id][2]), 2) + std::pow((digit_map[sector][globalrow][maxima_digits[sector][globalrow][current_idx_dig]][1] - ideal_cog_map[sector][globalrow][current_idx_id][1]), 2);
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
                if (is_min_dist && (std::pow((digit_map[sector][globalrow][maxima_digits[sector][globalrow][max_point]][2] - ideal_max_map[sector][globalrow][current_idx_id][2]), 2) + std::pow((digit_map[sector][globalrow][maxima_digits[sector][globalrow][max_point]][1] - ideal_max_map[sector][globalrow][current_idx_id][1]), 2) <= 5)) {
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

        tr_data_Y_class[max_point] = check_assignment;
        if (check_assignment > 0 && is_min_dist) {
          tr_data_Y_reg[0][max_point] = ideal_cog_map[sector][globalrow][index_assignment][2] - digit_map[sector][globalrow][maxima_digits[sector][globalrow][max_point]][2]; // time
          tr_data_Y_reg[1][max_point] = ideal_cog_map[sector][globalrow][index_assignment][1] - digit_map[sector][globalrow][maxima_digits[sector][globalrow][max_point]][1]; // pad
          tr_data_Y_reg[2][max_point] = ideal_sigma_map[sector][globalrow][index_assignment][0];                                                                              // sigma pad
          tr_data_Y_reg[3][max_point] = ideal_sigma_map[sector][globalrow][index_assignment][1];                                                                              // sigma time
          tr_data_Y_reg[4][max_point] = ideal_cog_q[sector][globalrow][index_assignment] / q_max;
          // if(std::abs(digit_map[loop_sectors][maxima_digits[max_point]][2] - ideal_cog_map[loop_sectors][index_assignment][2]) > 3 || std::abs(digit_map[loop_sectors][maxima_digits[max_point]][1] - ideal_cog_map[loop_sectors][index_assignment][1]) > 3){
          //   LOG(info) << "#Maxima: " << maxima_digits.size() << ", Index (point) " << max_point << " & (max) " << maxima_digits[max_point] << " & (ideal) " << index_assignment << ", ideal_cog_map[loop_sectors].size(): " <<  ideal_cog_map[loop_sectors].size() << ", index_assignment: " << index_assignment;
          // }
        } else {
          tr_data_Y_reg[0][max_point] = -1.f;
          tr_data_Y_reg[1][max_point] = -1.f;
          tr_data_Y_reg[2][max_point] = -1.f;
          tr_data_Y_reg[3][max_point] = -1.f;
          tr_data_Y_reg[4][max_point] = -1.f;
        }
      } else {
        if (current_idx_id > 20000000) {
          LOG(warning) << "Element at index [" << digit_map[sector][globalrow][maxima_digits[sector][globalrow][max_point]][2] << " " << digit_map[sector][globalrow][maxima_digits[sector][globalrow][max_point]][0] << " " << digit_map[sector][globalrow][maxima_digits[sector][globalrow][max_point]][1] << "] has value " << map_dig_idx;
        }
      }
    }
    if (verbose >= 3)
      LOG(info) << "Done creating training data. Writing to file.";

    std::stringstream file_in;
    file_in << "training_data_" << sector << "_" << globalrow << ".root";
    TFile* outputFileTrData = new TFile(file_in.str().c_str(), "RECREATE");
    TTree* tr_data = new TTree("tr_data", "tree");

    // Defining the branches
    for (int time = 0; time < mat_size_time; time++) {
      for (int pad = 0; pad < mat_size_pad; pad++) {
        std::stringstream branch_name;
        branch_name << "in_time_" << time << "_pad_" << pad;
        tr_data->Branch(branch_name.str().c_str(), &atomic_unit[mat_size_pad - pad - 1][time]); // Switching pad and time here makes the transformatio in python easier
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
      idx_sector = sector;
      idx_row = digit_map[sector][globalrow][maxima_digits[sector][globalrow][element]][0];
      idx_pad = digit_map[sector][globalrow][maxima_digits[sector][globalrow][element]][1];
      idx_time = digit_map[sector][globalrow][maxima_digits[sector][globalrow][element]][2];
      tr_data->Fill();
    }
    tr_data->Write();
    outputFileTrData->Close();
  }
}

// ---------------------------------
template <class T>
void qaIdeal::runQa(int sector, std::vector<int> rows)
{
  LOG(info) << "Running QA for sector " << sector << ", rows " << rows[0] << " - " << rows[1];
  T map2d = init_map2d<T>(max_time[sector], rows);
  LOG(info) << "Initialized 2D charge map";
  fill_map2d<T>(sector, rows, map2d, -1, 1);
  LOG(info) << "Filled charge map";
  for (int row = rows[0]; row < rows[1]; row++) {
    runQaSingle(sector, row, row - rows[0], map2d);
    LOG(info) << "Cleaning up...";
    clear_memory(sector, rows[0] + row, row);
  }
}

// ---------------------------------
void qaIdeal::run(ProcessingContext& pc)
{

  const int rows_size = 10;
  typedef std::array<std::vector<std::array<std::array<int, 170>, rows_size>>, 2> qa_t;

  // init array
  for (int i = 0; i < 25; i++) {
    assignments_ideal[i] = 0;
    assignments_digit[i] = 0;
    assignments_ideal_findable[i] = 0;
    assignments_digit_findable[i] = 0;
  }

  // int current_max_dig_counter=0, current_max_id_counter=0;
  for (int loop_sectors = 0; loop_sectors < o2::tpc::constants::MAXSECTOR; loop_sectors++) {
    if (mode.find(std::string("native")) != std::string::npos) {
      read_native(loop_sectors);
    } else if (mode.find(std::string("network")) != std::string::npos) {
      read_network(loop_sectors);
    } else {
      read_digits(loop_sectors);
    }
    read_ideal(loop_sectors);

    if (numThreads > 1) {
      thread_group group;
      for (int loop_rows = 0; loop_rows < 16; loop_rows++) {
        group.create_thread(boost::bind(&qaIdeal::runQa<qa_t>, this, loop_sectors, std::vector<int>{loop_rows * rows_size, (loop_rows + 1) * rows_size}));
      }
      group.join_all();
    } else {
      runQa<qa_t>(loop_sectors, std::vector<int>{0, o2::tpc::constants::MAXGLOBALPADROW});
    }

    std::stringstream merge, remove;
    merge << "hadd -k -f ./training_data.root training_data_" << loop_sectors << "_*.root";
    remove << "rm -rf training_data_" << loop_sectors << "_*.root";
    gSystem->Exec(merge.str().c_str());
    gSystem->Exec(remove.str().c_str());
  }

  LOG(info) << "------- RESULTS -------\n";
  LOG(info) << "Number of digit maxima: " << number_of_digit_max;
  LOG(info) << "Number of ideal maxima (total): " << number_of_ideal_max;
  LOG(info) << "Number of ideal maxima (findable): " << number_of_ideal_max_findable << "\n";

  unsigned int efficiency_normal = 0;
  unsigned int efficiency_findable = 0;
  for (int ass = 0; ass < 10; ass++) {
    LOG(info) << "Number of assigned digit maxima (#assignments " << ass << "): " << assignments_digit[ass];
    LOG(info) << "Number of assigned ideal maxima (#assignments " << ass << "): " << assignments_ideal[ass] << "\n";
    if (ass > 0) {
      efficiency_normal += assignments_ideal[ass];
    }
  }

  assignments_ideal_findable[0] -= (number_of_ideal_max - number_of_ideal_max_findable);
  for (int ass = 0; ass < 10; ass++) {
    LOG(info) << "Number of finable assigned digit maxima (#assignments " << ass << "): " << assignments_digit_findable[ass];
    LOG(info) << "Number of finable assigned ideal maxima (#assignments " << ass << "): " << assignments_ideal_findable[ass] << "\n";
    if (ass > 0) {
      efficiency_findable += assignments_ideal_findable[ass];
    }
  }

  LOG(info) << "Efficiency - Number of assigned (ideal -> digit) clusters: " << efficiency_normal << " (" << (float)efficiency_normal * 100 / (float)number_of_ideal_max << "% of ideal maxima)";
  LOG(info) << "Efficiency (findable) - Number of assigned (ideal -> digit) clusters: " << efficiency_findable << " (" << (float)efficiency_findable * 100 / (float)number_of_ideal_max_findable << "% of ideal maxima)";
  LOG(info) << "Clones (Int, clone-order >= 2 for ideal cluster): " << clones << " (" << (float)clones * 100 / (float)number_of_digit_max << "% of digit maxima)";
  LOG(info) << "Clones (Float, fractional clone-order): " << fractional_clones << " (" << (float)fractional_clones * 100 / (float)number_of_digit_max << "% of digit maxima)";
  LOG(info) << "Fakes (number of digit hits that can't be assigned to any ideal hit): " << assignments_digit[0] << " (" << (float)assignments_digit[0] * 100 / (float)number_of_digit_max << "% of digit maxima)";

  if (mode.find(std::string("training_data")) != std::string::npos) {
    LOG(info) << "------- Merging training data -------";
    gSystem->Exec("hadd -k -f ./training_data.root ./training_data_*.root");
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
      {"mode", VariantType::String, "training_data", {"Enables different settings (e.g. creation of training data for NN, running with tpc-native clusters)."}},
      {"threads", VariantType::Int, 1, {"Number of CPU cores to be used."}},
      {"infile-digits", VariantType::String, "tpcdigits.root", {"Input file name (digits)"}},
      {"infile-native", VariantType::String, "tpc-native-clusters.root", {"Input file name (native)"}},
      {"network-data-output", VariantType::String, "network_out.root", {"Input file for the network output"}},
      {"network-classification-path", VariantType::String, "./net_classification.onnx", {"Absolute path to the network file (classification)"}},
      {"network-regression-path", VariantType::String, "./net_regression.onnx", {"Absolute path to the network file (regression)"}},
      {"network-input-size", VariantType::Int, 100, {"Size of the vector to be fed through the neural network"}},
      {"enable-network-optimizations", VariantType::Bool, true, {"Enable ONNX network optimizations"}},
      {"network-num-threads", VariantType::Int, 10, {"Set the number of CPU threads for network execution"}}}};
}

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{processIdealClusterizer()};
}