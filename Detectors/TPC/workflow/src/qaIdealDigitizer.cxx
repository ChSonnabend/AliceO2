#include <cmath>
#include <boost/thread.hpp>
#include <stdlib.h>
#include <unordered_map>
#include <regex>
#include <chrono>
#include <thread>

#include "Algorithm/RangeTokenizer.h"

#include "DetectorsRaw/HBFUtils.h"

#include "DataFormatsTPC/WorkflowHelper.h"
#include "DataFormatsTPC/ClusterNativeHelper.h"
#include "DataFormatsTPC/ClusterNative.h"
#include "DataFormatsTPC/ClusterGroupAttribute.h"
#include "DataFormatsTPC/Constants.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "DataFormatsGlobalTracking/TrackTuneParams.h"

#include "DPLUtils/RootTreeReader.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"

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

template <typename T>
using BranchDefinition = o2::framework::MakeRootTreeWriterSpec::BranchDefinition<T>;

class qaIdeal : public Task
{
 public:
  template <typename T>
  int sign(T);

  qaIdeal(std::unordered_map<std::string, std::string>);
  void init(InitContext&) final;
  void setGeomFromTxt(std::string, std::vector<int> = {152, 14});
  int padOffset(int);
  int rowOffset(int);
  bool isBoundary(int, int);
  bool checkIdx(int);
  void read_digits(int, std::vector<std::array<int, 3>>&, std::vector<float>&);
  void read_ideal(int, std::vector<std::array<int, 3>>&, std::vector<float>&, std::vector<std::array<float, 3>>&, std::vector<std::array<float, 2>>&, std::vector<float>&, std::vector<std::array<int, 3>>&);
  void read_native(int, std::vector<std::array<int, 3>>&, std::vector<std::array<float, 7>>&, std::vector<float>&);
  void write_custom_native(ProcessingContext&, std::vector<std::array<float, 11>>&);
  void read_kinematics(std::vector<std::vector<std::vector<o2::MCTrack>>>&);
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

  template <class T, class C>
  std::vector<std::vector<std::vector<int>>> looper_tagger(int, int, T&, C&, std::vector<float>&, std::vector<int>&, std::vector<std::array<int, 3>>&, std::string = "digits", int = 0);

  template <class T>
  std::vector<std::vector<std::vector<int>>> looper_tagger_full(int, int, T&, std::vector<int>&, std::vector<std::array<int, 3>>&, std::vector<std::array<float, 2>>&);

  template <class T>
  int tagLabel(T&, std::vector<std::vector<std::vector<int>>>&);

  template <class T>
  void remove_loopers_digits(int, int, std::vector<std::vector<std::vector<int>>>&, T&, std::vector<int>&);

  template <class T>
  void remove_loopers_native(int, int, std::vector<std::vector<std::vector<int>>>&, T&, std::vector<int>&);

  void remove_loopers_ideal(int, int, std::vector<std::vector<std::vector<int>>>&, std::vector<std::array<int, 3>>&, std::vector<std::array<float, 3>>&, std::vector<float>&, std::vector<float>&, std::vector<std::array<float, 2>>&, std::vector<std::array<int, 3>>&);

  template <class T>
  void run_network(int, T&, std::vector<int>&, std::vector<std::array<int, 3>>&, std::vector<float>&, std::vector<std::array<float, 7>>&, int = 0);

  template <class T>
  void overwrite_map2d(int, T&, std::vector<std::array<int, 3>>&, std::vector<int>&, int = 0);

  template <class T>
  int test_neighbour(std::array<int, 3>, std::array<int, 2>, T&, int = 1);

  void runQa(int);
  void run(ProcessingContext&) final;

 private:
  std::vector<int> global_shift = {5, 5, 0};             // shifting digits to select windows easier, (pad, time, row)
  int charge_limits[2] = {2, 1024};                      // upper and lower charge limits
  int verbose = 0;                                       // chunk_size in time direction
  int create_output = 1;                                 // Create output files specific for any mode
  int dim = 2;                                           // Dimensionality of the training data
  int networkInputSize = 1000;                           // vector input size for neural network
  float networkClassThres = 0.5f;                        // Threshold where network decides to keep / reject digit maximum
  bool networkOptimizations = true;                      // ONNX session optimizations
  int networkNumThreads = 1;                             // Future: Add Cuda and CoreML Execution providers to run on CPU
  int numThreads = 1;                                    // Number of cores for multithreading
  int use_max_cog = 1;                                   // 0 = use ideal maxima position; 1 = use ideal CoG position (rounded) for assignment
  float threshold_cogq = 5.f;                            // Threshold for ideal cluster to be findable (Q_tot)
  float threshold_maxq = 3.f;                            // Threshold for ideal cluster to be findable (Q_max)
  int normalization_mode = 1;                            // Normalization of the charge: 0 = divide by 1024; 1 = divide by central charge
  int remove_individual_files = 0;                       // Remove sector-individual files after task is done
  bool write_native_file = 1;                            // Whether or not to write a custom file with native clsuters
  bool native_file_single_branch = 1;                    // Whether the native clusters should be written into a single branch
  std::vector<int> looper_tagger_granularity = {5};      // Granularity of looper tagger (time bins in which loopers are excluded in rectangular areas)
  std::vector<int> looper_tagger_timewindow = {20};      // Total time-window size of the looper tagger for evaluating if a region is looper or not
  std::vector<int> looper_tagger_padwindow = {3};        // Total pad-window size of the looper tagger for evaluating if a region is looper or not
  std::vector<int> looper_tagger_threshold_num = {5};    // Threshold of number of clusters over which rejection takes place
  std::vector<float> looper_tagger_threshold_q = {70.f}; // Threshold of charge-per-cluster that should be rejected
  std::string looper_tagger_opmode = "digit";            // Operational mode of the looper tagger
  std::vector<int> tpcSectors;                           // The TPC sectors for which processing should be started

  std::array<int, o2::tpc::constants::MAXSECTOR> max_time, max_pad;
  std::string mode = "training_data";
  std::string inFileDigits = "tpcdigits.root";
  std::string inFileNative = "tpc-cluster-native.root";
  std::string inFileDigitizer = "mclabels_digitizer.root";
  std::string inFileKinematics = "collisioncontext.root";
  std::string networkDataOutput = "./network_out.root";
  std::string networkClassification = "./net_classification.onnx";
  std::string networkRegression = "./net_regression.onnx";
  std::string outCustomNative = "tpc-cluster-native-custom.root";

  std::vector<std::vector<std::array<int, 2>>> adj_mat = {{{0, 0}}, {{1, 0}, {0, 1}, {-1, 0}, {0, -1}}, {{1, 1}, {-1, 1}, {-1, -1}, {1, -1}}, {{2, 0}, {0, -2}, {-2, 0}, {0, 2}}, {{2, 1}, {1, 2}, {-1, 2}, {-2, 1}, {-2, -1}, {-1, -2}, {1, -2}, {2, -1}}, {{2, 2}, {-2, 2}, {-2, -2}, {2, -2}}};
  std::vector<std::vector<float>> TPC_GEOM;
  OnnxModel network_classification, network_regression;

  int num_ideal_max = 0, num_digit_max = 0, num_ideal_max_findable = 0;
  std::vector<std::vector<std::vector<o2::MCTrack>>> mctracks; // mc_track = mctracks[sourceId][eventId][trackId]
  std::array<std::array<unsigned int, 25>, o2::tpc::constants::MAXSECTOR> assignments_ideal, assignments_digit, assignments_ideal_findable, assignments_digit_findable;
  std::array<unsigned int, o2::tpc::constants::MAXSECTOR> number_of_ideal_max, number_of_digit_max, number_of_ideal_max_findable;
  std::array<float, o2::tpc::constants::MAXSECTOR> clones, fractional_clones;
  std::vector<std::array<float, 11>> native_writer_map;
  std::mutex m;

  float landau_approx(float x)
  {
    return (1.f / TMath::Sqrt(2.f * (float)TMath::Pi())) * TMath::Exp(-(x + TMath::Exp(-x)) / 2.f) + 0.005; // +0.005 for regularization
  }

  struct ArrayHasher {
    std::size_t operator()(const std::array<int, 3>& a) const
    {
      std::size_t h = 0;

      for (auto e : a) {
        h ^= std::hash<int>{}(e) + 0x9e3779b9 + (h << 6) + (h >> 2);
      }
      return h;
    }
  };

  bool hasElementAppearedMoreThanNTimesInVectors(const std::vector<std::vector<std::array<int, 3>>>& vectors, int n)
  {
    // std::unordered_map<std::array<int, 3>, int, ArrayHasher> elementCount;
    std::unordered_map<int, int> elementCount;
    for (const std::vector<std::array<int, 3>>& vec : vectors) {
      for (std::array<int, 3> element : vec) {
        elementCount[element[0]]++;
        if (elementCount[element[0]] >= n) {
          return true;
        }
      }
    }
    return false;
  }

  std::vector<int> hasElementAppearedMoreThanNTimesInVectors(const std::vector<std::vector<int>>& index_vector, const std::vector<std::array<int, 3>>& assignment_vector, int n)
  {
    std::unordered_map<int, int> elementCount;
    std::vector<int> return_labels;
    int vector_resize;
    for (const std::vector<int>& vec : index_vector) {
      for (int element : vec) {
        elementCount[assignment_vector[element][0]]++;
      }
    }
    for (auto elem : elementCount) {
      if (elem.second >= n) {
        return_labels.push_back(elem.first);
      }
    }

    return return_labels;
  }

  std::vector<int> elementsAppearedMoreThanNTimesInVectors(const std::vector<std::vector<int>>& index_vector, const std::vector<std::array<int, 3>>& assignment_vector, int n)
  {
    std::unordered_map<int, std::vector<int>> elementCount;
    std::vector<int> return_idx;
    for (const std::vector<int>& vec : index_vector) {
      for (int element : vec) {
        elementCount[assignment_vector[element][0]].push_back(element);
      }
    }
    for (auto elem : elementCount) {
      if (elem.second.size() >= n) {
        for (int idx : elem.second) {
          return_idx.push_back(idx);
        }
      }
    }
    return return_idx;
  }

  std::unordered_map<int, std::vector<int>> distinctElementAppearance(const std::vector<std::vector<int>>& index_vector, const std::vector<std::array<int, 3>>& assignment_vector)
  {
    std::unordered_map<int, std::vector<int>> elementCount;
    for (const std::vector<int>& vec : index_vector) {
      for (int element : vec) {
        elementCount[assignment_vector[element][0]].push_back(element);
      }
    }
    return elementCount;
  }

  std::vector<int> fastElementBuckets(const std::vector<std::vector<int>>& index_vector, const std::vector<std::array<int, 3>>& assignment_vector, int n)
  {

    std::vector<int> exclude_elements;
    int max_track_label = -1, min_track_label = 1e9;

    for (auto vec : index_vector) {
      for (int elem : vec) {
        if (assignment_vector[elem][0] > max_track_label) {
          max_track_label = assignment_vector[elem][0];
        }
        if (assignment_vector[elem][0] < min_track_label) {
          min_track_label = assignment_vector[elem][0];
        }
      }
    }

    std::vector<std::vector<int>> buckets(max_track_label - min_track_label + 1);
    std::vector<int> exclude_elements_size(max_track_label - min_track_label + 1);
    for (auto vec : index_vector) {
      for (int elem : vec) {
        exclude_elements_size[assignment_vector[elem][0] - min_track_label]++;
      }
    }
    for (int i = 0; i < exclude_elements_size.size(); i++) {
      buckets[i].resize(exclude_elements_size[i]);
    }

    std::fill(exclude_elements_size.begin(), exclude_elements_size.end(), 0);
    int index = 0;
    for (auto vec : index_vector) {
      for (int elem : vec) {
        index = assignment_vector[elem][0] - min_track_label;
        buckets[index][exclude_elements_size[index]] = elem;
        exclude_elements_size[index]++;
      }
    }

    int full_exclude_size = 0;
    for (auto bucket : buckets) {
      if (bucket.size() >= n) {
        full_exclude_size += bucket.size();
      }
    }

    exclude_elements.resize(full_exclude_size);
    full_exclude_size = 0;
    for (auto bucket : buckets) {
      if (bucket.size() >= n) {
        for (auto elem : bucket) {
          exclude_elements[full_exclude_size] = elem;
          full_exclude_size++;
        }
      }
    }

    return exclude_elements;
  }

  // Filling a nested vector / array structure
  // Helper template struct for checking if a type is a std::vector
  template <typename T>
  struct is_vector : std::false_type {
  };

  template <typename T, typename Alloc>
  struct is_vector<std::vector<T, Alloc>> : std::true_type {
  };

  // Helper template struct for checking if a type is a std::array
  template <typename T>
  struct is_array : std::false_type {
  };

  template <typename T, std::size_t N>
  struct is_array<std::array<T, N>> : std::true_type {
  };

  // Helper template struct for checking if a type is a container (vector or array)
  template <typename T>
  struct is_container : std::conditional_t<is_vector<T>::value || is_array<T>::value, std::true_type, std::false_type> {
  };

  // Helper function for filling a nested container with a specified value and size for the innermost vector
  template <typename Container, typename ValueType>
  void fill_nested_container(Container& container, const ValueType& value, std::size_t innermostSize = 0)
  {
    if (innermostSize > 0) {
      if constexpr (is_vector<Container>::value) {
        if constexpr (std::is_same_v<typename Container::value_type, ValueType>) {
          container.resize(innermostSize, value);
        }
      }
    } else if constexpr (is_container<Container>::value) {
      for (auto& elem : container) {
        fill_nested_container(elem, value, innermostSize);
      }
    } else {
      container = value;
    }
  };

  template <typename Container, typename ValueType>
  void sum_nested_container(Container& container, ValueType& value)
  {
    if constexpr (is_container<Container>::value) {
      for (auto& elem : container) {
        sum_nested_container(elem, value);
      }
    } else {
      value += container;
    }
  };

  template <typename Container, typename T>
  bool all_elements_equal_to_value(const Container& container, const T& value)
  {
    return std::all_of(container.begin(), container.end(), [&](const auto& element) {
      return element == value;
    });
  }

  template <typename T>
  std::vector<int> sort_indices(const T& arr)
  {
    int length = arr.size();

    // returns the index array with the sorted indices of the array
    if (length > 1) {
      std::vector<int> idx(length);
      std::iota(idx.begin(), idx.end(), 0);
      std::stable_sort(idx.begin(), idx.end(), [&arr](int i1, int i2) { return arr[i1] < arr[i2]; });
      return idx;
    } else {
      return {0};
    }
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
qaIdeal::qaIdeal(std::unordered_map<std::string, std::string> options_map){
  write_native_file = (bool)std::stoi(options_map["write-native-file"]);
  native_file_single_branch = (bool)std::stoi(options_map["native-file-single-branch"]);
  tpcSectors = o2::RangeTokenizer::tokenize<int>(options_map["tpc-sectors"]);
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
  inFileKinematics = ic.options().get<std::string>("infile-kinematics");
  networkDataOutput = ic.options().get<std::string>("network-data-output");
  networkClassification = ic.options().get<std::string>("network-classification-path");
  networkRegression = ic.options().get<std::string>("network-regression-path");
  networkInputSize = ic.options().get<int>("network-input-size");
  networkClassThres = ic.options().get<float>("network-class-threshold");
  networkOptimizations = ic.options().get<bool>("enable-network-optimizations");
  networkNumThreads = ic.options().get<int>("network-num-threads");
  normalization_mode = ic.options().get<int>("normalization-mode");
  looper_tagger_granularity = ic.options().get<std::vector<int>>("looper-tagger-granularity");
  looper_tagger_timewindow = ic.options().get<std::vector<int>>("looper-tagger-timewindow");
  looper_tagger_padwindow = ic.options().get<std::vector<int>>("looper-tagger-padwindow");
  looper_tagger_threshold_num = ic.options().get<std::vector<int>>("looper-tagger-threshold-num");
  looper_tagger_threshold_q = ic.options().get<std::vector<float>>("looper-tagger-threshold-q");
  looper_tagger_opmode = ic.options().get<std::string>("looper-tagger-opmode");
  remove_individual_files = ic.options().get<int>("remove-individual-files");

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
    LOG(info) << "[" << sector << "] Reading the digits...";

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

  digitFile->Close();
}

// ---------------------------------
void qaIdeal::read_native(int sector, std::vector<std::array<int, 3>>& digit_map, std::vector<std::array<float, 7>>& native_map, std::vector<float>& digit_q)
{

  if (verbose >= 1)
    LOG(info) << "[" << sector << "] Reading native clusters...";

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
        native_map[count_clusters][3] = cl.getSigmaPad();
        native_map[count_clusters][4] = cl.getSigmaTime();
        native_map[count_clusters][5] = cl.getQtot();
        native_map[count_clusters][6] = cl.getQmax();

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
}

// ---------------------------------
void qaIdeal::write_custom_native(ProcessingContext& pc, std::vector<std::array<float, 11>>& assigned_clusters)
{

  // Build cluster native access structure
  ClusterNativeAccess clusterIndex;
  ClusterNative cluster_native_array[assigned_clusters.size()];
  o2::dataformats::MCTruthContainer<o2::MCCompLabel> mcLabelContainer;
  std::memset(clusterIndex.nClusters, 0, sizeof(clusterIndex.nClusters));
  int total_clusters = 0;

  const size_t BIGSIZE{assigned_clusters.size()};
  mcLabelContainer.addNoLabelIndex(0); // the first index does not have a label

  for (auto cls : assigned_clusters) {
    // storing cluster natives
    cluster_native_array[total_clusters].setTime(cls[3]);
    cluster_native_array[total_clusters].setPad(cls[2]);
    cluster_native_array[total_clusters].setSigmaTime(cls[5]);
    cluster_native_array[total_clusters].setSigmaPad(cls[4]);
    cluster_native_array[total_clusters].qMax = cls[6];
    cluster_native_array[total_clusters].qTot = cls[7];

    // creating ConstMCTruthContainer
    mcLabelContainer.addElement(total_clusters, o2::MCCompLabel(cls[8], cls[9], cls[10], false));

    clusterIndex.nClusters[(int)cls[0]][(int)cls[1]]++;
    total_clusters++;
  }
  clusterIndex.clustersLinear = cluster_native_array;

  // Some reshuffeling to accomodate for MCTruthContainer structure
  std::vector<char> buffer;
  mcLabelContainer.flatten_to(buffer);
  o2::dataformats::IOMCTruthContainerView tmp_container(buffer);
  o2::dataformats::ConstMCTruthContainer<o2::MCCompLabel> constMcLabelContainer;
  tmp_container.copyandflatten(constMcLabelContainer);
  o2::dataformats::ConstMCTruthContainerView containerView(constMcLabelContainer);
  clusterIndex.clustersMCTruth = &containerView;

  int arr_counter = 0, counter = 0, cluster_counter = 0;
  for (int sec = 0; sec < o2::tpc::constants::MAXSECTOR; sec++) {
    for (int row = 0; row < o2::tpc::constants::MAXGLOBALPADROW; row++) {
      ClusterNative tmp_clus_arr[clusterIndex.nClusters[sec][row]];
      for (auto cls : assigned_clusters) {
        if ((cls[0] == sec) && (cls[1] == row)) {
          tmp_clus_arr[counter] = cluster_native_array[cluster_counter];
          counter++;
        }
        cluster_counter++;
      }
      clusterIndex.clusters[sec][row] = tmp_clus_arr;
      counter = 0;
      cluster_counter = 0;
    }
  }
  clusterIndex.setOffsetPtrs();

  // Clusters are shipped by sector, we are copying into per-sector buffers (anyway only for ROOT output)
  o2::tpc::TPCSectorHeader clusterOutputSectorHeader{0};
  for (unsigned int i = 0; i < o2::tpc::constants::MAXSECTOR; i++) {
    unsigned int subspec = i;
    clusterOutputSectorHeader.sectorBits = (1ul << i);
    char* buffer = pc.outputs().make<char>({o2::header::gDataOriginTPC, "CLUSTERNATIVE", subspec, {clusterOutputSectorHeader}}, clusterIndex.nClustersSector[i] * sizeof(*clusterIndex.clustersLinear) + sizeof(o2::tpc::ClusterCountIndex)).data();
    o2::tpc::ClusterCountIndex* outIndex = reinterpret_cast<o2::tpc::ClusterCountIndex*>(buffer);
    memset(outIndex, 0, sizeof(*outIndex));
    for (int j = 0; j < o2::tpc::constants::MAXGLOBALPADROW; j++) {
      outIndex->nClusters[i][j] = clusterIndex.nClusters[i][j];
    }
    memcpy(buffer + sizeof(*outIndex), clusterIndex.clusters[i][0], clusterIndex.nClustersSector[i] * sizeof(*clusterIndex.clustersLinear));
  
    o2::dataformats::MCLabelContainer cont;
    for (unsigned int j = 0; j < clusterIndex.nClustersSector[i]; j++) {
      const auto& labels = clusterIndex.clustersMCTruth->getLabels(clusterIndex.clusterOffset[i][0] + j);
      for (const auto& label : labels) {
        cont.addElement(j, label);
      }
    }
    o2::dataformats::ConstMCLabelContainer contflat;
    cont.flatten_to(contflat);
    pc.outputs().snapshot({o2::header::gDataOriginTPC, "CLNATIVEMCLBL", subspec, {clusterOutputSectorHeader}}, contflat);
  }

  LOG(info) << "------- Native clusters structure created -------";
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
}

// ---------------------------------
void qaIdeal::read_ideal(int sector, std::vector<std::array<int, 3>>& ideal_max_map, std::vector<float>& ideal_max_q, std::vector<std::array<float, 3>>& ideal_cog_map, std::vector<std::array<float, 2>>& ideal_sigma_map, std::vector<float>& ideal_cog_q, std::vector<std::array<int, 3>>& ideal_mclabel)
{

  int sec, row, maxp, maxt, pcount, trkid, evid, srcid;
  float cogp, cogt, cogq, maxq, sigmap, sigmat;
  int elements = 0;

  if (verbose > 0)
    LOG(info) << "[" << sector << "] Reading ideal clusters";
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
  digitizerSector->SetBranchAddress("cluster_trackid", &trkid);
  digitizerSector->SetBranchAddress("cluster_eventid", &evid);
  digitizerSector->SetBranchAddress("cluster_sourceid", &srcid);
  // digitizerSector->SetBranchAddress("cluster_points", &pcount);

  ideal_max_map.resize(digitizerSector->GetEntries());
  ideal_max_q.resize(digitizerSector->GetEntries());
  ideal_cog_map.resize(digitizerSector->GetEntries());
  ideal_sigma_map.resize(digitizerSector->GetEntries());
  ideal_cog_q.resize(digitizerSector->GetEntries());
  ideal_mclabel.resize(digitizerSector->GetEntries());

  if (verbose >= 3)
    LOG(info) << "[" << sector << "] Trying to read " << digitizerSector->GetEntries() << " ideal digits";
  for (unsigned int j = 0; j < digitizerSector->GetEntries(); j++) {
    try {
      digitizerSector->GetEntry(j);
      // ideal_point_count.push_back(pcount);

      ideal_max_map[j] = std::array<int, 3>{row, maxp, maxt};
      ideal_max_q[j] = maxq;
      ideal_cog_map[j] = std::array<float, 3>{(float)row, cogp, cogt};
      ideal_sigma_map[j] = std::array<float, 2>{sigmap, sigmat};
      ideal_cog_q[j] = cogq;
      ideal_mclabel[j] = {trkid, evid, srcid};
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
      LOG(info) << "[" << sector << "] (Digitizer) Problem occured in sector " << sector;
    }
  }
  inputFile->Close();
}

// ---------------------------------
void qaIdeal::read_kinematics(std::vector<std::vector<std::vector<o2::MCTrack>>>& tracks)
{

  LOG(info) << "Reading kinematics information.";

  o2::steer::MCKinematicsReader reader("collisioncontext.root");

  tracks.resize(reader.getNSources());
  for (int src = 0; src < reader.getNSources(); src++) {
    tracks[src].resize(reader.getNEvents(src));
    for (int ev = 0; ev < reader.getNEvents(src); ev++) {
      tracks[src][ev] = reader.getTracks(src, ev);
    }
  }

  // LOG(info) << "Done reading kinematics, exporting to file (for python readout)";
}

template <class T>
T qaIdeal::init_map2d(int sector)
{
  T map2d;
  for (int i = 0; i < 2; i++) {
    map2d[i].resize(max_time[sector] + (2 * global_shift[1]));
    for (int time_size = 0; time_size < (max_time[sector] + 2 * global_shift[1]); time_size++) {
      map2d[i][time_size].resize(o2::tpc::constants::MAXGLOBALPADROW + (3 * global_shift[2]));
      for (int row = 0; row < (o2::tpc::constants::MAXGLOBALPADROW + 3 * global_shift[2]); row++) {
        // Support for IROC - OROC1 transition: Add rows after row 62 + global_shift[2]
        map2d[i][time_size][row].resize(TPC_GEOM[o2::tpc::constants::MAXGLOBALPADROW - 1][2] + 1 + 2 * global_shift[0]);
        for (int pad = 0; pad < (TPC_GEOM[o2::tpc::constants::MAXGLOBALPADROW - 1][2] + 1 + 2 * global_shift[0]); pad++) {
          map2d[i][time_size][row][pad] = -1;
        }
      }
    };
  };

  if (verbose >= 1)
    LOG(info) << "[" << sector << "] Initialized 2D map! Time size is " << map2d[0].size();

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
    LOG(info) << "[" << sector << "] Finding local maxima";
  }

  bool is_max = true;
  float current_charge = 0;
  int row_offset = 0, pad_offset = 0;
  for (int row = 0; row < o2::tpc::constants::MAXGLOBALPADROW; row++) {
    for (int pad = 0; pad < TPC_GEOM[row][2] + 1; pad++) {
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
  }

  if (verbose >= 1)
    LOG(info) << "[" << sector << "] Found " << maxima_digits.size() << " maxima. Done!";
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
template <class T, class C>
std::vector<std::vector<std::vector<int>>> qaIdeal::looper_tagger(int sector, int counter, T& index_map, C& sigma_map, std::vector<float>& array_q, std::vector<int>& index_array, std::vector<std::array<int, 3>>& ideal_mclabels, std::string op_mode, int exclusion_zones_counter)
{
  // looper_tagger_granularity[counter] = 1; // to be removed later: Testing for now
  int looper_detector_timesize = std::ceil((float)max_time[sector] / (float)looper_tagger_granularity[counter]);

  std::vector<std::vector<std::vector<float>>> tagger(looper_detector_timesize, std::vector<std::vector<float>>(o2::tpc::constants::MAXGLOBALPADROW)); // time_slice (=std::floor(time/looper_tagger_granularity[counter])), row, pad array -> looper_tagged = 1, else 0
  std::vector<std::vector<std::vector<int>>> tagger_counter(looper_detector_timesize, std::vector<std::vector<int>>(o2::tpc::constants::MAXGLOBALPADROW));
  std::vector<std::vector<std::vector<int>>> looper_tagged_region(max_time[sector] + 1, std::vector<std::vector<int>>(o2::tpc::constants::MAXGLOBALPADROW)); // accumulates all the regions that should be tagged: looper_tagged_region[time_slice][row] = (pad_low, pad_high)
  std::vector<std::vector<std::vector<std::vector<int>>>> sorted_idx(looper_detector_timesize, std::vector<std::vector<std::vector<int>>>(o2::tpc::constants::MAXGLOBALPADROW));

  int operation_mode = 0;
  op_mode.find(std::string("digit")) != std::string::npos ? operation_mode = 1 : operation_mode = operation_mode;
  op_mode.find(std::string("ideal")) != std::string::npos ? operation_mode = 2 : operation_mode = operation_mode;

  for (int t = 0; t < looper_detector_timesize; t++) {
    for (int r = 0; r < o2::tpc::constants::MAXGLOBALPADROW; r++) {
      tagger[t][r].resize(TPC_GEOM[r][2] + looper_tagger_padwindow[counter]);
      tagger_counter[t][r].resize(TPC_GEOM[r][2] + looper_tagger_padwindow[counter]);
      for (int full_time = 0; full_time < looper_tagger_granularity[counter]; full_time++) {
        if ((t * looper_tagger_granularity[counter]) + full_time <= max_time[sector]) {
          looper_tagged_region[(t * looper_tagger_granularity[counter]) + full_time][r].resize(TPC_GEOM[r][2] + 1);
          std::fill(looper_tagged_region[(t * looper_tagger_granularity[counter]) + full_time][r].begin(), looper_tagged_region[(t * looper_tagger_granularity[counter]) + full_time][r].end(), -1);
        }
      }
      // indv_charges.resize(TPC_GEOM[r][2] + 1);
      if (operation_mode == 2) {
        sorted_idx[t][r].resize(TPC_GEOM[r][2] + looper_tagger_padwindow[counter]);
      }
    }
  }

  // Improvements:
  // - Check the charge sigma -> Looper should have narrow sigma in charge
  // - Check width between clusters -> Looper should have regular distance -> peak in distribution of distance
  // - Check for gaussian distribution of charge: D'Agostino-Pearson

  int row = 0, pad = 0, time_slice = 0, pad_offset = (looper_tagger_padwindow[counter] - 1) / 2;
  for (auto idx : index_array) {
    row = std::round(index_map[idx][0]);
    pad = std::round(index_map[idx][1]) + pad_offset;
    time_slice = std::floor(index_map[idx][2] / (float)looper_tagger_granularity[counter]);

    tagger[time_slice][row][pad]++;
    tagger_counter[time_slice][row][pad]++;
    tagger[time_slice][row][pad] += array_q[idx]; // / landau_approx((array_q[idx] - 25.f) / 17.f);
    // indv_charges[time_slice][row][pad].push_back(array_q[idx]);

    if (operation_mode == 2) {
      sorted_idx[time_slice][row][pad].push_back(idx);
    }

    // Approximate Landau and scale for the width:
    // Lindhards theory: L(x, mu=0, c=pi/2) ~ exp(-1/x)/(x*(x+1))
    // Estimation of peak-charge: Scale by landau((charge-25)/17)
  }

  if (verbose > 2)
    LOG(info) << "[" << sector << "] Tagger done. Building tagged regions.";

  // int unit_volume = looper_tagger_timewindow[counter] * 3;
  float avg_charge = 0;
  int num_elements = 0, sigma_pad = 0, sigma_time = 0, ideal_pad = 0, ideal_time = 0;
  bool accept = false;
  std::vector<int> elementAppearance;
  std::vector<std::vector<int>> idx_vector; // In case hashing is needed, e.g. trackID appears multiple times in different events
  for (int t = 0; t < (looper_detector_timesize - std::ceil(looper_tagger_timewindow[counter] / looper_tagger_granularity[counter])); t++) {
    // for (int t = 0; t < looper_detector_timesize; t++) {
    for (int r = 0; r < o2::tpc::constants::MAXGLOBALPADROW; r++) {
      for (int p = 0; p < TPC_GEOM[r][2] + 1; p++) {
        for (int t_acc = 0; t_acc < std::ceil(looper_tagger_timewindow[counter] / looper_tagger_granularity[counter]); t_acc++) {
          for (int padwindow = 0; padwindow < looper_tagger_padwindow[counter]; padwindow++) {
            // if ((p + padwindow <= TPC_GEOM[r][2]) && (t + t_acc < max_time[sector])) {
            if (operation_mode == 1) {
              num_elements += tagger_counter[t + t_acc][r][p + padwindow];
              if (tagger_counter[t + t_acc][r][p + padwindow] > 0)
                avg_charge += tagger[t + t_acc][r][p + padwindow] / tagger_counter[t + t_acc][r][p + padwindow];
            } else if (operation_mode == 2) {
              num_elements += tagger_counter[t + t_acc][r][p + padwindow];
              idx_vector.push_back(sorted_idx[t + t_acc][r][p + padwindow]); // only filling trackID, otherwise use std::array<int,3> and ArrayHasher for undorder map and unique association
            }
            // }
          }
        }

        accept = (operation_mode == 1 && avg_charge >= looper_tagger_threshold_q[counter] && num_elements >= looper_tagger_threshold_num[counter]);
        if (!accept && operation_mode == 2 && num_elements >= looper_tagger_threshold_num[counter]) {
          elementAppearance = hasElementAppearedMoreThanNTimesInVectors(idx_vector, ideal_mclabels, looper_tagger_threshold_num[counter]);
          accept = (elementAppearance.size() == 0 ? false : true);
        }

        if (accept) {
          // This needs to be modified still
          for (int lbl : elementAppearance) {
            for (std::vector<int> idx_v : idx_vector) {
              for (int idx : idx_v) {
                if (ideal_mclabels[idx][0] == lbl) {
                  ideal_pad = std::round(index_map[idx][1]);
                  ideal_time = std::round(index_map[idx][2]);
                  sigma_pad = std::round(sigma_map[idx][0]);
                  sigma_time = std::round(sigma_map[idx][1]);
                  for (int excl_time = ideal_time - sigma_time; excl_time <= ideal_time + sigma_time; excl_time++) {
                    for (int excl_pad = ideal_pad - sigma_pad; excl_pad <= ideal_pad + sigma_pad; excl_pad++) {
                      if ((excl_pad < 0) || (excl_pad > TPC_GEOM[r][2]) || (excl_time < 0) || (excl_time > (max_time[sector]))) {
                        continue;
                      } else {
                        looper_tagged_region[excl_time][r][excl_pad] = lbl;
                      }
                    }
                  }
                }
              }
            }
          }
          // for (int t_tag = 0; t_tag < std::ceil(looper_tagger_timewindow[counter] / looper_tagger_granularity[counter]); t_tag++) {
          //   looper_tagged_region[t + t_tag][r][p] = 1;
          // }
        }

        accept = false;
        idx_vector.clear();
        elementAppearance.clear();
        avg_charge = 0;
        num_elements = 0;
      }
    }
  }

  if (create_output == 1) {
    // Saving the tagged region to file
    std::stringstream file_in;
    file_in << "looper_tagger_" << sector << "_" << counter << ".root";
    TFile* outputFileLooperTagged = new TFile(file_in.str().c_str(), "RECREATE");
    TTree* looper_tagged_tree = new TTree("tagged_region", "tree");

    int tagged_sector = sector, tagged_row = 0, tagged_pad = 0, tagged_time = 0;
    looper_tagged_tree->Branch("tagged_sector", &sector);
    looper_tagged_tree->Branch("tagged_row", &tagged_row);
    looper_tagged_tree->Branch("tagged_pad", &tagged_pad);
    looper_tagged_tree->Branch("tagged_time", &tagged_time);

    for (int t = 0; t < max_time[sector]; t++) {
      for (int r = 0; r < o2::tpc::constants::MAXGLOBALPADROW; r++) {
        for (int p = 0; p < TPC_GEOM[r][2] + 1; p++) {
          if (looper_tagged_region[t][r][p] >= 0) {
            tagged_row = r;
            tagged_pad = p;
            tagged_time = t;
            looper_tagged_tree->Fill();
          }
        }
      }
    }

    looper_tagged_tree->Write();
    outputFileLooperTagged->Close();
  }

  if (verbose > 2)
    LOG(info) << "[" << sector << "] Looper tagging complete.";

  tagger_counter.clear();
  tagger.clear();

  return looper_tagged_region;
}

// ---------------------------------
template <class T>
std::vector<std::vector<std::vector<int>>> qaIdeal::looper_tagger_full(int sector, int counter, T& index_map, std::vector<int>& index_array, std::vector<std::array<int, 3>>& ideal_mclabels, std::vector<std::array<float, 2>>& sigma_map)
{

  std::vector<std::vector<int>> sorted_idx(o2::tpc::constants::MAXGLOBALPADROW);
  std::vector<std::vector<std::vector<int>>> tagger_map(max_time[sector]); // some safety margin for the sigma_time

  for (int time = 0; time < max_time[sector]; time++) {
    tagger_map[time].resize(o2::tpc::constants::MAXGLOBALPADROW);
    for (int row = 0; row < o2::tpc::constants::MAXGLOBALPADROW; row++) {
      tagger_map[time][row].resize(TPC_GEOM[row][2] + 1);
      for (int pad = 0; pad < (TPC_GEOM[row][2] + 1); pad++) {
        tagger_map[time][row][pad] = 0;
      }
    }
  }

  // Improvements:
  // - Check the charge sigma -> Looper should have narrow sigma in charge
  // - Check width between clusters -> Looper should have regular distance -> peak in distribution of distance
  // - Check for gaussian distribution of charge: D'Agostino-Pearson

  for (auto idx : index_array) {
    sorted_idx[std::round(index_map[idx][0])].push_back(idx);
  }

  // int unit_volume = looper_tagger_timewindow[counter] * 3;
  std::unordered_map<int, std::vector<int>> distinct_elements;
  int row = 0, pad = 0, time = 0, sigma_pad = 0, sigma_time = 0;
  std::vector<std::vector<int>> idx_vector; // In case hashing is needed, e.g. trackID appears multiple times in different events
  for (int r = 0; r < o2::tpc::constants::MAXGLOBALPADROW; r++) {
    idx_vector.push_back(sorted_idx[r]); // only filling trackID, otherwise use std::array<int,3> and ArrayHasher for undorder map and unique association
    distinct_elements = distinctElementAppearance(idx_vector, ideal_mclabels);
    for (auto elem : distinct_elements) {
      if (elem.second.size() > looper_tagger_threshold_num[counter]) {
        for (int idx : elem.second) {
          row = std::round(index_map[idx][0]);
          pad = std::round(index_map[idx][1]);
          time = std::round(index_map[idx][2]);
          sigma_pad = std::round(sigma_map[idx][0]);
          sigma_time = std::round(sigma_map[idx][1]);
          for (int excl_time = time - sigma_time; excl_time < time + sigma_time; excl_time++) {
            for (int excl_pad = pad - sigma_pad; excl_pad < pad + sigma_pad; excl_pad++) {
              if ((excl_pad < 0) || (excl_pad > TPC_GEOM[row][2]) || (excl_time < 0) || (excl_time > (max_time[sector] - 1))) {
                continue;
              } else {
                tagger_map[excl_time][row][excl_pad] = 1;
              }
            }
          }
        }
      }
    }
    distinct_elements.clear();
  }

  if (create_output == 1) {
    // Saving the tagged region to file
    std::stringstream file_in;
    file_in << "looper_tagger_" << sector << "_" << counter << ".root";
    TFile* outputFileLooperTagged = new TFile(file_in.str().c_str(), "RECREATE");
    TTree* looper_tagged_tree = new TTree("tagged_region", "tree");

    int tagged_sector = sector, tagged_row = 0, tagged_pad = 0, tagged_time = 0;
    looper_tagged_tree->Branch("tagged_sector", &sector);
    looper_tagged_tree->Branch("tagged_row", &tagged_row);
    looper_tagged_tree->Branch("tagged_pad", &tagged_pad);
    looper_tagged_tree->Branch("tagged_time", &tagged_time);

    for (int t = 0; t < max_time[sector]; t++) {
      for (int r = 0; r < o2::tpc::constants::MAXGLOBALPADROW; r++) {
        for (int p = 0; p < TPC_GEOM[r][2] + 1; p++) {
          if (tagger_map[t][r][p] == 1) {
            tagged_row = r;
            tagged_pad = p;
            tagged_time = t;
            looper_tagged_tree->Fill();
          }
        }
      }
    }

    looper_tagged_tree->Write();
    outputFileLooperTagged->Close();
  }

  if (verbose > 2)
    LOG(info) << "[" << sector << "] Looper tagging complete.";

  return tagger_map;
}

// ---------------------------------
template <class T>
void qaIdeal::remove_loopers_digits(int sector, int counter, std::vector<std::vector<std::vector<int>>>& looper_map, T& map, std::vector<int>& index_array)
{
  std::vector<int> new_index_array;
  T new_map;

  for (auto idx : index_array) {
    // if (looper_map[std::floor(map[idx][2] / (float)looper_tagger_granularity[counter])][std::round(map[idx][0])][std::round(map[idx][1])] == 0) {
    if (looper_map[std::round(map[idx][2])][std::round(map[idx][0])][std::round(map[idx][1])] < 0) {
      new_index_array.push_back(idx);
    }
  }

  if (verbose > 2)
    LOG(info) << "[" << sector << "] Old size of maxima index array: " << index_array.size() << "; New size: " << new_index_array.size();

  index_array = new_index_array;
}

// ---------------------------------
template <class T>
void qaIdeal::remove_loopers_native(int sector, int counter, std::vector<std::vector<std::vector<int>>>& looper_map, T& map, std::vector<int>& index_array)
{
  // This function does not remove by using the looper_map because index_array corresponds to maxima_digits which are removed from the digits anyway in a previous step
  T new_map;
  for (auto idx : index_array) {
    new_map.push_back(map[idx]);
  }
  map = new_map;
}

// ---------------------------------
void qaIdeal::remove_loopers_ideal(int sector, int counter, std::vector<std::vector<std::vector<int>>>& looper_map, std::vector<std::array<int, 3>>& ideal_max_map, std::vector<std::array<float, 3>>& ideal_cog_map, std::vector<float>& ideal_max_q, std::vector<float>& ideal_cog_q, std::vector<std::array<float, 2>>& ideal_sigma_map, std::vector<std::array<int, 3>>& ideal_mclabels)
{

  std::vector<int> ideal_idx_map(ideal_max_map.size()), new_ideal_idx_map;
  std::iota(ideal_idx_map.begin(), ideal_idx_map.end(), 0);

  for (int m = 0; m < ideal_idx_map.size(); m++) {
    // if (looper_map[std::floor(ideal_cog_map[m][2] / (float)looper_tagger_granularity[counter])][std::round(ideal_cog_map[m][0])][std::round(ideal_cog_map[m][1])] == 0)
    // if (looper_map[std::round(ideal_cog_map[m][2])][std::round(ideal_cog_map[m][0])][std::round(ideal_cog_map[m][1])] >= 0)
    if (looper_map[std::round(ideal_cog_map[m][2])][std::round(ideal_cog_map[m][0])][std::round(ideal_cog_map[m][1])] != ideal_mclabels[m][0]) // Compares trkID and only removes ideal clusters with identical ID
      new_ideal_idx_map.push_back(m);
  }

  std::vector<std::array<int, 3>> new_ideal_max_map;
  std::vector<std::array<float, 3>> new_ideal_cog_map;
  std::vector<float> new_ideal_max_q;
  std::vector<float> new_ideal_cog_q;
  std::vector<std::array<float, 2>> new_ideal_sigma_map;
  std::vector<std::array<int, 3>> new_ideal_mclabels;

  for (auto m : new_ideal_idx_map) {
    new_ideal_max_map.push_back(ideal_max_map[m]);
    new_ideal_cog_map.push_back(ideal_cog_map[m]);
    new_ideal_max_q.push_back(ideal_max_q[m]);
    new_ideal_cog_q.push_back(ideal_cog_q[m]);
    new_ideal_sigma_map.push_back(ideal_sigma_map[m]);
    new_ideal_mclabels.push_back(ideal_mclabels[m]);
  }

  ideal_max_map = new_ideal_max_map;
  ideal_cog_map = new_ideal_cog_map;
  ideal_max_q = new_ideal_max_q;
  ideal_cog_q = new_ideal_cog_q;
  ideal_sigma_map = new_ideal_sigma_map;
  ideal_mclabels = new_ideal_mclabels;
}

// ---------------------------------
template <class T>
int qaIdeal::tagLabel(T& element, std::vector<std::vector<std::vector<int>>>& looper_map)
{
  if (std::is_same<T, int>::value) {
    return looper_map[element[2]][element[0]][element[1]];
  } else {
    return looper_map[rint(element[2])][rint(element[0])][rint(element[1])];
  }
}

// ---------------------------------
template <class T>
void qaIdeal::run_network(int sector, T& map2d, std::vector<int>& maxima_digits, std::vector<std::array<int, 3>>& digit_map, std::vector<float>& digit_q, std::vector<std::array<float, 7>>& network_map, int eval_mode)
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
    bool compromised_charge = (central_charges[max] <= 0);
    if(compromised_charge){
      LOG(warning) << "[" << sector << "] Central charge < 0 detected at index " << maxima_digits[max] << " = (sector: " << sector << ", row: " << digit_map[maxima_digits[max]][0] << ", pad: " << digit_map[maxima_digits[max]][1] << ", time: " << digit_map[maxima_digits[max]][2] << ") ! Continuing with input vector set to -1 everywhere...";
    }
    for (int row = 0; row < 2 * global_shift[2] + 1; row++) {
      for (int pad = 0; pad < 2 * global_shift[0] + 1; pad++) {
        for (int time = 0; time < 2 * global_shift[1] + 1; time++) {
          if(compromised_charge){
            input_vector[max * index_shift_global + row * index_shift_row + pad * index_shift_pad + time] = -1;
          }
          else{
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
  }

  if (eval_mode == 0 || eval_mode == 2) {

    network_reg_size = 0;
    for (int max = 0; max < maxima_digits.size(); max++) {
      for (int idx = 0; idx < index_shift_global; idx++) {
        temp_input[(max % networkInputSize) * index_shift_global + idx] = input_vector[max * index_shift_global + idx];
      }
      /*
      if (verbose >= 5 && max == 10) {
        LOG(info) << "Size of the input vector: " << temp_input.size();
        LOG(info) << "Example input for neural network";
        for (int i = 0; i < 11; i++) {
          LOG(info) << "[ " << temp_input[11 * i + 0] << " " << temp_input[11 * i + 1] << " " << temp_input[11 * i + 2] << " " << temp_input[11 * i + 3] << " " << temp_input[11 * i + 4] << " " << temp_input[11 * i + 5] << " " << temp_input[11 * i + 6] << " " << temp_input[11 * i + 7] << " " << temp_input[11 * i + 8] << " " << temp_input[11 * i + 9] << " " << temp_input[11 * i + 10] << " ]";
        }
        LOG(info) << "Example output (classification): " << network_classification.inference(temp_input, networkInputSize)[0];
        LOG(info) << "Example output (regression): " << network_regression.inference(temp_input, networkInputSize)[0] << ", " << network_regression.inference(temp_input, networkInputSize)[1] << ", " << network_regression.inference(temp_input, networkInputSize)[2];
      }
      */
      if ((max + 1) % networkInputSize == 0 || max + 1 == maxima_digits.size()) {
        float* out_net = network_classification.inference(temp_input, networkInputSize);
        for (int idx = 0; idx < networkInputSize; idx++) {
          if (max + 1 == maxima_digits.size() && idx > (max % networkInputSize))
            break;
          else {
            output_network_class[int(max / networkInputSize) * networkInputSize  + idx] = out_net[idx];
            if (out_net[idx] > networkClassThres) {
              network_reg_size++;
            } else {
              new_max_dig[int(max / networkInputSize) * networkInputSize  + idx] = -1;
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
    index_pass.resize(network_reg_size);
    maxima_digits.resize(network_reg_size);
    network_map.resize(network_reg_size);

    for (int max = 0; max < new_max_dig.size(); max++) {
      if (new_max_dig[max] > -1) {
        maxima_digits[counter_max_dig] = new_max_dig[max];
        index_pass[counter_max_dig] = max;
        network_map[counter_max_dig][0] = digit_map[new_max_dig[max]][0];
        network_map[counter_max_dig][1] = digit_map[new_max_dig[max]][1];
        network_map[counter_max_dig][2] = digit_map[new_max_dig[max]][2];
        network_map[counter_max_dig][3] = 1;
        network_map[counter_max_dig][4] = 1;
        network_map[counter_max_dig][5] = digit_q[new_max_dig[max]];
        network_map[counter_max_dig][6] = digit_q[new_max_dig[max]];
        counter_max_dig++;
      }
    }

    LOG(info) << "[" << sector << "] Classification network done!";
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
      network_map[i][3] = 1;
      network_map[i][4] = 1;
      network_map[i][5] = digit_q[maxima_digits[i]];
      network_map[i][6] = digit_q[maxima_digits[i]];
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

  // for (int row = 0; row < o2::tpc::constants::MAXGLOBALPADROW + 3 * global_shift[2]; row++) {
  //   for (int pad = 0; pad < TPC_GEOM[151][2] + 2 * global_shift[0]; pad++) {
  //     for (int time = 0; time < (max_time[sector] + 2 * global_shift[1]); time++) {
  //       map2d[mode][time][row][pad] = -1;
  //     }
  //   }
  // }

  fill_nested_container(map2d[mode], -1);
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
  std::vector<std::array<int, 3>> ideal_mclabels;
  std::vector<std::array<int, 3>> digit_map, ideal_max_map;
  std::vector<std::array<float, 3>> ideal_cog_map, digit_clusterizer_map;
  std::vector<std::array<float, 2>> ideal_sigma_map;
  std::vector<std::array<float, 7>> network_map, native_map;
  std::vector<float> ideal_max_q, ideal_cog_q, digit_q, digit_clusterizer_q;
  std::vector<std::vector<std::vector<std::vector<int>>>> tagger_maps(looper_tagger_granularity.size());

  LOG(info) << "--- Starting process for sector " << loop_sectors << " ---";

  if (mode.find(std::string("native")) != std::string::npos) {
    read_native(loop_sectors, digit_map, native_map, digit_q);
  } else if (mode.find(std::string("network")) != std::string::npos) {
    read_digits(loop_sectors, digit_map, digit_q);
  } else {
    read_digits(loop_sectors, digit_map, digit_q);
  }

  read_ideal(loop_sectors, ideal_max_map, ideal_max_q, ideal_cog_map, ideal_sigma_map, ideal_cog_q, ideal_mclabels);

  qa_t map2d = init_map2d<qa_t>(loop_sectors);

  std::vector<int> ideal_idx(ideal_cog_map.size());
  std::iota(ideal_idx.begin(), ideal_idx.end(), 0);

  if (mode.find(std::string("looper_tagger")) != std::string::npos) {
    for (int counter = 0; counter < looper_tagger_granularity.size(); counter++) {
      tagger_maps[counter] = looper_tagger(loop_sectors, counter, ideal_cog_map, ideal_sigma_map, ideal_cog_q, ideal_idx, ideal_mclabels, looper_tagger_opmode);
      // remove_loopers_ideal(loop_sectors, counter, tagger_maps[counter], ideal_max_map, ideal_cog_map, ideal_max_q, ideal_cog_q, ideal_sigma_map, ideal_mclabels);
    }
  }

  fill_map2d<qa_t>(loop_sectors, map2d, digit_map, ideal_max_map, ideal_max_q, ideal_cog_map, ideal_cog_q, -1);

  if ((mode.find(std::string("network")) == std::string::npos) && (mode.find(std::string("native")) == std::string::npos)) {
    find_maxima<qa_t>(loop_sectors, map2d, maxima_digits, digit_q);
    // if (mode.find(std::string("looper_tagger")) != std::string::npos) {
    //   for (int counter = 0; counter < looper_tagger_granularity.size(); counter++) {
    //     remove_loopers_digits(loop_sectors, counter, tagger_maps[counter], digit_map, maxima_digits);
    //   }
    // }
    if (mode.find(std::string("clusterizer")) != std::string::npos) {
      native_clusterizer(map2d, digit_map, maxima_digits, digit_q, digit_clusterizer_map, digit_clusterizer_q);
    }
    overwrite_map2d<qa_t>(loop_sectors, map2d, digit_map, maxima_digits, 1);
  } else {
    if (mode.find(std::string("native")) == std::string::npos) {
      find_maxima<qa_t>(loop_sectors, map2d, maxima_digits, digit_q);
      // if (mode.find(std::string("looper_tagger")) != std::string::npos) {
      //   for (int counter = 0; counter < looper_tagger_granularity.size(); counter++) {
      //     remove_loopers_digits(loop_sectors, counter, tagger_maps[counter], digit_map, maxima_digits);
      //   }
      // }
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
      // if (mode.find(std::string("looper_tagger")) != std::string::npos) {
      //   for (int counter = 0; counter < looper_tagger_granularity.size(); counter++) {
      //     remove_loopers_digits(loop_sectors, counter, tagger_maps[counter], digit_map, maxima_digits);
      //     remove_loopers_native(loop_sectors, counter, tagger_maps[counter], native_map, maxima_digits);
      //   }
      // }
      overwrite_map2d<qa_t>(loop_sectors, map2d, digit_map, maxima_digits, 1);
    }
  }

  std::vector<int> assigned_ideal(ideal_max_map.size(), 0);
  std::vector<std::array<int, 25>> assignments_dig_to_id(ideal_max_map.size());
  std::vector<int> assigned_digit(maxima_digits.size(), 0);
  std::vector<std::array<int, 25>> assignments_id_to_dig(maxima_digits.size());
  int current_neighbour;
  std::vector<float> clone_order(maxima_digits.size(), 0), fractional_clones_vector(maxima_digits.size(), 0);

  fill_nested_container(assignments_dig_to_id, -1);
  fill_nested_container(assignments_id_to_dig, -1);
  fill_nested_container(assigned_digit, 0);
  fill_nested_container(assigned_ideal, 0);
  fill_nested_container(clone_order, 0);

  number_of_digit_max[loop_sectors] += maxima_digits.size();
  number_of_ideal_max[loop_sectors] += ideal_max_map.size();

  for (int max = 0; max < ideal_max_map.size(); max++) {
    if (ideal_cog_q[max] >= threshold_cogq && ideal_max_q[max] >= threshold_maxq) {
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
        if (current_neighbour >= -1) {
          assignments_id_to_dig[locdigit][layer_count + nn] = ((current_neighbour != -1 && assigned_digit[locdigit] == 0) ? (assigned_ideal[current_neighbour] == 0 ? current_neighbour : -1) : -1);
        }
      }
      if (verbose >= 4)
        LOG(info) << "[" << loop_sectors << "] Done with assignment for digit maxima, layer " << layer;

      // Level-3 loop: Goes through all ideal maxima and checks neighbourhood for potential digit maxima
      std::array<int, 3> rounded_cog;
      for (unsigned int locideal = 0; locideal < ideal_max_map.size(); locideal++) {
        for (int i = 0; i < 3; i++) {
          rounded_cog[i] = round(ideal_cog_map[locideal][i]);
        }
        current_neighbour = test_neighbour(rounded_cog, adj_mat[layer][nn], map2d, 1);
        if (current_neighbour >= -1) {
          assignments_dig_to_id[locideal][layer_count + nn] = ((current_neighbour != -1 && assigned_ideal[locideal] == 0) ? (assigned_digit[current_neighbour] == 0 ? current_neighbour : -1) : -1);
        }
      }
      if (verbose >= 4)
        LOG(info) << "[" << loop_sectors << "] Done with assignment for ideal maxima, layer " << layer;
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

    layer_count += adj_mat[layer].size();
  }

  // Check tagging
  std::vector<int> ideal_tagged(ideal_cog_map.size(), 0), digit_tagged(maxima_digits.size(), 0);
  std::vector<std::vector<int>> ideal_tag_label((int)ideal_cog_map.size(), std::vector<int>((int)tagger_maps.size(), -1)), digit_tag_label((int)maxima_digits.size(), std::vector<int>((int)tagger_maps.size(), -1));

  if (mode.find(std::string("looper_tagger")) != std::string::npos) {
    int tm_counter = 0;
    for (auto tm : tagger_maps) {
      int counter = 0;
      for (auto elem_id : ideal_cog_map) {
        ideal_tag_label[counter][tm_counter] = tm[rint(elem_id[2])][rint(elem_id[0])][rint(elem_id[1])];
        ideal_tagged[counter] = (int)(((bool)ideal_tagged[counter]) || (tm[rint(elem_id[2])][rint(elem_id[0])][rint(elem_id[1])] > -1));
        counter += 1;
      }
      counter = 0;
      for (auto elem_dig : maxima_digits) {
        digit_tag_label[counter][tm_counter] = tm[digit_map[elem_dig][2]][digit_map[elem_dig][0]][digit_map[elem_dig][1]];
        digit_tagged[counter] = (int)(((bool)digit_tagged[counter]) || (tm[digit_map[elem_dig][2]][digit_map[elem_dig][0]][digit_map[elem_dig][1]] > -1));
        counter += 1;
      }
      tm_counter += 1;
    }
  }

  // Checks the number of assignments that have been made with the above loops
  int count_elements_findable = 0, count_elements_dig = 0, count_elements_id = 0;
  for (unsigned int locideal = 0; locideal < assignments_dig_to_id.size(); locideal++) {
    if (!ideal_tagged[locideal]) { // if region is tagged, don't use ideal cluster for ECF calculation
      count_elements_id = 0;
      count_elements_findable = 0;
      for (auto elem_dig : assignments_dig_to_id[locideal]) {
        if (checkIdx(elem_dig) && !digit_tagged[elem_dig]) {
          count_elements_id += 1;
          if (ideal_cog_q[locideal] >= threshold_cogq && ideal_max_q[locideal] >= threshold_maxq) { // FIXME: assignemts to an ideal cluster which are findable? -> Digit maxima which satisfy the criteria not ideal clsuters?!
            count_elements_findable += 1;
          }
        }
      }
      // if (verbose >= 5 && (locideal%10000)==0) LOG(info) << "Count elements: " << count_elements_id << " locideal: " << locideal << " assignments_ideal: " << assignments_ideal[count_elements_id];
      assignments_ideal[loop_sectors][count_elements_id] += 1;
      assignments_ideal_findable[loop_sectors][count_elements_findable] += 1;
    }
  }
  for (unsigned int locdigit = 0; locdigit < assignments_id_to_dig.size(); locdigit++) {
    if (!digit_tagged[locdigit]) { // if region is tagged, don't use digit maximum for ECF calculation
      count_elements_dig = 0;
      count_elements_findable = 0;
      for (auto elem_id : assignments_id_to_dig[locdigit]) {
        if (checkIdx(elem_id) && !ideal_tagged[elem_id]) {
          count_elements_dig += 1;
          if (ideal_cog_q[elem_id] >= threshold_cogq && ideal_max_q[elem_id] >= threshold_maxq) {
            count_elements_findable += 1;
          }
        }
      }
      // if (verbose >= 5 && (locdigit%10000)==0) LOG(info) << "Count elements: " << count_elements_dig << " locdigit: " << locdigit << " assignments_digit: " << assignments_digit[count_elements_dig];
      // LOG(info) << "sector " << loop_sectors << "; row " <<  digit_map[maxima_digits[locdigit]][0] << "; pad " <<  digit_map[maxima_digits[locdigit]][1] << "; time " <<  digit_map[maxima_digits[locdigit]][2] << "; tagged: " << digit_tagged[locdigit];
      assignments_digit[loop_sectors][count_elements_dig] += 1;
      assignments_digit_findable[loop_sectors][count_elements_findable] += 1;
    }
  }

  if (verbose >= 3)
    LOG(info) << "[" << loop_sectors << "] Done checking the number of assignments";

  // Clone-rate (Integer)
  for (unsigned int locdigit = 0; locdigit < assignments_id_to_dig.size(); locdigit++) {
    if (!digit_tagged[locdigit]) {
      int count_links = 0;
      float count_weighted_links = 0;
      for (auto elem_id : assignments_id_to_dig[locdigit]) {
        if (checkIdx(elem_id)) {
          count_links++;
          int count_links_second = 0;
          for (auto elem_dig : assignments_dig_to_id[elem_id]) {
            if (checkIdx(elem_dig)) { //&& (elem_dig != locdigit)){
              count_links_second++;
            }
          }
          if (count_links_second == 0) {
            count_weighted_links += 1;
          }
        }
      }
      if (count_weighted_links > 1) {
        clone_order[locdigit] = count_weighted_links - 1.f;
      }
    }
  }
  for (unsigned int locdigit = 0; locdigit < maxima_digits.size(); locdigit++) {
    clones[loop_sectors] += clone_order[locdigit];
  }

  // Clone-rate (fractional)
  for (unsigned int locideal = 0; locideal < assignments_dig_to_id.size(); locideal++) {
    if (!ideal_tagged[locideal]) {
      int count_links = 0;
      for (auto elem_dig : assignments_dig_to_id[locideal]) {
        if (checkIdx(elem_dig)) {
          count_links += 1;
        }
      }
      for (auto elem_dig : assignments_dig_to_id[locideal]) {
        if (checkIdx(elem_dig)) {
          fractional_clones_vector[elem_dig] += 1.f / (float)count_links;
        }
      }
    }
  }
  for (auto elem_frac : fractional_clones_vector) {
    if (elem_frac > 1) {
      fractional_clones[loop_sectors] += elem_frac - 1;
    }
  }

  if (verbose >= 3)
    LOG(info) << "[" << loop_sectors << "] Done determining the clone rate";

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

    std::vector<std::array<float, 13>> native_ideal_assignemnt;
    std::array<float, 13> current_element;

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
            if ((ideal_cog_q[current_idx_id] < threshold_cogq && ideal_max_q[current_idx_id] < threshold_maxq) || (assigned_ideal[current_idx_id] != 0)) {
              is_min_dist = false;
              break;
            } else {
              current_distance_dig_to_id = std::pow((native_map[max_point][2] - ideal_cog_map[current_idx_id][2]), 2) + std::pow((native_map[max_point][1] - ideal_cog_map[current_idx_id][1]), 2);
              // if the distance is less than the previous one check if update should be made
              if (current_distance_dig_to_id < distance_assignment) {
                for (int j = 0; j < 25; j++) {
                  current_idx_dig = assignments_dig_to_id[current_idx_id][j];
                  if (checkIdx(current_idx_dig)) {
                    if (assigned_digit[current_idx_dig] == 0) {
                      // calculate mutual distance from current ideal CoG to all assigned digit maxima. Update if and only if distance is minimal. Else do not assign.
                      current_distance_id_to_dig = std::pow((native_map[current_idx_dig][2] - ideal_cog_map[current_idx_id][2]), 2) + std::pow((native_map[current_idx_dig][1] - ideal_cog_map[current_idx_id][1]), 2);
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
                  check_assignment += 1;
                  distance_assignment = current_distance_dig_to_id;
                  index_assignment = current_idx_id;
                  // Adding an assignment in order to avoid duplication
                  assigned_digit[max_point] += 1;
                  assigned_ideal[current_idx_id] += 1;
                  current_element[0] = ideal_cog_map[current_idx_id][0];
                  current_element[1] = ideal_cog_map[current_idx_id][1];
                  current_element[2] = ideal_cog_map[current_idx_id][2];
                  current_element[3] = native_map[max_point][0];
                  current_element[4] = native_map[max_point][1];
                  current_element[5] = native_map[max_point][2];
                  current_element[6] = native_map[max_point][3];
                  current_element[7] = native_map[max_point][4];
                  current_element[8] = native_map[max_point][5];
                  current_element[9] = native_map[max_point][6];
                  current_element[10] = ideal_mclabels[current_idx_id][0];
                  current_element[11] = ideal_mclabels[current_idx_id][1];
                  current_element[12] = ideal_mclabels[current_idx_id][2];
                }
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

    // native_writer_map.clear();
    // native_writer_map.resize(native_ideal_assignemnt.size());

    float nat_row = 0, nat_time = 0, nat_pad = 0, nat_sigma_time = 0, nat_sigma_pad = 0, id_row = 0, id_time = 0, id_pad = 0, native_minus_ideal_time = 0, native_minus_ideal_pad = 0, nat_qTot = 0, nat_qMax = 0;
    native_ideal->Branch("sector", &loop_sectors);
    native_ideal->Branch("native_row", &nat_row);
    native_ideal->Branch("native_cog_time", &nat_time);
    native_ideal->Branch("native_cog_pad", &nat_pad);
    native_ideal->Branch("native_sigma_time", &nat_sigma_time);
    native_ideal->Branch("native_sigma_pad", &nat_sigma_pad);
    native_ideal->Branch("ideal_row", &id_row);
    native_ideal->Branch("ideal_cog_time", &id_time);
    native_ideal->Branch("ideal_cog_pad", &id_pad);
    native_ideal->Branch("native_minus_ideal_time", &native_minus_ideal_time);
    native_ideal->Branch("native_minus_ideal_pad", &native_minus_ideal_pad);

    m.lock();
    for (int elem = 0; elem < native_ideal_assignemnt.size(); elem++) {
      id_row = native_ideal_assignemnt[elem][0];
      id_pad = native_ideal_assignemnt[elem][1];
      id_time = native_ideal_assignemnt[elem][2];
      nat_row = native_ideal_assignemnt[elem][3];
      nat_pad = native_ideal_assignemnt[elem][4];
      nat_time = native_ideal_assignemnt[elem][5];
      nat_sigma_pad = native_ideal_assignemnt[elem][6];
      nat_sigma_time = native_ideal_assignemnt[elem][7];
      nat_qTot = native_ideal_assignemnt[elem][8];
      nat_qMax = native_ideal_assignemnt[elem][9];
      native_minus_ideal_time = nat_time - id_time;
      native_minus_ideal_pad = nat_pad - id_pad;
      native_ideal->Fill();

      if (write_native_file) {
        native_writer_map.push_back({(float)loop_sectors, nat_row, nat_pad, nat_time, nat_sigma_time, nat_sigma_pad, nat_qMax, nat_qTot, native_ideal_assignemnt[elem][10], native_ideal_assignemnt[elem][11], native_ideal_assignemnt[elem][12]});
      }
    }
    m.unlock();

    native_ideal->Write();
    outputFileNativeIdeal->Close();

    native_ideal_assignemnt.clear();
  }

  if (mode.find(std::string("network")) != std::string::npos && create_output == 1) {

    if (verbose >= 3)
      LOG(info) << "[" << loop_sectors << "] Network-Ideal assignment...";

    // creating training data for the neural network
    int data_size = maxima_digits.size();

    std::vector<std::array<float, 8>> network_ideal_assignemnt;
    std::array<float, 8> current_element;

    std::fill(assigned_ideal.begin(), assigned_ideal.end(), 0);
    std::fill(assigned_digit.begin(), assigned_digit.end(), 0);

    // Some useful variables
    int map_dig_idx = 0, map_q_idx = 0, check_assignment = 0, index_assignment = -1, current_idx_id = -1, current_idx_dig = -1;
    float distance_assignment = 100000.f, current_distance_dig_to_id = 0, current_distance_id_to_dig = 0;
    bool is_min_dist = true;

    m.lock();
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
            if ((ideal_cog_q[current_idx_id] < threshold_cogq && ideal_max_q[current_idx_id] < threshold_maxq) || (assigned_ideal[current_idx_id] != 0)) {
              is_min_dist = false;
              break;
            } else {
              current_distance_dig_to_id = std::pow((network_map[max_point][2] - ideal_cog_map[current_idx_id][2]), 2) + std::pow((network_map[max_point][1] - ideal_cog_map[current_idx_id][1]), 2);
              // if the distance is less than the previous one check if update should be made
              if (current_distance_dig_to_id < distance_assignment) {
                for (int j = 0; j < 25; j++) {
                  current_idx_dig = assignments_dig_to_id[current_idx_id][j];
                  if (checkIdx(current_idx_dig)) {
                    if (assigned_digit[current_idx_dig] == 0) {
                      // calculate mutual distance from current ideal CoG to all assigned digit maxima. Update if and only if distance is minimal. Else do not assign.
                      current_distance_id_to_dig = std::pow((network_map[current_idx_dig][2] - ideal_cog_map[current_idx_id][2]), 2) + std::pow((network_map[current_idx_dig][1] - ideal_cog_map[current_idx_id][1]), 2);
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
                  check_assignment += 1;
                  distance_assignment = current_distance_dig_to_id;
                  index_assignment = current_idx_id;
                  // Adding an assignment in order to avoid duplication
                  assigned_digit[max_point] += 1;
                  assigned_ideal[current_idx_id] += 1;
                  current_element[0] = network_map[max_point][2];
                  current_element[1] = network_map[max_point][1];
                  current_element[2] = ideal_cog_map[current_idx_id][2];
                  current_element[3] = ideal_cog_map[current_idx_id][1];
                  current_element[5] = ideal_mclabels[current_idx_id][0];
                  current_element[6] = ideal_mclabels[current_idx_id][1];
                  current_element[7] = ideal_mclabels[current_idx_id][2];
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
          if (write_native_file) {
            native_writer_map.push_back(std::array<float, 11>{(float)loop_sectors, network_map[max_point][0], network_map[max_point][1], network_map[max_point][2], network_map[max_point][3], network_map[max_point][4], network_map[max_point][5], network_map[max_point][6], current_element[5], current_element[6], current_element[7]});
          }
        }
      }
    }
    m.unlock();
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

    LOG(info) << "[" << loop_sectors << "] Network map size: " << network_map.size();
    LOG(info) << "[" << loop_sectors << "] Network-ideal size: " << network_ideal_assignemnt.size();

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

    // Checks if digit is assigned / has non-looper assignments
    std::vector<int> digit_has_non_looper_assignments(maxima_digits.size(), -1); // -1 = has no assignments, 0 = has 0 non-looper assignments, n = has n non-looper assignments
    std::vector<std::vector<int>> digit_non_looper_assignment_labels(maxima_digits.size());
    for (int dig_max = 0; dig_max < maxima_digits.size(); dig_max++) {
      bool digit_has_assignment = false;
      for (int ass : assignments_id_to_dig[dig_max]) {
        if (ass != -1) {
          digit_has_assignment = true;
          bool is_tagged = false;
          if (ideal_tagged[ass]) {
            for (int lbl : ideal_tag_label[ass]) {
              is_tagged |= (lbl != -1 ? (ideal_mclabels[ass][0] == lbl) : false);
            }
          }
          if (!is_tagged) {
            digit_has_non_looper_assignments[dig_max] == -1 ? digit_has_non_looper_assignments[dig_max] += 2 : digit_has_non_looper_assignments[dig_max] += 1;
            digit_non_looper_assignment_labels[dig_max].push_back(ass);
          }
        }
      }
      if (digit_has_non_looper_assignments[dig_max] == -1 && digit_has_assignment) {
        digit_has_non_looper_assignments[dig_max] = 0;
      }
    }

    // Creation of training data
    std::vector<int> index_digits(digit_map.size(), 0);
    std::iota(index_digits.begin(), index_digits.end(), 0);
    overwrite_map2d(loop_sectors, map2d, digit_map, index_digits, 0);

    if (verbose >= 3)
      LOG(info) << "[" << loop_sectors << "] Creating training data...";

    // Training data: NN input
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

    o2::MCTrack current_track;
    std::vector<float> cluster_pT(data_size, -1), cluster_eta(data_size, -1), cluster_mass(data_size, -1), cluster_p(data_size, -1);
    std::vector<int> tr_data_Y_class(data_size, -1), cluster_isPrimary(data_size, -1), cluster_isTagged(data_size, -1);
    std::vector<std::array<std::array<float, 5>, 5>> tr_data_Y_reg(data_size); // for each data element: for all possible assignments: {trY_time, trY_pad, trY_sigma_pad, trY_sigma_time, trY_q}
    fill_nested_container(tr_data_Y_reg, -1);

    std::fill(assigned_ideal.begin(), assigned_ideal.end(), 0);
    std::fill(assigned_digit.begin(), assigned_digit.end(), 0);

    // Some useful variables
    int map_dig_idx = 0, map_q_idx = 0, check_assignment = 0, index_assignment = -1, current_idx_id = -1, current_idx_dig = -1, row_offset = 0, pad_offset = 0, class_label = 0;
    float distance_assignment = 100000.f, current_distance_dig_to_id = 0, current_distance_id_to_dig = 0;
    bool is_min_dist = true, is_tagged = false;

    for (int max_point = 0; max_point < data_size; max_point++) {
      // is_tagged = (bool)digit_tagged[max_point];
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

        tr_data_Y_class[max_point] = digit_has_non_looper_assignments[max_point];
        cluster_isTagged[max_point] = (bool)digit_tagged[max_point];

        std::vector<int> sorted_idcs;
        if (digit_has_non_looper_assignments[max_point] > 0) {
          std::vector<float> distance_array(digit_has_non_looper_assignments[max_point], -1);
          for (int counter = 0; counter < digit_has_non_looper_assignments[max_point]; counter++) {
            int ideal_idx = digit_non_looper_assignment_labels[max_point][counter];
            distance_array[counter] = std::pow((digit_map[maxima_digits[max_point]][2] - ideal_cog_map[ideal_idx][2]), 2) + std::pow((digit_map[maxima_digits[max_point]][1] - ideal_cog_map[ideal_idx][1]), 2);
          }

          distance_array.size() > 1 ? sorted_idcs = sort_indices(distance_array) : sorted_idcs = {0};
          for (int counter = 0; counter < digit_has_non_looper_assignments[max_point]; counter++) {
            int ideal_idx = digit_non_looper_assignment_labels[max_point][sorted_idcs[counter]];
            tr_data_Y_reg[max_point][counter][0] = ideal_cog_map[ideal_idx][1] - digit_map[maxima_digits[max_point]][1]; // pad
            tr_data_Y_reg[max_point][counter][1] = ideal_cog_map[ideal_idx][2] - digit_map[maxima_digits[max_point]][2]; // time
            tr_data_Y_reg[max_point][counter][2] = ideal_sigma_map[ideal_idx][0];                                        // sigma pad
            tr_data_Y_reg[max_point][counter][3] = ideal_sigma_map[ideal_idx][1];                                        // sigma time
            if (normalization_mode == 0) {
              tr_data_Y_reg[max_point][counter][4] = ideal_cog_q[ideal_idx] / 1024.f;
            } else if (normalization_mode == 1) {
              tr_data_Y_reg[max_point][counter][4] = ideal_cog_q[ideal_idx] / q_max;
            }

            if (counter == 0) {
              current_track = mctracks[ideal_mclabels[ideal_idx][2]][ideal_mclabels[ideal_idx][1]][ideal_mclabels[ideal_idx][0]];
              cluster_pT[max_point] = current_track.GetPt();
              cluster_eta[max_point] = current_track.GetEta();
              cluster_mass[max_point] = current_track.GetMass();
              cluster_p[max_point] = current_track.GetP();
              cluster_isPrimary[max_point] = (int)current_track.isPrimary();
            }
          }
        }
      }
    }
    if (verbose >= 3)
      LOG(info) << "[" << loop_sectors << "] Done creating training data. Writing to file...";

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
    std::array<std::array<float, 5>, 5> trY; // [branch][data]; data = {trY_time, trY_pad, trY_sigma_pad, trY_sigma_time, trY_q}
    fill_nested_container(trY, -1);
    std::array<std::string, 5> branches{"out_reg_pad", "out_reg_time", "out_reg_sigma_pad", "out_reg_sigma_time", "out_reg_qTotOverqMax"};
    for (int reg_br = 0; reg_br < 5; reg_br++) {
      for (int reg_data = 0; reg_data < 5; reg_data++) {
        std::stringstream current_branch;
        current_branch << branches[reg_br] << "_" << reg_data;
        tr_data->Branch(current_branch.str().c_str(), &trY[reg_data][reg_br]);
      }
    }

    int class_val = 0, idx_sector = 0, idx_row = 0, idx_pad = 0, idx_time = 0;
    float pT = 0, eta = 0, mass = 0, p = 0, isPrimary = 0, isTagged = 0;
    tr_data->Branch("out_class", &class_val);
    tr_data->Branch("out_idx_sector", &idx_sector);
    tr_data->Branch("out_idx_row", &idx_row);
    tr_data->Branch("out_idx_pad", &idx_pad);
    tr_data->Branch("out_idx_time", &idx_time);
    tr_data->Branch("cluster_pT", &pT);
    tr_data->Branch("cluster_eta", &eta);
    tr_data->Branch("cluster_mass", &mass);
    tr_data->Branch("cluster_p", &p);
    tr_data->Branch("cluster_isPrimary", &isPrimary);
    tr_data->Branch("cluster_isTagged", &isTagged);

    // Filling elements
    for (int element = 0; element < data_size; element++) {
      atomic_unit = tr_data_X[element];
      trY = tr_data_Y_reg[element];
      class_val = tr_data_Y_class[element];
      idx_sector = loop_sectors;
      idx_row = digit_map[maxima_digits[element]][0];
      idx_pad = digit_map[maxima_digits[element]][1];
      idx_time = digit_map[maxima_digits[element]][2];
      pT = cluster_pT[element];
      eta = cluster_eta[element];
      mass = cluster_mass[element];
      p = cluster_p[element];
      isPrimary = cluster_isPrimary[element];
      isTagged = cluster_isTagged[element];
      tr_data->Fill();
    }
    tr_data->Write();
    outputFileTrData->Close();

    tr_data_X.clear();
    tr_data_Y_reg.clear();
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

  LOG(info) << "--- Done with sector " << loop_sectors << " ---\n";
}

// ---------------------------------
void qaIdeal::run(ProcessingContext& pc)
{

  if (mode.find(std::string("looper_tagger")) != std::string::npos) {
    for (int i = 0; i < looper_tagger_granularity.size(); i++) {
      LOG(info) << "Looper tagger active, settings: granularity " << looper_tagger_granularity[i] << ", time-window: " << looper_tagger_timewindow[i] << ", pad-window: " << looper_tagger_padwindow[i] << ", threshold (num. points): " << looper_tagger_threshold_num[i] << ", threshold (Q): " << looper_tagger_threshold_q[i] << ", operational mode: " << looper_tagger_opmode;
    }
  }

  read_kinematics(mctracks);

  number_of_ideal_max.fill(0);
  number_of_digit_max.fill(0);
  number_of_ideal_max_findable.fill(0);
  clones.fill(0);
  fractional_clones.fill(0.f);

  // init array
  fill_nested_container(assignments_ideal, 0);
  fill_nested_container(assignments_digit, 0);
  fill_nested_container(assignments_ideal_findable, 0);
  fill_nested_container(assignments_digit_findable, 0);

  numThreads = std::min(numThreads, 36);
  thread_group group;
  if (numThreads > 1) {
    for (int loop_sectors : tpcSectors) {
      group.create_thread(boost::bind(&qaIdeal::runQa, this, loop_sectors));
      if ((loop_sectors + 1) % numThreads == 0 || loop_sectors + 1 == o2::tpc::constants::MAXSECTOR) {
        group.join_all();
      }
    }
  } else {
    for (int loop_sectors : tpcSectors) {
      runQa(loop_sectors);
    }
  }

  unsigned int number_of_ideal_max_sum = 0, number_of_digit_max_sum = 0, number_of_ideal_max_findable_sum = 0;
  float clones_sum = 0, fractional_clones_sum = 0;
  sum_nested_container(assignments_ideal, number_of_ideal_max_sum);
  sum_nested_container(assignments_digit, number_of_digit_max_sum);
  sum_nested_container(assignments_ideal_findable, number_of_ideal_max_findable_sum);
  sum_nested_container(clones, clones_sum);
  sum_nested_container(fractional_clones, fractional_clones_sum);

  LOG(info) << "------- RESULTS -------\n";
  LOG(info) << "Number of digit maxima: " << number_of_digit_max_sum;
  LOG(info) << "Number of ideal maxima (total): " << number_of_ideal_max_sum;
  LOG(info) << "Number of ideal maxima (findable): " << number_of_ideal_max_findable_sum << "\n";

  unsigned int efficiency_normal = 0;
  unsigned int efficiency_findable = 0;
  for (int ass = 0; ass < 25; ass++) {
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

  for (int ass = 0; ass < 25; ass++) {
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

  int fakes_dig = 0;
  for (int s = 0; s < o2::tpc::constants::MAXSECTOR; s++) {
    fakes_dig += assignments_digit[s][0];
  }
  int fakes_id = 0;
  for (int s = 0; s < o2::tpc::constants::MAXSECTOR; s++) {
    fakes_id += assignments_ideal[s][0];
  }

  LOG(info) << "Efficiency - Number of assigned (ideal -> digit) clusters: " << efficiency_normal << " (" << (float)efficiency_normal * 100 / (float)number_of_ideal_max_sum << "% of ideal maxima)";
  LOG(info) << "Efficiency (findable) - Number of assigned (ideal -> digit) clusters: " << efficiency_findable << " (" << (float)efficiency_findable * 100 / (float)number_of_ideal_max_findable_sum << "% of ideal maxima)";
  LOG(info) << "Clones (Int, clone-order >= 2 for ideal cluster): " << clones_sum << " (" << (float)clones_sum * 100 / (float)number_of_digit_max_sum << "% of digit maxima)";
  LOG(info) << "Clones (Float, fractional clone-order): " << fractional_clones_sum << " (" << (float)fractional_clones_sum * 100 / (float)number_of_digit_max_sum << "% of digit maxima)";
  LOG(info) << "Fakes for digits (number of digit hits that can't be assigned to any ideal hit): " << fakes_dig << " (" << (float)fakes_dig * 100 / (float)number_of_digit_max_sum << "% of digit maxima)";
  LOG(info) << "Fakes for ideal (number of ideal hits that can't be assigned to any digit hit): " << fakes_id << " (" << (float)fakes_id * 100 / (float)number_of_ideal_max_sum << "% of ideal maxima)";

  if (mode.find(std::string("looper_tagger")) != std::string::npos && create_output == 1) {
    LOG(info) << "------- Merging looper tagger regions -------";
    gSystem->Exec("hadd -k -f ./looper_tagger.root ./looper_tagger_*.root");
  }

  if (mode.find(std::string("training_data")) != std::string::npos && create_output == 1) {
    LOG(info) << "------- Merging training data -------";
    gSystem->Exec("hadd -k -f ./training_data.root ./training_data_*.root");
  }

  if (mode.find(std::string("native")) != std::string::npos && create_output == 1) {
    LOG(info) << "------- Merging native-ideal assignments -------";
    gSystem->Exec("hadd -k -f ./native_ideal.root ./native_ideal_*.root");
  }

  if (mode.find(std::string("network")) != std::string::npos && create_output == 1) {
    LOG(info) << "------- Merging network-ideal assignments -------";
    gSystem->Exec("hadd -k -f ./network_ideal.root ./network_ideal_*.root");
  }

  if (create_output == 1 && write_native_file == 1) {

    if (mode.find(std::string("network")) != std::string::npos) {
      write_custom_native(pc, native_writer_map);

      // LOG(info) << "------- Merging tpc-native-clusters-network_*.root files -------";
      // gSystem->Exec("hadd -k -f ./tpc-native-clusters-network.root ./tpc-native-clusters-network_*.root");
      // gSystem->Exec("rm -rf ./tpc-native-clusters-network_*.root");
    }
    if (mode.find(std::string("native")) != std::string::npos) {
      write_custom_native(pc, native_writer_map);
      // LOG(info) << "------- Merging tpc-native-clusters-native_*.root files -------";
      // gSystem->Exec("hadd -k -f ./tpc-native-clusters-native.root ./tpc-native-clusters-native_*.root");
      // gSystem->Exec("rm -rf ./tpc-native-clusters-native_*.root");
    }
  }

  if (remove_individual_files > 0) {
    LOG(info) << "!!! Removing sector-individual files !!!";
    gSystem->Exec("rm -rf ./looper_tagger_*.root");
    gSystem->Exec("rm -rf ./training_data_*.root");
    gSystem->Exec("rm -rf ./native_ideal_*.root");
    gSystem->Exec("rm -rf ./network_ideal_*.root");
  }

  pc.services().get<ControlService>().endOfStream();
  pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
}


// ----- Input and output processors -----

// customize clusterers and cluster decoders to process immediately what comes in
void customize(std::vector<o2::framework::CompletionPolicy>& policies) {
  policies.push_back(o2::framework::CompletionPolicyHelpers::consumeWhenAllOrdered(".*(?:TPC|tpc).*[w,W]riter.*"));
}

void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  std::vector<ConfigParamSpec> options{
    {"tpc-sectors", VariantType::String, "0-35", {"TPC sector range, e.g. 5-7,8,9"}},
    {"write-native-file", VariantType::Int, 0, {"Whether or not to write a custom native file"}},
    {"native-file-single-branch", VariantType::Int, 1, {"Whether or not to write a single branch in the custom native file"}}
  };
  std::swap(workflowOptions, options);
}

// ---------------------------------
#include "Framework/runDataProcessing.h"

DataProcessorSpec processIdealClusterizer(ConfigContext const& cfgc, std::vector<InputSpec>& inputs, std::vector<OutputSpec>& outputs)
{

  // A copy of the global workflow options from customize() to pass to the task
  std::unordered_map<std::string, std::string> options_map{
    {"tpc-sectors" , cfgc.options().get<std::string>("tpc-sectors")},
    {"write-native-file" , cfgc.options().get<std::string>("write-native-file")},
    {"native-file-single-branch" , cfgc.options().get<std::string>("native-file-single-branch")},
  };

  if(cfgc.options().get<int>("write-native-file")){
    // setOutputAllocator("CLUSTERNATIVE", true, outputRegions.clustersNative, std::make_tuple(gDataOriginTPC, mSpecConfig.sendClustersPerSector ? (DataDescription) "CLUSTERNATIVETMP" : (DataDescription) "CLUSTERNATIVE", NSectors, clusterOutputSectorHeader), sizeof(o2::tpc::ClusterCountIndex));
    for (int i = 0; i < o2::tpc::constants::MAXSECTOR; i++) {
      outputs.emplace_back(o2::header::gDataOriginTPC, "CLUSTERNATIVE", i, Lifetime::Timeframe); // Dropping incomplete Lifetime::Transient?
      outputs.emplace_back(o2::header::gDataOriginTPC, "CLNATIVEMCLBL", i, Lifetime::Timeframe); // Dropping incomplete Lifetime::Transient?
    }
  }

  return DataProcessorSpec{
    "tpc-qa-ideal",
    inputs,
    outputs,
    adaptFromTask<qaIdeal>(options_map),
    Options{
      {"verbose", VariantType::Int, 0, {"Verbosity level"}},
      {"mode", VariantType::String, "training_data", {"Enables different settings (e.g. creation of training data for NN, running with tpc-native clusters). Options are: training_data, native, network_classification, network_regression, network_full, clusterizer"}},
      {"normalization-mode", VariantType::Int, 1, {"Normalization: 0 = normalization by 1024.f; 1 = normalization by q_center"}},
      {"create-output", VariantType::Int, 1, {"Create output, specific to any given mode."}},
      {"use-max-cog", VariantType::Int, 1, {"Use maxima for assignment = 0, use CoG's = 1"}},
      {"size-pad", VariantType::Int, 11, {"Training data selection size: Images are (size-pad, size-time, size-row)."}},
      {"size-time", VariantType::Int, 11, {"Training data selection size: Images are (size-pad, size-time, size-row)."}},
      {"size-row", VariantType::Int, 1, {"Training data selection size: Images are (size-pad, size-time, size-row)."}},
      {"threads", VariantType::Int, 1, {"Number of CPU threads to be used."}},
      {"looper-tagger-opmode", VariantType::String, "ideal", {"Mode in which the looper tagger is run: ideal or digit."}},
      {"looper-tagger-granularity", VariantType::ArrayInt, std::vector<int>{5}, {"Granularity of looper tagger (time bins in which loopers are excluded in rectangular areas)."}}, // Needs to be called with e.g. --looper-tagger-granularity [2,3]
      {"looper-tagger-padwindow", VariantType::ArrayInt, std::vector<int>{3}, {"Total pad-window size of the looper tagger for evaluating if a region is looper or not."}},
      {"looper-tagger-timewindow", VariantType::ArrayInt, std::vector<int>{20}, {"Total time-window size of the looper tagger for evaluating if a region is looper or not."}},
      {"looper-tagger-threshold-num", VariantType::ArrayInt, std::vector<int>{5}, {"Threshold of number of clusters over which rejection takes place."}},
      {"looper-tagger-threshold-q", VariantType::ArrayFloat, std::vector<float>{70.f}, {"Threshold of charge-per-cluster that should be rejected."}},
      {"infile-digits", VariantType::String, "tpcdigits.root", {"Input file name (digits)"}},
      {"infile-native", VariantType::String, "tpc-native-clusters.root", {"Input file name (native)"}},
      {"infile-kinematics", VariantType::String, "collisioncontext.root", {"Input file name (kinematics)"}},
      {"network-data-output", VariantType::String, "network_out.root", {"Input file for the network output"}},
      {"network-classification-path", VariantType::String, "./net_classification.onnx", {"Absolute path to the network file (classification)"}},
      {"network-regression-path", VariantType::String, "./net_regression.onnx", {"Absolute path to the network file (regression)"}},
      {"network-input-size", VariantType::Int, 1000, {"Size of the vector to be fed through the neural network"}},
      {"network-class-threshold", VariantType::Float, 0.5f, {"Threshold for classification network: Keep or reject maximum (default: 0.5)"}},
      {"enable-network-optimizations", VariantType::Bool, true, {"Enable ONNX network optimizations"}},
      {"network-num-threads", VariantType::Int, 1, {"Set the number of CPU threads for network execution"}},
      {"remove-individual-files", VariantType::Int, 0, {"Remove sector-individual files that are created during the task and only keep merged files"}},
    }
  };
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
  specs.push_back(processIdealClusterizer(cfgc, inputs, outputs));

  // Native writer
  if(cfgc.options().get<int>("write-native-file")){
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
          def.keys = mergeInputs(def.keys, o2::tpc::constants::MAXSECTOR, amendInput);
          // the branch is disabled if set to 0
          def.nofBranches = o2::tpc::constants::MAXSECTOR;
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
                                  "tpc-native-clusters-custom.root",
                                  "tpcrec",
                                  BranchDefinition<const char*>{InputSpec{"data", ConcreteDataTypeMatcher{"TPC", o2::header::DataDescription("CLUSTERNATIVE")}},
                                                                "TPCClusterNative",
                                                                "databranch"},
                                  BranchDefinition<std::vector<char>>{InputSpec{"mc", ConcreteDataTypeMatcher{"TPC", o2::header::DataDescription("CLNATIVEMCLBL")}},
                                                                      "TPCClusterNativeMCTruth",
                                                                      "mcbranch", fillLabels}));
  };

  return specs;
}