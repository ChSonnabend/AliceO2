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

typedef std::array<std::vector<std::vector<std::vector<int>>>, 2> tpc2d; // local 2D charge map, 0 - digits; 1 - ideal

namespace o2
{
namespace tpc
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


class qaCluster : public Task
{
 public:
  template <typename T>
  int sign(T);

  qaCluster(std::unordered_map<std::string, std::string>);
  void init(InitContext&) final;
  void setGeomFromTxt(std::string, std::vector<int> = {152, 14});
  int padOffset(int);
  int rowOffset(int);
  bool isBoundary(int, int);
  bool checkIdx(int);

  // Readers
  void read_digits(int, std::vector<customCluster>&);
  void read_ideal(int, std::vector<customCluster>&);
  void read_native(int, std::vector<customCluster>&, std::vector<customCluster>&);
  void read_kinematics(std::vector<std::vector<std::vector<o2::MCTrack>>>&);

  // Writers
  void write_custom_native(ProcessingContext&, std::vector<customCluster>&);
  void write_tabular_data();

  tpc2d init_map2d(int);

  void fill_map2d(int, tpc2d&, std::vector<customCluster>&, std::vector<customCluster>&, int = 0);
  void find_maxima(int, tpc2d&, std::vector<customCluster>&, std::vector<int>&);

  bool is_local_minimum(tpc2d&, std::array<int, 3>&, std::vector<float>&);
  int local_saddlepoint(tpc2d&, std::array<int, 3>&, std::vector<float>&);
  void native_clusterizer(tpc2d&, std::vector<std::array<int, 3>>&, std::vector<int>&, std::vector<float>&, std::vector<std::array<float, 3>>&, std::vector<float>&);

  std::vector<std::vector<std::vector<int>>> looper_tagger(int, int, std::vector<customCluster>&, std::vector<int>&);

  void remove_loopers_digits(int, std::vector<std::vector<std::vector<int>>>&, std::vector<customCluster>&, std::vector<int>&);
  void remove_loopers_native(int, std::vector<std::vector<std::vector<int>>>&, std::vector<customCluster>&, std::vector<int>&);
  void remove_loopers_ideal(int, std::vector<std::vector<std::vector<int>>>&, std::vector<customCluster>&);

  std::tuple<std::vector<float>, std::vector<uint8_t>> create_network_input(int, tpc2d&, std::vector<int>&, std::vector<customCluster>&);
  void run_network_classification(int, tpc2d&, std::vector<int>&, std::vector<customCluster>&, std::vector<customCluster>&);
  void run_network_regression(int, tpc2d&, std::vector<int>&, std::vector<customCluster>&, std::vector<customCluster>&);
  void overwrite_map2d(int, tpc2d&, std::vector<customCluster>&, std::vector<int>&, int = 0);

  int test_neighbour(std::array<int, 3>, std::array<int, 2>, tpc2d&, int = 1);

  void runQa(int);
  void run(ProcessingContext&) final;

 private:
  std::vector<int> tpc_sectors; // The TPC sectors for which processing should be started

  std::vector<int> global_shift = {5, 5, 0}; // shifting digits to select windows easier, (pad, time, row)
  int charge_limits[2] = {2, 1024};          // upper and lower charge limits
  int verbose = 0;                           // chunk_size in time direction
  int create_output = 1;                     // Create output files specific for any mode
  int dim = 2;                               // Dimensionality of the training data
  int networkInputSize = 1000;               // vector input size for neural network
  float networkClassThres = 0.5f;            // Threshold where network decides to keep / reject digit maximum
  int networkNumThreads = 1;                 // Future: Add Cuda and CoreML Execution providers to run on CPU
  int numThreads = 1;                        // Number of cores for multithreading
  int use_max_cog = 1;                       // 0 = use ideal maxima position; 1 = use ideal CoG position (rounded) for assignment
  float threshold_cogq = 5.f;                // Threshold for ideal cluster to be findable (Q_tot)
  float threshold_maxq = 3.f;                // Threshold for ideal cluster to be findable (Q_max)
  int normalization_mode = 1;                // Normalization of the charge: 0 = divide by 1024; 1 = divide by central charge
  int remove_individual_files = 0;           // Remove sector-individual files after task is done
  bool write_native_file = 1;                // Whether or not to write a custom file with native clsuters
  bool native_file_single_branch = 1;        // Whether the native clusters should be written into a single branch

  std::vector<int> looper_tagger_granularity = {5};      // Granularity of looper tagger (time bins in which loopers are excluded in rectangular areas)
  std::vector<int> looper_tagger_timewindow = {20};      // Total time-window size of the looper tagger for evaluating if a region is looper or not
  std::vector<int> looper_tagger_padwindow = {3};        // Total pad-window size of the looper tagger for evaluating if a region is looper or not
  std::vector<int> looper_tagger_threshold_num = {5};    // Threshold of number of clusters over which rejection takes place
  std::vector<float> looper_tagger_threshold_q = {70.f}; // Threshold of charge-per-cluster that should be rejected
  std::string looper_tagger_opmode = "digit";            // Operational mode of the looper tagger

  bool overwrite_max_time = true;
  std::array<int, o2::tpc::constants::MAXSECTOR> max_time;
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

  int networkOptimizations = 1;
  std::vector<std::string> network_classification_paths, network_regression_paths;
  OnnxModel network_classification[5], network_regression[5];

  int num_total_ideal_max = 0, num_total_digit_max = 0;
  std::vector<std::vector<std::vector<o2::MCTrack>>> mctracks; // mc_track = mctracks[sourceId][eventId][trackId]
  std::array<std::array<unsigned int, 25>, o2::tpc::constants::MAXSECTOR> assignments_ideal, assignments_digit, assignments_ideal_findable, assignments_digit_findable;
  std::array<unsigned int, o2::tpc::constants::MAXSECTOR> number_of_ideal_max, number_of_digit_max, number_of_ideal_max_findable;
  std::array<float, o2::tpc::constants::MAXSECTOR> clones, fractional_clones;
  std::vector<customCluster> native_writer_map;
  std::mutex m;

};

namespace custom
{
std::vector<std::string> splitString(const std::string& input, const std::string& delimiter) {
    std::vector<std::string> tokens;
    std::size_t pos = 0;
    std::size_t found;

    while ((found = input.find(delimiter, pos)) != std::string::npos) {
        tokens.push_back(input.substr(pos, found - pos));
        pos = found + delimiter.length();
    }
    tokens.push_back(input.substr(pos));

    return tokens;
  }

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

  std::vector<int> hasElementAppearedMoreThanNTimesInVectors(const std::vector<std::vector<int>>& index_vector, const std::vector<customCluster>& assignment_vector, int n)
  {
    std::unordered_map<int, int> elementCount;
    std::vector<int> return_labels;
    int vector_resize;
    for (const std::vector<int>& vec : index_vector) {
      for (int element : vec) {
        elementCount[assignment_vector[element].mcTrkId]++;
      }
    }
    for (auto elem : elementCount) {
      if (elem.second >= n) {
        return_labels.push_back(elem.first);
      }
    }

    return return_labels;
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

  template <typename T>
  struct is_exact_container : std::conditional_t<(is_vector<T>::value || is_array<T>::value) && !is_vector<T[0]>::value && is_array<T[0]>::value, std::true_type, std::false_type> {
  };

  template<typename Container>
  void fill_container_by_range(Container& destination, const Container& source, int startIndex, int endIndex, bool flexible = true) {
    // Clear the destination vector
    destination.clear();

    // Validate startIndex and endIndex
    if(flexible){
      if(endIndex == -1 || endIndex >= source.size()){
        endIndex = source.size() - 1;
      }
      if(startIndex < 0){
        startIndex = 0;
      }
    }
    else if (startIndex < 0 || endIndex >= source.size() || startIndex > endIndex) {
        // Handle invalid range
        return;
    }

    destination.resize(endIndex - startIndex + 1);
    std::copy(source.begin() + startIndex, source.begin() + endIndex + 1, destination.begin());
  }

  template<typename Container>
  void append_to_container(Container& destination, const Container& source, int startIndex = -1, int endIndex = -1) {
    int destination_initial_size = destination.size();
    if(endIndex == -1 || endIndex >= source.size()){
      endIndex = source.size() - 1;
    }
    if(startIndex < 0){
      startIndex = 0;
    }
    destination.resize(destination_initial_size + endIndex - startIndex + 1);
    std::copy(source.begin() + startIndex, source.begin() + endIndex + 1, destination.begin() + destination_initial_size);
  }

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

  template<typename Container, typename ValueType>
  void container_full_size_internal(const Container& input, std::size_t& size) {
      size += input.size();

      for (const auto& elem : input) {
          if constexpr (is_exact_container<Container>::value) {
            size += elem.size();
          } else {
            size += container_full_size_internal(elem);
          }
      }
  }

  template<typename Container, typename ValueType>
  std::size_t container_full_size(const Container& input) {
      std::size_t size = 0;

      if constexpr (is_container<Container>::value){
      for (const auto& elem : input) {
          container_full_size_internal(elem, size);
      }
      } else {
        size = 1;
      }

      return size;
  }

  template<typename Container>
  void flatten_internal(Container& input, Container& output, size_t& index) {
      for (const auto& elem : input) {
        if constexpr (is_exact_container<decltype(elem)>::value) {
          std::copy(output.begin()+index, elem.begin(), elem.end());
          index += elem.size();
        } else {
          flatten_internal(elem, output, index);
        }
      }
  }

  template<typename Container, typename ValueType>
  void flatten(Container& input) {
      if constexpr (!is_exact_container<Container>::value) {
        std::vector<ValueType> output(container_full_size(input));
        size_t index = 0;
        for (const auto& innerVec : input) {
            flatten_internal(innerVec, output, index);
        }
        input = output;
      }
  }

  // Helper function for resizing a nested container
  template <typename Container>
  void resize_nested_container(Container& container, const std::vector<size_t>& size, const double& value = 0)
  {
    if constexpr (is_container<Container>::value) {
      if constexpr (is_vector<Container>::value) {
        container.resize(size[0]);
      }
      std::vector<size_t> nested_size(size.size() - 1);
      std::copy(size.begin() + 1, size.end(), nested_size.begin());
      for (auto& elem : container) {
        resize_nested_container(elem, nested_size, value);
      }
    } else {
      container = (decltype(container))value;
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

  template<class T>
  void WriteClustersToRootFile(std::string fileName, std::vector<T>& clusters) {
    // Create a TFile
    TFile *file = new TFile(fileName.c_str(), "RECREATE");
    if (!file || file->IsZombie()) {
        printf("Error: Cannot create file %s\n", fileName);
        return;
    }

    // Create a TTree
    TTree *tree = new TTree("customClusters", "Tree with customCluster struct data");

    // Create a branch and associate the struct with it
    T elem;
    TBranch *branch = tree->Branch("customClusters", &elem);

    // Fill the tree with some example data
    for (int i = 0; i < clusters.size(); ++i) {
      elem = clusters[i];
        tree->Fill();
    }

    // Write the tree to the file
    tree->Write();
    file->Close();
}

template<class T>
void ReadClustersFromRootFile(std::string fileName, std::vector<T>& clusters) {
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
}
}
// ---------------------------------
template <typename T>
int qaCluster::sign(T val)
{
  return (T(0) < val) - (val < T(0));
}

// ---------------------------------
void qaCluster::setGeomFromTxt(std::string inputFile, std::vector<int> size)
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
int qaCluster::padOffset(int row)
{
  return (int)((TPC_GEOM[o2::tpc::constants::MAXGLOBALPADROW-1][2] - TPC_GEOM[row][2]) / 2);
}

// ---------------------------------
int qaCluster::rowOffset(int row)
{
  if (row <= 62) {
    return 0;
  } else {
    return global_shift[2];
  }
}

// ---------------------------------
bool qaCluster::isBoundary(int row, int pad)
{
  if (row < 0 || pad < 0) {
    return true;
  } else if (row <= 62) {
    if (pad < (TPC_GEOM[o2::tpc::constants::MAXGLOBALPADROW-1][2] - TPC_GEOM[row][2]) / 2 || pad > (TPC_GEOM[o2::tpc::constants::MAXGLOBALPADROW-1][2] + TPC_GEOM[row][2]) / 2) {
      return true;
    } else {
      return false;
    }
  } else if (row <= 62 + global_shift[2]) {
    return true;
  } else if (row <= o2::tpc::constants::MAXGLOBALPADROW-1 + global_shift[2]) {
    if (pad < (TPC_GEOM[o2::tpc::constants::MAXGLOBALPADROW-1][2] - TPC_GEOM[row - global_shift[2]][2]) / 2 || pad > (TPC_GEOM[o2::tpc::constants::MAXGLOBALPADROW-1][2] + TPC_GEOM[row - global_shift[2]][2]) / 2) {
      return true;
    } else {
      return false;
    }
  } else if (row > o2::tpc::constants::MAXGLOBALPADROW-1 + global_shift[2]) {
    return true;
  } else {
    return false;
  }
}
}
}