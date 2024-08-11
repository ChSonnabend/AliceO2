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
  float X = -1.f;
  float Y = -1.f;
  float Z = -1.f;

  ~customCluster(){}

  std::vector<std::pair<std::string, std::string>> getMemberMap() const {
    return {
      {"sector", typeid(sector).name()},
      {"row", typeid(row).name()},
      {"max_pad", typeid(max_pad).name()},
      {"max_time", typeid(max_time).name()},
      {"cog_pad", typeid(cog_pad).name()},
      {"cog_time", typeid(cog_time).name()},
      {"sigmaPad", typeid(sigmaPad).name()},
      {"sigmaTime", typeid(sigmaTime).name()},
      {"qMax", typeid(qMax).name()},
      {"qTot", typeid(qTot).name()},
      {"flag", typeid(flag).name()},
      {"mcTrkId", typeid(mcTrkId).name()},
      {"mcEvId", typeid(mcEvId).name()},
      {"mcSrcId", typeid(mcSrcId).name()},
      {"index", typeid(index).name()},
      {"label", typeid(label).name()},
      {"X", typeid(X).name()},
      {"Y", typeid(Y).name()},
      {"Z", typeid(Z).name()}
    };
  }
};

class TPCMap
{
  public:
    int GetRegion(int row) const { return mRegion[row]; }
    int GetRegionRows(int region) const { return mRegionRows[region]; }
    int GetRegionStart(int region) const { return mRegionStart[region]; }
    int GetSampaMapping(int region) const { return mSampaMapping[region]; }
    int GetChannelOffset(int region) const { return mChannelOffset[region]; }
    int GetSectorFECOffset(int partition) const { return mSectorFECOffset[partition]; }
    int GetROC(int row) const { return row < 97 ? (row < 63 ? 0 : 1) : (row < 127 ? 2 : 3); }
    float GetTimeBoundary() const { return T_BOUNDARY; }
    int EndIROC() const { return 63; }
    int EndOROC1() const { return 97; }
    int EndOROC2() const { return 127; }

    float TPCLength() { return 250.f - 0.275f; }
    float Row2X(int row) const { return (mX[row]); }
    float PadHeight(int row) const { return (mPadHeight[GetRegion(row)]); }
    float PadHeightByRegion(int region) const { return (mPadHeight[region]); }
    float PadWidth(int row) const { return (mPadWidth[GetRegion(row)]); }
    int NPads(int row) const { return mNPads[row]; }

    float LinearPad2Y(int slice, int row, float pad) const
    {
      float u = (pad - 0.5 * mNPads[row]) * PadWidth(row);
      return (slice >= o2::tpc::constants::MAXSECTOR / 2) ? -u : u;
    }

    float LinearTime2Z(int slice, float time)
    {
      float v = T_BOUNDARY - time * FACTOR_T2Z; // Used in compression, must remain constant at 250cm!
      return (slice >= o2::tpc::constants::MAXSECTOR / 2) ? -v : v;
    }

    float LinearY2Pad(int slice, int row, float y) const
    {
      float u = (slice >= o2::tpc::constants::MAXSECTOR / 2) ? -y : y;
      return u / PadWidth(row) + 0.5 * mNPads[row];
    }

    float LinearZ2Time(int slice, float z)
    {
      float v = (slice >= o2::tpc::constants::MAXSECTOR / 2) ? -z : z;
      return (T_BOUNDARY - v) * FACTOR_Z2T; // Used in compression, must remain constant at 250cm
    }

    std::vector<std::array<float, 4>> getSectorsXY() // From /data.local1/csonnab/MyO2/O2/Detectors/TPC/base/src/Painter.cxx:692
    {
      constexpr float phiWidth = float(SECPHIWIDTH);
      const float rFactor = std::cos(phiWidth / 2.);
      const float rLow = 83.65 / rFactor;
      const float rIROCup = 133.3 / rFactor;
      const float rOROClow = 133.5 / rFactor;
      const float rOROC12 = 169.75 / rFactor;
      const float rOROC23 = 207.85 / rFactor;
      const float rOut = 247.7 / rFactor;
      const float rText = rLow * rFactor * 3. / 4.;

      std::vector<std::array<float, 4>> tpc_xy_boundaries;

      for (Int_t isector = 0; isector < 18; ++isector) {
        const float sinText = std::sin(phiWidth * (isector + 0.5));
        const float cosText = std::cos(phiWidth * (isector + 0.5));

        const float xText = rText * cosText;
        const float yText = rText * sinText;

        tpc_xy_boundaries.push_back({xText, yText, 0, 0});
      }

      for (Int_t isector = 0; isector < 18; ++isector) {
        const float sinR = std::sin(phiWidth * isector);
        const float cosR = std::cos(phiWidth * isector);

        const float sinL = std::sin(phiWidth * ((isector + 1) % 18));
        const float cosL = std::cos(phiWidth * ((isector + 1) % 18));

        const float xR1 = rLow * cosR;
        const float yR1 = rLow * sinR;
        const float xR2 = rOut * cosR;
        const float yR2 = rOut * sinR;

        const float xL1 = rLow * cosL;
        const float yL1 = rLow * sinL;
        const float xL2 = rOut * cosL;
        const float yL2 = rOut * sinL;

        const float xOROCmup1 = rOROClow * cosR;
        const float yOROCmup1 = rOROClow * sinR;
        const float xOROCmup2 = rOROClow * cosL;
        const float yOROCmup2 = rOROClow * sinL;

        const float xIROCmup1 = rIROCup * cosR;
        const float yIROCmup1 = rIROCup * sinR;
        const float xIROCmup2 = rIROCup * cosL;
        const float yIROCmup2 = rIROCup * sinL;

        const float xO121 = rOROC12 * cosR;
        const float yO121 = rOROC12 * sinR;
        const float xO122 = rOROC12 * cosL;
        const float yO122 = rOROC12 * sinL;

        const float xO231 = rOROC23 * cosR;
        const float yO231 = rOROC23 * sinR;
        const float xO232 = rOROC23 * cosL;
        const float yO232 = rOROC23 * sinL;

        tpc_xy_boundaries.push_back({xR1, yR1, xR2, yR2});
        tpc_xy_boundaries.push_back({xR1, yR1, xL1, yL1});
        tpc_xy_boundaries.push_back({xIROCmup1, yIROCmup1, xIROCmup2, yIROCmup2});
        tpc_xy_boundaries.push_back({xOROCmup1, yOROCmup1, xOROCmup2, yOROCmup2});
        tpc_xy_boundaries.push_back({xO121, yO121, xO122, yO122});
        tpc_xy_boundaries.push_back({xO231, yO231, xO232, yO232});
        tpc_xy_boundaries.push_back({xR2, yR2, xL2, yL2});
      }
      return tpc_xy_boundaries;
    }

  private:
    const std::vector<float> mX = {85.225f, 85.975f, 86.725f, 87.475f, 88.225f, 88.975f, 89.725f, 90.475f, 91.225f, 91.975f, 92.725f, 93.475f, 94.225f, 94.975f, 95.725f, 96.475f, 97.225f, 97.975f, 98.725f, 99.475f, 100.225f, 100.975f,
                                                        101.725f, 102.475f, 103.225f, 103.975f, 104.725f, 105.475f, 106.225f, 106.975f, 107.725f, 108.475f, 109.225f, 109.975f, 110.725f, 111.475f, 112.225f, 112.975f, 113.725f, 114.475f, 115.225f, 115.975f, 116.725f, 117.475f,
                                                        118.225f, 118.975f, 119.725f, 120.475f, 121.225f, 121.975f, 122.725f, 123.475f, 124.225f, 124.975f, 125.725f, 126.475f, 127.225f, 127.975f, 128.725f, 129.475f, 130.225f, 130.975f, 131.725f, 135.2f, 136.2f, 137.2f,
                                                        138.2f, 139.2f, 140.2f, 141.2f, 142.2f, 143.2f, 144.2f, 145.2f, 146.2f, 147.2f, 148.2f, 149.2f, 150.2f, 151.2f, 152.2f, 153.2f, 154.2f, 155.2f, 156.2f, 157.2f, 158.2f, 159.2f,
                                                        160.2f, 161.2f, 162.2f, 163.2f, 164.2f, 165.2f, 166.2f, 167.2f, 168.2f, 171.4f, 172.6f, 173.8f, 175.f, 176.2f, 177.4f, 178.6f, 179.8f, 181.f, 182.2f, 183.4f, 184.6f, 185.8f,
                                                        187.f, 188.2f, 189.4f, 190.6f, 191.8f, 193.f, 194.2f, 195.4f, 196.6f, 197.8f, 199.f, 200.2f, 201.4f, 202.6f, 203.8f, 205.f, 206.2f, 209.65f, 211.15f, 212.65f, 214.15f, 215.65f,
                                                        217.15f, 218.65f, 220.15f, 221.65f, 223.15f, 224.65f, 226.15f, 227.65f, 229.15f, 230.65f, 232.15f, 233.65f, 235.15f, 236.65f, 238.15f, 239.65f, 241.15f, 242.65f, 244.15f, 245.65f};

    const  std::vector<int> mNPads = {66, 66, 66, 68, 68, 68, 70, 70, 70, 72, 72, 72, 74, 74, 74, 74, 76, 76, 76, 76, 78, 78, 78, 80, 80, 80, 82, 82, 82, 84, 84, 84, 86, 86, 86, 88, 88, 88,
                                                                    90, 90, 90, 90, 92, 92, 92, 94, 94, 94, 92, 92, 92, 94, 94, 94, 96, 96, 96, 98, 98, 98, 100, 100, 100, 76, 76, 76, 76, 78, 78, 78, 80, 80, 80, 80, 82, 82,
                                                                    82, 84, 84, 84, 84, 86, 86, 86, 88, 88, 88, 90, 90, 90, 90, 92, 92, 92, 94, 94, 94, 94, 96, 96, 96, 98, 98, 98, 100, 100, 102, 102, 102, 104, 104, 104, 106, 110,
                                                                    110, 112, 112, 112, 114, 114, 114, 116, 116, 116, 118, 118, 118, 118, 118, 120, 120, 122, 122, 124, 124, 124, 126, 126, 128, 128, 128, 130, 130, 132, 132, 132, 134, 134, 136, 136, 138, 138};

    const std::vector<int> mRegion = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                                                                    4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9};
    const std::vector<int> mRegionRows = {17, 15, 16, 15, 18, 16, 16, 14, 13, 12};
    const std::vector<int> mRegionStart = {0, 17, 32, 48, 63, 81, 97, 113, 127, 140};

    const std::vector<int> mSampaMapping = {0, 0, 1, 1, 2, 3, 3, 4, 4, 2};
    const std::vector<int> mChannelOffset = {0, 16, 0, 16, 0, 0, 16, 0, 16, 16};
    const std::vector<int> mSectorFECOffset = {0, 15, 15 + 18, 15 + 18 + 18, 15 + 18 + 18 + 20};

    const std::vector<float> mPadHeight = {.75f, .75f, .75f, .75f, 1.f, 1.f, 1.2f, 1.2f, 1.5f, 1.5f};
    const std::vector<float> mPadWidth = {.416f, .420f, .420f, .436f, .6f, .6f, .608f, .588f, .604f, .607f};

    const float T_BOUNDARY = 250.f;
    const float FACTOR_T2Z = 250.f / 512.f;
    const float FACTOR_Z2T = 1.f / FACTOR_T2Z;
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
  void read_tracking_clusters(bool = true);

  // Writers
  void write_custom_native(ProcessingContext&, std::vector<customCluster>&, bool = true);
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

  std::tuple<std::vector<float>, std::vector<uint8_t>> create_network_input(int, tpc2d&, std::vector<int>&, std::vector<customCluster>&, int);
  void run_network_classification(int, tpc2d&, std::vector<int>&, std::vector<customCluster>&, std::vector<customCluster>&);
  void run_network_regression(int, tpc2d&, std::vector<int>&, std::vector<customCluster>&, std::vector<customCluster>&, std::vector<std::array<float,2>>&);
  void overwrite_map2d(int, tpc2d&, std::vector<customCluster>&, std::vector<int>&, int = 0);

  int test_neighbour(std::array<int, 3>, std::array<int, 2>, tpc2d&, int = 1);

  void runQa(int);
  void run(ProcessingContext&) final;

 private:
  TPCMap tpcmap;
  std::shared_ptr<o2::base::GRPGeomRequest> grp_geom;

  std::vector<int> tpc_sectors;              // The TPC sectors for which processing should be started
  std::vector<int> global_shift = {5, 5, 0}; // shifting digits to select windows easier, (pad, time, row)
  int charge_limits[2] = {2, 1024};          // upper and lower charge limits
  int verbose = 0;                           // chunk_size in time direction
  int create_output = 1;                     // Create output files specific for any mode
  int dim = 2;                               // Dimensionality of the training data
  int networkInputSize = 1000;               // vector input size for neural network
  float networkClassThres = 0.5f;            // Threshold where network decides to keep / reject digit maximum
  int networkNumThreads = 1;                 // Future: Add Cuda and CoreML Execution providers to run on CPU
  bool networkSplitIrocOroc = 0;             // Whether or not to split the used networks for IROC and OROC's
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
  float training_data_distance_cluster_path = 2.f;

  bool overwrite_max_time = true;
  std::array<int, o2::tpc::constants::MAXSECTOR> max_time;
  std::string mode = "training_data";
  int realData = 0;
  std::string simulationPath = ".";
  std::string outputPath = ".";
  std::string inFileDigits = "tpcdigits.root";
  std::string inFileNative = "tpc-cluster-native.root";
  std::string inFileDigitizer = "mclabels_digitizer.root";
  std::string inFileKinematics = "collisioncontext.root";
  std::string inFileTracks = "tpctracks.root";
  std::string networkDataOutput = "network_out.root";
  std::string networkClassification = "net_classification.onnx";
  std::string networkRegression = "net_regression.onnx";
  std::string outCustomNative = "tpc-cluster-native-custom.root";
  std::string outFileCustomClusters = "custom-clusters.root";

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

  // Training data -> Momentum vector assignment
  std::array<std::vector<std::array<float,3>>, o2::tpc::constants::MAXSECTOR> momentum_vectors;
  std::array<std::vector<customCluster>, o2::tpc::constants::MAXSECTOR> tracking_clusters, tracking_paths;
  // std::array<std::vector<int>, o2::tpc::constants::MAXSECTOR> track_cluster_to_ideal_assignment; // index vector: track_cluster_to_ideal_assignment[ideal_idx] = track_cluster_idx
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

  GlobalPosition2D convertSecRowPadToXY(int sector, int row, float pad, TPCMap tpcmap){

    const auto& mapper = Mapper::instance();

    if(row > o2::tpc::constants::MAXGLOBALPADROW || sector > o2::tpc::constants::MAXSECTOR){
      LOG(warning) << "Stepping over boundary: " << sector << " / " << o2::tpc::constants::MAXSECTOR << ", " << row << " / " << o2::tpc::constants::MAXGLOBALPADROW;
    }

    GlobalPosition2D pos = mapper.getPadCentre(PadSecPos(sector, row, pad));
    float fractionalPad = 0;
    if(int(pad) != float(pad)){
      fractionalPad = mapper.getPadRegionInfo(tpcmap.GetRegion(row)).getPadWidth()*(pad - int(pad) - 0.5);
    }
    return GlobalPosition2D(pos.X(), pos.Y());
  }

  GlobalPosition3D convertSecRowPadToXY(int sector, int row, float pad, float z, TPCMap tpcmap){

    const auto& mapper = Mapper::instance();

    if(row > o2::tpc::constants::MAXGLOBALPADROW || sector > o2::tpc::constants::MAXSECTOR){
      LOG(warning) << "Stepping over boundary: " << sector << " / " << o2::tpc::constants::MAXSECTOR << ", " << row << " / " << o2::tpc::constants::MAXGLOBALPADROW;
    }

    GlobalPosition2D pos = mapper.getPadCentre(PadSecPos(sector, row, pad));
    float fractionalPad = 0;
    if(int(pad) != float(pad)){
      fractionalPad = mapper.getPadRegionInfo(tpcmap.GetRegion(row)).getPadWidth()*(pad - int(pad) - 0.5);
    }
    return GlobalPosition3D(pos.X(), pos.Y(), z);
  }

  // Write function for vector of customCluster
  void writeStructToRootFile(std::string filename, std::string treeName, const std::vector<customCluster>& data) {
      if (data.empty()) {
          std::cerr << "Error: Data vector is empty." << std::endl;
          return;
      }

      TFile file(filename.c_str(), "RECREATE");
      TTree tree(treeName.c_str(), "Tree with struct data");

      customCluster cls;
      tree.Branch("sector", &cls.sector);
      tree.Branch("row", &cls.row);
      tree.Branch("max_pad", &cls.max_pad);
      tree.Branch("max_time", &cls.max_time);
      tree.Branch("cog_pad", &cls.cog_pad);
      tree.Branch("cog_time", &cls.cog_time);
      tree.Branch("sigmaPad", &cls.sigmaPad);
      tree.Branch("sigmaTime", &cls.sigmaTime);
      tree.Branch("qMax", &cls.qMax);
      tree.Branch("qTot", &cls.qTot);
      tree.Branch("flag", &cls.flag);
      tree.Branch("mcTrkId", &cls.mcTrkId);
      tree.Branch("mcEvId", &cls.mcEvId);
      tree.Branch("mcSrcId", &cls.mcSrcId);
      tree.Branch("index", &cls.index);
      tree.Branch("label", &cls.label);
      tree.Branch("X", &cls.X);
      tree.Branch("Y", &cls.Y);
      tree.Branch("Z", &cls.Z);

      // Fill the tree with the data
      for (const auto& item : data) {
        cls = item;
        tree.Fill();
      }

      // Write the tree to the file
      file.Write();
      file.Close();

      std::cout << filename << " written." << std::endl;
  }

  template<class T>
  void writeTabularToRootFile(std::vector<std::string> branch_names, T data, std::string filepath, std::string treename = "tree", std::string treeinfo = "tree"){
    // T is something like std::vector<szd::vector<float>> or std::vector<std::array<float, ...>>
    TFile file(filepath.c_str(), "RECREATE");
    TTree tree(treename.c_str(), treeinfo.c_str());

    auto tmp_data = data[0];
    for(int b = 0; b < branch_names.size(); b++){
      tree.Branch(branch_names[b].c_str(), &tmp_data[b]);
    }
    for(auto elem : data){
      tmp_data = elem;
      tree.Fill();
    }
    file.Write();
    file.Close();

    std::cout << filepath << " written." << std::endl;
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
