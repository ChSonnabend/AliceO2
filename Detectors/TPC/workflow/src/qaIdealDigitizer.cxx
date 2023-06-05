#include <cmath>

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

using namespace o2;
using namespace GPUCA_NAMESPACE::gpu;
using namespace o2::tpc;
using namespace o2::framework;

class qaIdeal : public Task
{
 public:
  void init(InitContext&) final;
  void read_digits();
  void read_ideal();
  void fill_map2d(int, int);
  void clear_map2d();
  void find_maxima();
  bool test_neighbour(std::array<float, 5>, std::array<int, 4>, int);
  void effCloneFake(int);
  void run(ProcessingContext&) final;

 private:
  int global_shift[2] = {5, 5};     // shifting digits to select windows easier
  int charge_limits[2] = {2, 1024}; // upper and lower charge limits
  int verbose = 0, max_time = 0, chunk_size = 1000;            // chunk_size in time direction
  std::string mode = "digits,native,tracks,ideal_clusterizer";
  std::string inFileDigits = "tpcdigits.root";
  std::string inFileNative = "tpc-cluster-native.root";
  std::string inFileTracks = "tpctracks.root";
  std::string inFileKinematics = "collisioncontext.root";
  std::string inFileDigitizer = "mclabels_digitizer.root";

  std::array<std::vector<std::array<std::array<std::array<float, 160>, 152>, 36>>, 3> map2d; // 0 - charge; 1 - index ideal; 2 - index digits
  std::vector<int> ideal_point_count, maxima_digits;
  std::vector<std::array<float, 5>> ideal_max_map, ideal_cog_map, digit_map; // sector, row, pad, time, charge
  std::vector<std::vector<int>> assignments_dig_to_id, assignments_id_to_dig;
  std::vector<std::vector<float>> tr_data_X, tr_data_Y, NN_ideal_training_data_X, NN_ideal_training_data_Y;
};

void qaIdeal::init(InitContext& ic)
{
  verbose = ic.options().get<int>("verbose");
  mode = ic.options().get<std::string>("mode");
  chunk_size = ic.options().get<int>("chunk-size");
  inFileDigits = ic.options().get<std::string>("infile-digits");
  inFileNative = ic.options().get<std::string>("infile-native");

  if (verbose >= 1) {
    LOG(info) << "Initializing QA macro and 2D charge map...";
  }

  std::array<std::array<std::array<float, 160>, 152>, 36> temp_arr{0};
  for (int i = 0; i < 3; i++) {
    for (int time_size = 0; time_size < chunk_size + (2 * global_shift[1]); time_size++) {
      map2d[i].push_back(temp_arr);
    };
  };

  if (verbose >= 1) {
    LOG(info) << "Initialized! Vector size is " << map2d[0].size();
  }
}

void qaIdeal::clear_map2d()
{
  for (int i = 0; i < 3; i++) {
    for (int time_size = 0; time_size < chunk_size + (2 * global_shift[1]); time_size++) {
      map2d[i][time_size] = {0};
    };
  };

  if (verbose >= 1) {
    LOG(info) << "Cleared the 2D charge map!";
  }
}

void qaIdeal::read_digits()
{

  if (verbose >= 1) {
    LOG(info) << "Reading the digits to 2D charge map...";
  }

  // reading in the raw digit information
  TFile* digitFile = TFile::Open(inFileDigits.c_str());
  TTree* digitTree = (TTree*)digitFile->Get("o2sim");
  std::vector<o2::tpc::Digit>* digits[36] = {0};
  int current_time = 0;
  int index_digits = 0;
  for (int sector = 0; sector < 36; sector++) {
    std::string branch_name = fmt::format("TPCDigit_{:d}", sector).c_str();
    digitTree->SetBranchAddress(branch_name.c_str(), &digits[sector]);
  }
  for (int iEvent = 0; iEvent < digitTree->GetEntriesFast(); ++iEvent) {
    digitTree->GetEntry(iEvent);
    for (int sector = 0; sector < 36; sector++) {
      for (int i_digit = 0; i_digit < digits[sector]->size(); i_digit++) {
        const auto& digit = (*digits[sector])[i_digit];
        current_time = digit.getTimeStamp();
        if (current_time > max_time)
          max_time = current_time;

        digit_map.push_back(std::array<float, 5>{(float)sector, digit.getRow(), digit.getPad(), (float)current_time, digit.getChargeFloat()});
      }
    }
  }
  digitFile->Close();

  if (verbose >= 1) {
    LOG(info) << "Done reading digits!";
  }
}

void qaIdeal::read_ideal()
{

  float sec, row, maxp, maxt, pcount, lab, cogp, cogt, cogq, maxq;
  long elements = 0;

  for (int i = 0; i < 36; i++) {

    if (verbose > 0) {
      LOG(info) << "Processing ideal clusterizer, sector " << i << " ...";
    }
    std::stringstream tmp_file;
    tmp_file << "mclabels_digitizer_" << i << ".root";
    auto inputFile = TFile::Open(tmp_file.str().c_str());
    std::stringstream tmp_sec;
    tmp_sec << "sector_" << i;
    auto digitizerSector = (TTree*)inputFile->Get(tmp_sec.str().c_str());

    digitizerSector->SetBranchAddress("cluster_sector", &sec);
    digitizerSector->SetBranchAddress("cluster_row", &row);
    digitizerSector->SetBranchAddress("cluster_cog_pad", &cogp);
    digitizerSector->SetBranchAddress("cluster_cog_time", &cogt);
    digitizerSector->SetBranchAddress("cluster_cog_q", &cogq);
    digitizerSector->SetBranchAddress("cluster_max_pad", &maxp);
    digitizerSector->SetBranchAddress("cluster_max_time", &maxt);
    digitizerSector->SetBranchAddress("cluster_max_q", &maxq);
    digitizerSector->SetBranchAddress("cluster_points", &pcount);

    for (int j = 0; j < digitizerSector->GetEntries(); j++) {
      try {
        digitizerSector->GetEntry(j);
        ideal_point_count.push_back(pcount);

        ideal_max_map.push_back(std::array<float,5>{sec, row, maxp, maxt, maxq});
        ideal_cog_map.push_back(std::array<float,5>{sec, row, cogp, cogt, cogq});

        // for (int pad = 0; pad < 11; pad++) {
        //   for (int time = 0; time < 11; time++) {
        //   }
        // }

        elements++;
      } catch (...) {
        LOG(info) << "(Digitizer) Problem occured in sector " << i;
      }
    }

    inputFile->Close();
  }
}

void qaIdeal::fill_map2d(int current_chunk, int fillmode = 0)
{
  // Indices will be stored as index+1 in order to avoid confusion with the rest of the zeros in the array
  int current_time = 0;
  if (fillmode == 0) {
    for (int dig = 0; dig < digit_map.size(); dig++) {
      current_time = digit_map[dig][3];
      if ((current_time - current_chunk + global_shift[1] >= 0) && (current_time - current_chunk + global_shift[1] < chunk_size + (2 * global_shift[1]))) {
        map2d[0][current_time][digit_map[dig][0]][digit_map[dig][1]][digit_map[dig][2]] = digit_map[dig][4];
        map2d[2][current_time][digit_map[dig][0]][digit_map[dig][1]][digit_map[dig][2]] = dig+1;
      }
    }
  } else if (fillmode == 1) {
    for (int dig = 0; dig < ideal_max_map.size(); dig++) {
      current_time = ideal_max_map[dig][3];
      if ((current_time - current_chunk + global_shift[1] >= 0) && (current_time - current_chunk + global_shift[1] < chunk_size + (2 * global_shift[1]))) {
        map2d[1][current_time][ideal_max_map[dig][0]][ideal_max_map[dig][1]][ideal_max_map[dig][2]] = dig+1;
      }
    }
  } else if (fillmode == -1) {
    for (int dig = 0; dig < digit_map.size(); dig++) {
      current_time = digit_map[dig][3];
      if ((current_time - current_chunk + global_shift[1] >= 0) && (current_time - current_chunk + global_shift[1] < chunk_size + (2 * global_shift[1]))) {
        map2d[0][current_time][digit_map[dig][0]][digit_map[dig][1]][digit_map[dig][2]] = digit_map[dig][4];
        map2d[2][current_time][digit_map[dig][0]][digit_map[dig][1]][digit_map[dig][2]] = dig+1;
      }
    }
    for (int dig = 0; dig < ideal_max_map.size(); dig++) {
      current_time = ideal_max_map[dig][3];
      if ((current_time - current_chunk + global_shift[1] >= 0) && (current_time - current_chunk + global_shift[1] < chunk_size + (2 * global_shift[1]))) {
        map2d[1][current_time][ideal_max_map[dig][0]][ideal_max_map[dig][1]][ideal_max_map[dig][2]] = dig+1;
      }
    }
  } else {
    LOG(info) << "Fillmode unknown! No fill performed!";
  }
}

void qaIdeal::find_maxima()
{

  if (verbose >= 1) {
    LOG(info) << "Finding local maxima";
  }

  int counter=0;
  float current_charge = 0;
  for (int sec = 0; sec < 36; sec++) {
    for (int row = 0; row < 152; row++) {
      for (int pad = 0; pad < 160; pad++) {
        for (int time = 0; time < chunk_size; time++) {
          current_charge = map2d[0][time + global_shift[1]][sec][row][pad + global_shift[0]];
          counter++;
          if (current_charge == 0)
            continue;
          else {
            if (current_charge < map2d[0][time + global_shift[1]][sec][row][pad + global_shift[0] + 1])
              continue;
            if (current_charge < map2d[0][time + global_shift[1] + 1][sec][row][pad + global_shift[0]])
              continue;
            if (current_charge < map2d[0][time + global_shift[1]][sec][row][pad + global_shift[0] - 1])
              continue;
            if (current_charge < map2d[0][time + global_shift[1] - 1][sec][row][pad + global_shift[0]])
              continue;
            if (current_charge < map2d[0][time + global_shift[1] + 1][sec][row][pad + global_shift[0] + 1])
              continue;
            if (current_charge < map2d[0][time + global_shift[1] - 1][sec][row][pad + global_shift[0] + 1])
              continue;
            if (current_charge < map2d[0][time + global_shift[1] + 1][sec][row][pad + global_shift[0] - 1])
              continue;
            if (current_charge < map2d[0][time + global_shift[1] - 1][sec][row][pad + global_shift[0] - 1])
              continue;
          }
          maxima_digits.push_back(counter-1);
        }
      }
    }
    if (verbose >= 1) {
      LOG(info) << "Found maxima in sector " << sec << ". Continuing...";
    }
  }

  if (verbose >= 1) {
    LOG(info) << "Found all maxima. Done!";
  }
}

bool qaIdeal::test_neighbour(std::array<float, 5> index, std::array<int, 4> nn, int mode=1){
    return map2d[mode][(int)index[3]+nn[3]][(int)index[0]+nn[0]][(int)index[1]+nn[1]][(int)index[2]+nn[2]]>0;
}

void qaIdeal::effCloneFake(int production_mode=0)
{
  if(production_mode==0){
    for(int i=0; i<4; i++){
      std::vector<int> push_back_dig_to_id, push_back_id_to_dig;
      for (int locmax = 0; locmax < maxima_digits.size(); locmax++) {
        test_neighbour(digit_map[locmax], std::array<int, 4>{0,0,1,0}, 1) ? push_back_id_to_dig.push_back(map2d[1][(int)ideal_max_map[locmax][3]][(int)ideal_max_map[locmax][0]][(int)ideal_max_map[locmax][1]+1][(int)ideal_max_map[locmax][2]]-1) : push_back_id_to_dig.push_back(-1);
        test_neighbour(digit_map[locmax], std::array<int, 4>{0,0,0,1}, 1) ? push_back_id_to_dig.push_back(map2d[1][(int)ideal_max_map[locmax][3]+1][(int)ideal_max_map[locmax][0]][(int)ideal_max_map[locmax][1]][(int)ideal_max_map[locmax][2]]-1) : push_back_id_to_dig.push_back(-1);
        test_neighbour(digit_map[locmax], std::array<int, 4>{0,0,-1,0}, 1) ? push_back_id_to_dig.push_back(map2d[1][(int)ideal_max_map[locmax][3]][(int)ideal_max_map[locmax][0]][(int)ideal_max_map[locmax][1]-1][(int)ideal_max_map[locmax][2]]-1) : push_back_id_to_dig.push_back(-1);
        test_neighbour(digit_map[locmax], std::array<int, 4>{0,0,0,-1}, 1) ? push_back_id_to_dig.push_back(map2d[1][(int)ideal_max_map[locmax][3]-1][(int)ideal_max_map[locmax][0]][(int)ideal_max_map[locmax][1]][(int)ideal_max_map[locmax][2]]-1) : push_back_id_to_dig.push_back(-1);
        assignments_id_to_dig.push_back(push_back_id_to_dig);
      }
      for(int locideal=0; ideal_max_map.size(); locideal++){
        test_neighbour(ideal_max_map[locideal], std::array<int, 4>{0,0,1,0}, 2) ? push_back_dig_to_id.push_back(map2d[1][(int)digit_map[locideal][3]][(int)digit_map[locideal][0]][(int)digit_map[locideal][1]+1][(int)digit_map[locideal][2]]-1) : push_back_dig_to_id.push_back(-1);
        test_neighbour(ideal_max_map[locideal], std::array<int, 4>{0,0,0,1}, 2) ? push_back_dig_to_id.push_back(map2d[1][(int)digit_map[locideal][3]+1][(int)digit_map[locideal][0]][(int)digit_map[locideal][1]][(int)digit_map[locideal][2]]-1) : push_back_dig_to_id.push_back(-1);
        test_neighbour(ideal_max_map[locideal], std::array<int, 4>{0,0,-1,0}, 2) ? push_back_dig_to_id.push_back(map2d[1][(int)digit_map[locideal][3]][(int)digit_map[locideal][0]][(int)digit_map[locideal][1]-1][(int)digit_map[locideal][2]]-1) : push_back_dig_to_id.push_back(-1);
        test_neighbour(ideal_max_map[locideal], std::array<int, 4>{0,0,0,-1}, 2) ? push_back_dig_to_id.push_back(map2d[1][(int)digit_map[locideal][3]-1][(int)digit_map[locideal][0]][(int)digit_map[locideal][1]][(int)digit_map[locideal][2]]-1) : push_back_dig_to_id.push_back(-1);
        assignments_dig_to_id.push_back(push_back_dig_to_id);
      }

    }
  }
}

void qaIdeal::run(ProcessingContext& pc)
{

  LOG(info) << "Starting process for chunk 0 - " << chunk_size;
  read_ideal();
  read_digits();

  for (int loop_chunks = 0; loop_chunks < ceil((float)max_time / (float)chunk_size); loop_chunks++) {
    LOG(info) << "Starting process for chunk " << loop_chunks * chunk_size << " - " << (loop_chunks + 1) * chunk_size;
    clear_map2d();
    fill_map2d(loop_chunks * chunk_size, -1);
    find_maxima();
    if (verbose >= 2) {
      LOG(info) << "Found " << maxima_digits.size() << " maxima.";
    }
  }

  pc.services().get<ControlService>().endOfStream();
  pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
}

#include "Framework/runDataProcessing.h"

DataProcessorSpec readMonteCarloLabels()
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
      {"mode", VariantType::String, "digits,native", {"Mode for running over tracks-file or digits-file: digits, native, tracks, kinematics and/or digitizer."}},
      {"chunk-size", VariantType::Int, 1000, {"Chunk size in which the digits are read, in order to avoid memory overflows."}},
      {"infile-digits", VariantType::String, "tpcdigits.root", {"Input file name (digits)"}},
      {"infile-native", VariantType::String, "tpc-native-clusters.root", {"Input file name (native)"}}}};
}

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{readMonteCarloLabels()};
}