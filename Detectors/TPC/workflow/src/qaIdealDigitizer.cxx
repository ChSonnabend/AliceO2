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
  void read_digits(int);
  void read_ideal(int);
  void fill_map2d(int);
  void clear_memory();
  void find_maxima();
  void create_training_data(std::vector<int>, int);
  int test_neighbour(std::array<int, 3>, std::array<int, 2>, int);
  // void effCloneFake(int, int);
  void run(ProcessingContext&) final;

 private:
  int global_shift[2] = {5, 5};                     // shifting digits to select windows easier
  int charge_limits[2] = {2, 1024};                 // upper and lower charge limits
  int verbose = 0, max_time = 1, chunk_size=10000;  // chunk_size in time direction
  std::string mode = "digits,native,tracks,ideal_clusterizer";
  std::string inFileDigits = "tpcdigits.root";
  std::string inFileNative = "tpc-cluster-native.root";
  std::string inFileTracks = "tpctracks.root";
  std::string inFileKinematics = "collisioncontext.root";
  std::string inFileDigitizer = "mclabels_digitizer.root";

  std::array<std::vector<std::array<std::array<int, 160>, 152>>, 2> map2d; // 0 - index ideal; 1 - index digits
  std::vector<int> maxima_digits;
  std::vector<std::array<int, 3>> digit_map, ideal_max_map;
  std::vector<std::array<float, 3>> ideal_cog_map;
  std::vector<float> ideal_max_q, ideal_cog_q, digit_q;

  std::array<std::array<int, 2>, 24> adj_mat = {{{1, 0}, {0, 1}, {-1, 0}, {0, -1}, {1, 1}, {-1, 1}, {-1, -1}, {1, -1}, {2, 1}, {1, 2}, {-1, 2}, {-2, 1}, {-2, -1}, {-1, -2}, {1, -2}, {2, -1}, {2, 2}, {-2, 2}, {-2, -2}, {2, -2}}};
};


// ---------------------------------
void qaIdeal::init(InitContext& ic)
{
  verbose = ic.options().get<int>("verbose");
  mode = ic.options().get<std::string>("mode");
  chunk_size = ic.options().get<int>("chunk-size");
  inFileDigits = ic.options().get<std::string>("infile-digits");
  inFileNative = ic.options().get<std::string>("infile-native");

  if (verbose >= 1)
    LOG(info) << "Initialized QA macro!";

}


// ---------------------------------
void qaIdeal::clear_memory()
{
  map2d[0].clear();
  map2d[1].clear();

  digit_map.clear();
  ideal_max_map.clear();
  ideal_cog_map.clear();
  ideal_max_q.clear();
  ideal_cog_q.clear();
  digit_q.clear();
  maxima_digits.clear();

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
  int current_time = 0;

  std::string branch_name = fmt::format("TPCDigit_{:d}", sector).c_str();
  digitTree->SetBranchAddress(branch_name.c_str(), &digits);

  int counter = 0;
  digitTree->GetEntry(0);
  if (verbose >= 1)
    LOG(info) << "Trying to read " << digits->size() << " digits";
  
  digit_map.resize(digits->size());
  digit_q.resize(digits->size());
  
  for (int i_digit = 0; i_digit < digits->size(); i_digit++) {
    const auto& digit = (*digits)[i_digit];
    current_time = digit.getTimeStamp();
    if (current_time > max_time)
      max_time = current_time;
    digit_map[i_digit] = std::array<int, 3>{digit.getRow(), digit.getPad(), current_time};
    digit_q[i_digit] = digit.getChargeFloat();
    counter++;
  }
  if (verbose >= 1)
    LOG(info) << "Done with sector " << sector;

  digitFile->Close();
  (*digits).clear();

  if (verbose >= 1)
    LOG(info) << "Done reading digits!";
}


// ---------------------------------
void qaIdeal::read_ideal(int sector)
{

  int sec, row, maxp, maxt, pcount, lab;
  float cogp, cogt, cogq, maxq;
  long elements = 0;

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
  digitizerSector->SetBranchAddress("cluster_max_q", &maxq);
  // digitizerSector->SetBranchAddress("cluster_points", &pcount);

  ideal_max_map.resize(digitizerSector->GetEntries());
  ideal_max_q.resize(digitizerSector->GetEntries());
  ideal_cog_map.resize(digitizerSector->GetEntries());
  ideal_cog_q.resize(digitizerSector->GetEntries());

  if (verbose >= 1)
    LOG(info) << "Trying to read " << digitizerSector->GetEntries() << " ideal digits";
  for (int j = 0; j < digitizerSector->GetEntries(); j++) {
    try {
      digitizerSector->GetEntry(j);
      // ideal_point_count.push_back(pcount);

      ideal_max_map[j] = std::array<int, 3>{row, maxp, maxt};
      ideal_max_q[j] = maxq;
      ideal_cog_map[j] = std::array<float, 3>{(float)row, cogp, cogt};
      ideal_cog_q[j] = cogq;
      elements++;

    } catch (...) {
      LOG(info) << "(Digitizer) Problem occured in sector " << sector;
    }
  }

  inputFile->Close();
}


// ---------------------------------
void qaIdeal::fill_map2d(int fillmode = 0)
{

  std::array<std::array<int, 160>, 152> temp_arr{-1};
  for (int i = 0; i < 2; i++) {
    for (int time_size = 0; time_size < max_time + (2 * global_shift[1]); time_size++) {
      map2d[i].push_back(temp_arr);
    };
  };

  if (verbose >= 1)
    LOG(info) << "Initialized 2D map! Time size is " << map2d[0].size();

  // Storing the indices
  if (fillmode == 0) {
    for (int ind = 0; ind < digit_map.size(); ind++) {
      map2d[1][digit_map[ind][2]][digit_map[ind][0]][digit_map[ind][1]] = ind;
    }
  } else if (fillmode == 1) {
    for (int ind = 0; ind < ideal_max_map.size(); ind++) {
      map2d[0][ideal_max_map[ind][2]][ideal_max_map[ind][0]][ideal_max_map[ind][1]] = ind;
    }
  } else if (fillmode == -1) {
    for (int ind = 0; ind < digit_map.size(); ind++) {
      map2d[1][digit_map[ind][2]][digit_map[ind][0]][digit_map[ind][1]] = ind;
    }
    for (int ind = 0; ind < ideal_max_map.size(); ind++) {
      map2d[0][ideal_max_map[ind][2]][ideal_max_map[ind][0]][ideal_max_map[ind][1]] = ind;
    }
  } else {
    LOG(info) << "Fillmode unknown! No fill performed!";
  }
}


// ---------------------------------
void qaIdeal::find_maxima()
{

  if (verbose >= 1) {
    LOG(info) << "Finding local maxima";
  }

  int counter = 0;
  bool is_max = true;
  float current_charge = 0;
  for (int row = 0; row < 152; row++) {
    if(verbose >= 3) LOG(info) << "Finding maxima in row " << row;
    for (int pad = 0; pad < 160; pad++) {
      for (int time = 0; time < max_time; time++) {
        counter++;
        if(map2d[1][time + global_shift[1]][row][pad + global_shift[0]]!=-1){
          current_charge = digit_q[map2d[1][time + global_shift[1]][row][pad + global_shift[0]]];
          
          if(map2d[1][time + global_shift[1]][row][pad + global_shift[0] + 1]!=-1){
            is_max = (current_charge >= digit_q[map2d[1][time + global_shift[1]][row][pad + global_shift[0] + 1]]);
          }

          if(map2d[1][time + global_shift[1] + 1][row][pad + global_shift[0]]!=-1 && is_max){
            is_max = (current_charge >= digit_q[map2d[1][time + global_shift[1] + 1][row][pad + global_shift[0]]]);
          }

          if(map2d[1][time + global_shift[1]][row][pad + global_shift[0] - 1]!=-1 && is_max){
            is_max = (current_charge >= digit_q[map2d[1][time + global_shift[1]][row][pad + global_shift[0]] - 1]);
          }
          
          if(map2d[1][time + global_shift[1] - 1][row][pad + global_shift[0]]!=-1 && is_max){
            is_max = (current_charge >= digit_q[map2d[1][time + global_shift[1] - 1][row][pad + global_shift[0]]]);
          }

          if(map2d[1][time + global_shift[1] + 1][row][pad + global_shift[0] + 1]!=-1 && is_max){
            is_max = (current_charge >= digit_q[map2d[1][time + global_shift[1] + 1][row][pad + global_shift[0] + 1]]);
          }

          if(map2d[1][time + global_shift[1] - 1][row][pad + global_shift[0] + 1]!=-1 && is_max){
            is_max = (current_charge >= digit_q[map2d[1][time + global_shift[1] - 1][row][pad + global_shift[0] + 1]]);
          }

          if(map2d[1][time + global_shift[1] + 1][row][pad + global_shift[0] - 1]!=-1 && is_max){
            is_max = (current_charge >= digit_q[map2d[1][time + global_shift[1] + 1][row][pad + global_shift[0] - 1]]);
          }
          
          if(map2d[1][time + global_shift[1] - 1][row][pad + global_shift[0] - 1]!=-1 && is_max){
            is_max = (current_charge >= digit_q[map2d[1][time + global_shift[1] - 1][row][pad + global_shift[0] - 1]]);
          }
          
          if(is_max){
            maxima_digits.push_back(counter - 1);
          }
        }
      }
    }
  }

  if (verbose >= 1)
    LOG(info) << "Found all maxima. Done!";
}


// ---------------------------------
void qaIdeal::create_training_data(std::vector<int> max_indices, int mat_size = 11)
{

  int data_size = max_indices.size();
  std::vector<std::vector<float>> tr_data_X(data_size, std::vector<float>(mat_size * mat_size));
  std::vector<int> tr_data_Y_class(data_size);
  std::vector<std::array<float, 3>> tr_data_Y_reg(data_size);

  for(int max_point = 0; max_point<data_size; max_point++){
    for (int time = -((mat_size - 1) / 2); time <= ((mat_size - 1) / 2); time++) {
      for (int pad = -((mat_size - 1) / 2); pad <= ((mat_size - 1) / 2); pad++) {
        tr_data_X[max_point][time + pad*time] = digit_q[map2d[1][digit_map[max_point][2]+time][digit_map[max_point][0]][digit_map[max_point][1]+pad]]/digit_q[map2d[1][digit_map[max_point][2]][digit_map[max_point][0]][digit_map[max_point][1]]];
      }
    }
  }
}


// ---------------------------------
int qaIdeal::test_neighbour(std::array<int, 3> index, std::array<int, 2> nn, int mode = 1)
{
  return map2d[mode][(int)index[3] + nn[1]][(int)index[1] + nn[1]][(int)index[2] + nn[0]];
}

// void qaIdeal::effCloneFake(int production_mode=0, int current_chunk)
// {
//   if(production_mode==0){
//     // Assignment at d=1
//     for(int i=0; i<4; i++){
//       // Finding closest nearest neighbours
//       for (int locmax = 0; locmax < maxima_digits.size(); locmax++) {
//         for(int nn=0; nn<4; nn++){
//           if((digit_map[locmax][3] < current_chunk + chunk_size) && (digit_map[locmax][3] >= current_chunk)) assignments_id_to_dig[locmax][nn] = test_neighbour(digit_map[locmax], adj_mat[nn], 2)-1;
//         }
//       }
//       for(int locideal=0; ideal_max_map.size(); locideal++){
//         for(int nn=0; nn<4; nn++){
//           if((ideal_max_map[locideal][3] < current_chunk + chunk_size) && (ideal_max_map[locideal][3] >= current_chunk)) assignments_dig_to_id[locideal][nn] = test_neighbour(ideal_max_map[locideal], adj_mat[nn], 1)-1;
//         }
//       }
//       // Assigning by proximity and Eineindeutigkeit
//       for (int locmax = 0; locmax < maxima_digits.size(); locmax++){
//         int found_nfold = 0;
//         for(int nn = 0; nn<4; nn++){
//           assignments_id_to_dig[locmax][nn]>=0 ? found_nfold++ : continue;
//         }
//         found_nfold == 1 ? assigned_digit[locmax] = true : continue;
//       }
//       for (int locideal = 0; locideal < ideal_max_map.size(); locideal++){
//         int found_nfold = 0;
//         for(int nn = 0; nn<4; nn++){
//           assignments_dig_to_id[locideal][nn]>=0 ? found_nfold++ : continue;
//         }
//         found_nfold == 1 ? assigned_ideal[locideal] = true : continue;
//       }
//     }
//   }
// }


// ---------------------------------
void qaIdeal::run(ProcessingContext& pc)
{

  std::vector<int> assigned_ideal;
  std::vector<std::vector<int>> assignments_dig_to_id;
  std::vector<int> assigned_digit;
  std::vector<std::vector<int>> assignments_id_to_dig;
  int current_neighbour = -1;

  // int current_max_dig_counter=0, current_max_id_counter=0;
  for (int loop_sectors = 0; loop_sectors < 36; loop_sectors++) {
    LOG(info) << "Starting process for sector " << loop_sectors;
    read_digits(loop_sectors);
    read_ideal(loop_sectors);
    fill_map2d(-1);
    find_maxima();
    if (verbose >= 2)
      LOG(info) << "Found " << maxima_digits.size() << " maxima in sector " << loop_sectors;
    // effCloneFake(0, loop_chunks*chunk_size);
    // Assignment at d=1
    LOG(info) << "Maxima found in digits (before): " << maxima_digits.size() << "; Maxima found in ideal clustres (before): " << ideal_max_map.size();

    assigned_ideal.resize(ideal_max_map.size());
    assigned_digit.resize(maxima_digits.size());

    assignments_dig_to_id.resize(ideal_max_map.size());
    assignments_id_to_dig.resize(maxima_digits.size());

    for (int i = 0; i < 4; i++) {
      // Finding closest nearest neighbours
      for (int nn = 0; nn < 4; nn++) {
        for (int locmax = 0; locmax < maxima_digits.size(); locmax++) {
          current_neighbour = test_neighbour(digit_map[maxima_digits[locmax]], adj_mat[nn], 1);
          if(nn==0) assignments_id_to_dig.push_back(std::vector<int>());
          if(current_neighbour!=-1 && assigned_digit[locmax]!=1) assignments_id_to_dig[locmax].push_back(current_neighbour); // + current_max_id_counter;
        }
        for (int locideal = 0; ideal_max_map.size(); locideal++) {
          current_neighbour = test_neighbour(ideal_max_map[locideal], adj_mat[nn], 0);
          if(nn==0) assignments_dig_to_id.push_back(std::vector<int>());
          if(current_neighbour!=-1) assignments_dig_to_id[locideal].push_back(current_neighbour); // + current_max_dig_counter;
        }
      }
      // Assigning by proximity and Eineindeutigkeit
      for (int locmax = 0; locmax < maxima_digits.size(); locmax++) {
        if(assignments_id_to_dig[locmax].size() == 1){
          assigned_digit[locmax]+=1;
        }
      }
      for (int locideal = 0; locideal < ideal_max_map.size(); locideal++) {
        if(assignments_dig_to_id[locideal].size() == 1){
          assigned_ideal[locideal]+=1;
        }
      }

      assignments_dig_to_id.clear();
      assignments_id_to_dig.clear();

    }
    LOG(info) << "Maxima found in digits (after): " << maxima_digits.size() << "; Maxima found in ideal clustres (after): " << ideal_max_map.size();
    // for(int i=0; i<4; i++){
    //   LOG(info) << "Assignments (" << i << " NN): " << 
    // }
    // current_max_dig_counter += maxima_digits.size();
    // current_max_id_counter += ideal_max_map.size();

    clear_memory();

    assignments_dig_to_id.clear();
    assignments_id_to_dig.clear();
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
      {"mode", VariantType::String, "digits,native", {"Mode for running over tracks-file or digits-file: digits, native, tracks, kinematics and/or digitizer."}},
      {"chunk-size", VariantType::Int, 10000, {"Chunk size in which the digits are read, in order to avoid memory overflows."}},
      {"infile-digits", VariantType::String, "tpcdigits.root", {"Input file name (digits)"}},
      {"infile-native", VariantType::String, "tpc-native-clusters.root", {"Input file name (native)"}}}};
}

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{processIdealClusterizer()};
}