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
  void init_map2d(int);
  void fill_map2d(int);
  void clear_memory();
  void find_maxima();
  void overwrite_map2d();
  void create_training_data(std::vector<int>, int);
  int test_neighbour(std::array<int, 3>, std::array<int, 2>, int);
  // void effCloneFake(int, int);
  void run(ProcessingContext&) final;

 private:
  int global_shift[2] = {5, 5};                     // shifting digits to select windows easier
  int charge_limits[2] = {2, 1024};                 // upper and lower charge limits
  int verbose = 0, max_time = 1;                    // chunk_size in time direction
  std::string mode = "digits,native,tracks,ideal_clusterizer";
  std::string inFileDigits = "tpcdigits.root";
  std::string inFileNative = "tpc-cluster-native.root";
  std::string inFileTracks = "tpctracks.root";
  std::string inFileKinematics = "collisioncontext.root";
  std::string inFileDigitizer = "mclabels_digitizer.root";

  std::array<std::vector<std::array<std::array<long, 160>, 152>>, 2> map2d; // 0 - index ideal; 1 - index digits
  std::vector<int> maxima_digits;
  std::vector<std::array<int, 3>> digit_map, ideal_max_map;
  std::vector<std::array<float, 3>> ideal_cog_map;
  std::vector<float> ideal_max_q, ideal_cog_q, digit_q;

  std::vector<std::vector<std::array<int, 2>>> adj_mat = {{{0,0}}, {{1, 0}, {0, 1}, {-1, 0}, {0, -1}}, {{1, 1}, {-1, 1}, {-1, -1}, {1, -1}}, {{2, 1}, {1, 2}, {-1, 2}, {-2, 1}, {-2, -1}, {-1, -2}, {1, -2}, {2, -1}}, {{2, 2}, {-2, 2}, {-2, -2}, {2, -2}}};
};


// ---------------------------------
void qaIdeal::init(InitContext& ic)
{
  verbose = ic.options().get<int>("verbose");
  mode = ic.options().get<std::string>("mode");
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

  long counter = 0;
  digitTree->GetEntry(0);
  if (verbose >= 1)
    LOG(info) << "Trying to read " << digits->size() << " digits";
  
  digit_map.resize(digits->size());
  digit_q.resize(digits->size());
  
  for (unsigned long i_digit = 0; i_digit < digits->size(); i_digit++) {
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
  for (unsigned long j = 0; j < digitizerSector->GetEntries(); j++) {
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


void qaIdeal::init_map2d(int maxtime){
  std::array<std::array<long, 160>, 152> temp_arr;
  for (int i = 0; i < 2; i++) {
    for (int time_size = 0; time_size < maxtime + (2 * global_shift[1]); time_size++) {
      map2d[i].push_back(temp_arr);
      for(int row = 0; row<152; row++){
        for(int pad = 0; pad<160; pad++){
          map2d[i][time_size][row][pad] = -1;
        }
      }
    };
  };

  if (verbose >= 1)
    LOG(info) << "Initialized 2D map! Time size is " << map2d[0].size();
}


// ---------------------------------
void qaIdeal::fill_map2d(int fillmode = 0)
{

  // Storing the indices
  if (fillmode == 0) {
    for (unsigned long ind = 0; ind < digit_map.size(); ind++) {
      map2d[1][digit_map[ind][2] + global_shift[1]][digit_map[ind][0]][digit_map[ind][1] + global_shift[0]] = ind;
    }
  } else if (fillmode == 1) {
    for (unsigned long ind = 0; ind < ideal_max_map.size(); ind++) {
      map2d[0][ideal_max_map[ind][2] + global_shift[1]][ideal_max_map[ind][0]][ideal_max_map[ind][1] + global_shift[0]] = ind;
    }
  } else if (fillmode == -1) {
    for (unsigned long ind = 0; ind < digit_map.size(); ind++) {
      map2d[1][digit_map[ind][2] + global_shift[1]][digit_map[ind][0]][digit_map[ind][1] + global_shift[0]] = ind;
    }
    for (unsigned long ind = 0; ind < ideal_max_map.size(); ind++) {
      map2d[0][ideal_max_map[ind][2] + global_shift[1]][ideal_max_map[ind][0]][ideal_max_map[ind][1] + global_shift[0]] = ind;
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

  bool is_max = true;
  float current_charge = 0;
  for (int row = 0; row < 152; row++) {
    if(verbose >= 3) LOG(info) << "Finding maxima in row " << row;
    for (int pad = 0; pad < 160; pad++) {
      for (int time = 0; time < max_time; time++) {
        if(map2d[1][time + global_shift[1]][row][pad + global_shift[0]]!=-1){
          current_charge = digit_q[map2d[1][time + global_shift[1]][row][pad + global_shift[0]]];
          
          if(map2d[1][time + global_shift[1]][row][pad + global_shift[0] + 1]!=-1){
            is_max = (current_charge >= digit_q[map2d[1][time + global_shift[1]][row][pad + global_shift[0] + 1]]);
          }

          if(map2d[1][time + global_shift[1] + 1][row][pad + global_shift[0]]!=-1 && is_max){
            is_max = (current_charge >= digit_q[map2d[1][time + global_shift[1] + 1][row][pad + global_shift[0]]]);
          }

          if(map2d[1][time + global_shift[1]][row][pad + global_shift[0] - 1]!=-1 && is_max){
            is_max = (current_charge >= digit_q[map2d[1][time + global_shift[1]][row][pad + global_shift[0] - 1]]);
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
            maxima_digits.push_back(map2d[1][time + global_shift[1]][row][pad + global_shift[0]]);
          }
        }
        is_max=true;
      }
    }
    if(verbose >= 3) LOG(info) << "Found " << maxima_digits.size() << " maxima in row " << row;
  }

  if (verbose >= 1)
    LOG(info) << "Found " << maxima_digits.size() << " maxima. Done!";
}

void qaIdeal::overwrite_map2d(){

  for (int row = 0; row < 152; row++) {
    for (int pad = 0; pad < 160; pad++) {
      for (int time = 0; time < (max_time + 2*global_shift[1]); time++) {
        map2d[1][time][row][pad]=-1;
      }
    }
  }


  for(unsigned long max = 0; max < maxima_digits.size(); max++){
    map2d[1][digit_map[maxima_digits[max]][2]+global_shift[1]][digit_map[maxima_digits[max]][0]][digit_map[maxima_digits[max]][1]+global_shift[0]] = max;
  }

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
  return map2d[mode][(int)index[2] + global_shift[1] + nn[1]][(int)index[0]][(int)index[1] + global_shift[0] + nn[0]];
}


// ---------------------------------
void qaIdeal::run(ProcessingContext& pc)
{

  std::array<unsigned long, 25> assignments_ideal, assignments_digit;
  unsigned long number_of_ideal_max=0, number_of_digit_max=0;
  unsigned long clones=0;

  // init array
  for(int i = 0; i<25; i++){
    assignments_ideal[i]=0;
    assignments_digit[i]=0;
  }

  // int current_max_dig_counter=0, current_max_id_counter=0;
  for (int loop_sectors = 0; loop_sectors < 36; loop_sectors++) {
    LOG(info) << "Starting process for sector " << loop_sectors;

    read_digits(loop_sectors);
    read_ideal(loop_sectors);
    init_map2d(max_time);
    fill_map2d(-1);
    find_maxima();
    overwrite_map2d();

    // effCloneFake(0, loop_chunks*chunk_size);
    // Assignment at d=1
    LOG(info) << "Maxima found in digits (before): " << maxima_digits.size() << "; Maxima found in ideal clusters (before): " << ideal_max_map.size();

    std::vector<long> assigned_ideal(ideal_max_map.size()), clone_order(ideal_max_map.size());
    std::vector<std::array<long, 25>> assignments_dig_to_id(ideal_max_map.size());
    std::vector<long> assigned_digit(maxima_digits.size());
    std::vector<std::array<long, 25>> assignments_id_to_dig(maxima_digits.size());
    long current_neighbour;

    number_of_digit_max += maxima_digits.size();
    number_of_ideal_max += ideal_max_map.size();


    // Level-1 loop: Goes through the layers specified by the adjacency matrix <-> Loop of possible distances
    int layer_count = 0;
    for(int layer = 0; layer < adj_mat.size(); layer++){

      if (verbose >= 2) LOG(info) << "Layer " << layer;

      // Level-2 loop: Goes through the elements of the adjacency matrix at distance d, n times to assign neighbours iteratively
      for(int nn = 0; nn < adj_mat[layer].size(); nn++){

        // Level-3 loop: Goes through all digit maxima and checks neighbourhood for potential ideal maxima
        for(unsigned long locdigit = 0; locdigit < maxima_digits.size(); locdigit++){
          current_neighbour = test_neighbour(digit_map[maxima_digits[locdigit]], adj_mat[layer][nn], 0);
          // if (verbose >= 5) LOG(info) << "current_neighbour: " << current_neighbour;
          // if (verbose >= 5) LOG(info) << "Maximum digit " << maxima_digits[locdigit];
          // if (verbose >= 5) LOG(info) << "Digit max index: " << digit_map[maxima_digits[locdigit]][0] << " " << digit_map[maxima_digits[locdigit]][1] << " " << digit_map[maxima_digits[locdigit]][2];
          if(current_neighbour>=-1 && current_neighbour<=20000000){
            assignments_id_to_dig[locdigit][layer_count + nn] = ((current_neighbour!=-1 && assigned_digit[locdigit]==0) ? (assigned_ideal[current_neighbour]==0 ? current_neighbour : -1) : -1);
          }
          else{
            LOG(warning) << "Current neighbour: " << current_neighbour;
            LOG(warning) << "Ideal max index: " << digit_map[maxima_digits[locdigit]][0] << " " << digit_map[maxima_digits[locdigit]][1] << " " << digit_map[maxima_digits[locdigit]][2];
          }
        }
        if (verbose >= 4) LOG(info) << "Done with assignment for digit maxima layer " << layer;

        // Level-3 loop: Goes through all ideal maxima and checks neighbourhood for potential digit maxima
        for (unsigned long locideal = 0; locideal < ideal_max_map.size(); locideal++) {
          current_neighbour = test_neighbour(ideal_max_map[locideal], adj_mat[layer][nn], 1);
          // if (verbose >= 5) LOG(info) << "current_neighbour: " << current_neighbour;
          // if (verbose >= 5) LOG(info) << "Maximum ideal " << locideal;
          // if (verbose >= 5) LOG(info) << "Ideal max index: " << ideal_max_map[locideal][0] << " " << ideal_max_map[locideal][1] << " " << ideal_max_map[locideal][2];
          if(current_neighbour>=-1 && current_neighbour<=20000000){
            assignments_dig_to_id[locideal][layer_count + nn] = ((current_neighbour!=-1 && assigned_ideal[locideal]==0) ? (assigned_digit[current_neighbour]==0 ? current_neighbour : -1) : -1);
          }
          else{
            LOG(warning) << "Current neighbour: " << current_neighbour;
            LOG(warning) << "Ideal max index: " << ideal_max_map[locideal][0] << " " << ideal_max_map[locideal][1] << " " << ideal_max_map[locideal][2];
          }
        }
        if (verbose >= 4) LOG(info) << "Done with assignment for ideal maxima layer " << layer;

      }

      // Level-2 loop: Checks all digit maxima and how many ideal maxima neighbours have been found in the current layer
      if(layer>=2){
        for (unsigned long locdigit = 0; locdigit < maxima_digits.size(); locdigit++) {
          assigned_digit[locdigit] = 0;
          for(int counter_max=layer_count; counter_max <  layer_count+adj_mat[layer].size(); counter_max++){
            if(assignments_id_to_dig[locdigit][counter_max] >= 0 && assignments_id_to_dig[locdigit][counter_max]<20000000){
              assigned_digit[locdigit]++;
            }
          }
        }
      }

      // Level-2 loop: Checks all ideal maxima and how many digit maxima neighbours have been found in the current layer
      for (unsigned long locideal = 0; locideal < ideal_max_map.size(); locideal++) {
        assigned_ideal[locideal]=0;
        for(int counter_max=layer_count; counter_max <  layer_count+adj_mat[layer].size(); counter_max++){
          if(assignments_dig_to_id[locideal][counter_max] >= 0 && assignments_dig_to_id[locideal][counter_max]<20000000){
            assigned_ideal[locideal]++;
          }
        }
      }

      // Assign all possible digit maxima once you are above a certain distance away from the current maximum (here: At layer with distance greater than sqrt(2))

      if (verbose >= 2) LOG(info) << "Removed maxima for layer " << layer;

      layer_count+=adj_mat[layer].size();

      if(verbose >= 3) LOG(info) << "Layer count is now " << layer_count;

    }

    // Checks the number of assignments that have been made with the above loops
    for(unsigned long ass_id=0; ass_id<maxima_digits.size(); ass_id++){
      int count_elements_id = 0;
      for(auto elem : assignments_dig_to_id[ass_id]){
        if(elem>=0) count_elements_id++;
      }
      if (verbose >= 5 && (ass_id%10000)==0) LOG(info) << "Count elements: " << count_elements_id << " ass_id: " << ass_id << " assignments_ideal: " << assignments_ideal[count_elements_id];
      assignments_ideal[count_elements_id]++;
    }
    for(unsigned long ass_dig=0; ass_dig<ideal_max_map.size(); ass_dig++){
      int count_elements_dig = 0;
      for(auto elem : assignments_id_to_dig[ass_dig]){
        if(elem>=0) count_elements_dig++;
      }
      if (verbose >= 5 && (ass_dig%10000)==0) LOG(info) << "Count elements: " << count_elements_dig << " ass_dig: " << ass_dig << " assignments_digit: " << assignments_digit[count_elements_dig];
      assignments_digit[count_elements_dig]++;
    }

    if (verbose >= 3) LOG(info) << "Done checking the number of assignments";

    // Clone-rate
    for(unsigned long locideal=0; locideal<assignments_dig_to_id.size(); locideal++){
      for(auto elem_id : assignments_dig_to_id[locideal]){
        if(elem_id>=0){
          int count_elements_clone=0;
          for(auto elem_dig : assignments_id_to_dig[elem_id]){
            if(elem_dig>=0) count_elements_clone++;
          }
          if(count_elements_clone==1) clone_order[locideal]++;
        }
      }
    }
    for(unsigned long locideal=0; locideal<assignments_dig_to_id.size(); locideal++){
      if(clone_order[locideal]>1){
        clones++;
      }
    }

    if (verbose >= 3) LOG(info) << "Done with determining the clone rate";

    if (verbose >= 4){
      for(int ass = 0; ass<25; ass++){
        LOG(info) << "Number of assigned digit maxima (#assignments " << ass << "): " << assignments_digit[ass];
        LOG(info) << "Number of assigned ideal maxima (#assignments " << ass << "): " << assignments_ideal[ass] << "\n";
      }
    }

    clear_memory();
  }

  LOG(info) << "\n------- RESULTS -------\n";
  LOG(info) << "Number of digit maxima: " << number_of_digit_max;
  LOG(info) << "Number of ideal maxima: " << number_of_ideal_max << "\n";

  unsigned long efficiency = 0;
  for(int ass = 0; ass<25; ass++){
    LOG(info) << "Number of assigned digit maxima (#assignments " << ass << "): " << assignments_digit[ass];
    LOG(info) << "Number of assigned ideal maxima (#assignments " << ass << "): " << assignments_ideal[ass] << "\n";
    if(ass > 0){
      efficiency+=assignments_ideal[ass];
    }
  }

  LOG(info) << "Number of assigned (ideal -> digit) clusters: " << efficiency << " (" << (float)efficiency*100/(float)number_of_ideal_max << "% of ideal maxima)";
  LOG(info) << "Number of clones (clone-order >= 2 for ideal cluster): " << clones << " (" << (float)clones*100/(float)number_of_ideal_max << "% of ideal maxima)" ;
  LOG(info) << "Number of fakes (number of digit hits that can't be assigned to any ideal hit): " << assignments_digit[0] << " (" << (float)assignments_digit[0]*100/(float)number_of_digit_max << "% of digit maxima)" ;


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
      {"infile-digits", VariantType::String, "tpcdigits.root", {"Input file name (digits)"}},
      {"infile-native", VariantType::String, "tpc-native-clusters.root", {"Input file name (native)"}}}};
}

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{processIdealClusterizer()};
}