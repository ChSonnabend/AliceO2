#include "TPCWorkflow/QaIdealClusters.h"

// ---------------------------------
qaCluster::qaCluster(std::unordered_map<std::string, std::string> options_map)
{
  simulationPath = options_map["simulation-path"];
  outputPath = options_map["output-path"];
  create_output = std::stoi(options_map["create-output"]);
  write_native_file = (bool)std::stoi(options_map["write-native-file"]);
  native_file_single_branch = (bool)std::stoi(options_map["native-file-single-branch"]);
  outCustomNative = options_map["outfile-native"];
  outFileCustomClusters = options_map["outfile-clusters"];
  tpc_sectors = o2::RangeTokenizer::tokenize<int>(options_map["tpc-sectors"]);
}

// ---------------------------------
void qaCluster::init(InitContext& ic)
{
  verbose = ic.options().get<int>("verbose");
  mode = ic.options().get<std::string>("mode");
  realData = ic.options().get<int>("real-data");
  use_max_cog = ic.options().get<int>("use-max-cog");
  global_shift[0] = (int)((ic.options().get<int>("size-pad") - 1.f) / 2.f);
  global_shift[1] = (int)((ic.options().get<int>("size-time") - 1.f) / 2.f);
  global_shift[2] = (int)((ic.options().get<int>("size-row") - 1.f) / 2.f);
  numThreads = ic.options().get<int>("threads");
  inFileDigits = ic.options().get<std::string>("infile-digits");
  inFileNative = ic.options().get<std::string>("infile-native");
  inFileKinematics = ic.options().get<std::string>("infile-kinematics");
  inFileTracks = ic.options().get<std::string>("infile-tracks");
  networkDataOutput = ic.options().get<std::string>("network-data-output");
  networkInputSize = ic.options().get<int>("network-input-size");
  networkClassThres = ic.options().get<float>("network-class-threshold");
  networkSplitIrocOroc = ic.options().get<float>("network-split-iroc-oroc");
  networkOptimizations = ic.options().get<int>("enable-network-optimizations");
  networkNumThreads = ic.options().get<int>("network-num-threads");
  normalization_mode = ic.options().get<int>("normalization-mode");
  looper_tagger_granularity = ic.options().get<std::vector<int>>("looper-tagger-granularity");
  looper_tagger_timewindow = ic.options().get<std::vector<int>>("looper-tagger-timewindow");
  looper_tagger_padwindow = ic.options().get<std::vector<int>>("looper-tagger-padwindow");
  looper_tagger_threshold_num = ic.options().get<std::vector<int>>("looper-tagger-threshold-num");
  looper_tagger_threshold_q = ic.options().get<std::vector<float>>("looper-tagger-threshold-q");
  looper_tagger_opmode = ic.options().get<std::string>("looper-tagger-opmode");
  remove_individual_files = ic.options().get<int>("remove-individual-files");
  training_data_distance_cluster_path = ic.options().get<float>("training-data-distance-cluster-path");
  training_data_distance_cluster_path = std::pow(training_data_distance_cluster_path, 2); // Just to avoid multiple computations and sqrt's later

  if (ic.options().get<int>("max-time") > 0) {
    custom::fill_nested_container(max_time, ic.options().get<int>("max-time"));
    overwrite_max_time = false;
  } else {
    custom::fill_nested_container(max_time, 0);
  }

  if(ic.options().get<int>("network-threshold-sigmoid-trafo") == 1){
    networkClassThres = (float)std::log(networkClassThres/(1.f-networkClassThres));
  }

  ROOT::EnableThreadSafety();

  // LOG(info) << "Testing networks!";
  // std::vector<float> temp_input(10 * (2 * global_shift[0] + 1) * (2 * global_shift[1] + 1), 0);
  // float* out_net_class = network_classification.inference(temp_input, 10);
  //
  // for(int i = 0; i < 10; i++){
  //   LOG(info) << "Output of classification network (" << "): " << out_net_class[i];
  // }

  std::unordered_map<std::string, std::string> OrtOptions{
    {"device",  ic.options().get<std::string>("network-device")},
    {"device-id", "0"},
    {"allocate-device-memory", "0"},
    {"dtype", ic.options().get<std::string>("network-dtype")},
    {"intra-op-num-threads", std::to_string(networkNumThreads)},
    {"enable-optimizations", std::to_string((int)networkOptimizations)},
    {"enable-profiling", "0"},
    {"profiling-output-path", "."},
    {"logging-level", "0"}
  };

  if (mode.find(std::string("network_class")) != std::string::npos || mode.find(std::string("network_full")) != std::string::npos) {
    network_classification_paths = custom::splitString(ic.options().get<std::string>("network-classification-paths"), ";");
    int count_net_class = 0;
    if(networkSplitIrocOroc){
      for(auto path : network_classification_paths){
        OrtOptions["model-path"] = path;
        network_classification[count_net_class].init(OrtOptions);
        count_net_class++;
      }
    } else {
      for(auto path : network_classification_paths){
        OrtOptions["model-path"] = path;
        network_classification[count_net_class].init(OrtOptions);
        network_classification[count_net_class + 1].init(OrtOptions);
        count_net_class+=2;
      }
    }
  }
  if (mode.find(std::string("network_reg")) != std::string::npos || mode.find(std::string("network_full")) != std::string::npos) {
    network_regression_paths = custom::splitString(ic.options().get<std::string>("network-regression-paths"), ";");
    int count_net_reg = 0;
    if(networkSplitIrocOroc){
      for(auto path : network_regression_paths){
        OrtOptions["model-path"] = path;
        network_regression[count_net_reg].init(OrtOptions);
        count_net_reg++;
      }
    } else {
      for(auto path : network_regression_paths){
        OrtOptions["model-path"] = path;
        network_regression[count_net_reg].init(OrtOptions);
        network_regression[count_net_reg + 1].init(OrtOptions);
        count_net_reg+=2;
      }
    }
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

  if (create_output == 1) {
    std::stringstream command;
    command << "rm -rf " << outputPath;
    gSystem->Exec((command.str() + "/looper_tagger*.root").c_str());
    gSystem->Exec((command.str() + "/training_data*.root").c_str());
    gSystem->Exec((command.str() + "/native_ideal*.root").c_str());
    gSystem->Exec((command.str() + "/network_ideal*.root").c_str());
  }

  if (verbose >= 1)
    LOG(info) << "Initialized QA macro, ready to go!";
}

// ---------------------------------
bool qaCluster::checkIdx(int idx)
{
  return (idx > -1);
}

// ---------------------------------
void qaCluster::read_digits(int sector, std::vector<customCluster>& digit_map)
{

  if (verbose >= 1)
    LOG(info) << "[" << sector << "] Reading the digits...";

  // reading in the raw digit information
  TFile* digitFile = TFile::Open((inFileDigits).c_str());
  TTree* digitTree = (TTree*)digitFile->Get("o2sim");

  std::vector<o2::tpc::Digit>* digits = nullptr;
  int current_time = 0, current_pad = 0, current_row = 0;

  std::string branch_name = fmt::format("TPCDigit_{:d}", sector).c_str();
  digitTree->SetBranchAddress(branch_name.c_str(), &digits);

  int counter = 0;
  digitTree->GetEntry(0);

  if (overwrite_max_time) {
    counter = digits->size();
  } else {
    for (unsigned int i_digit = 0; i_digit < digits->size(); i_digit++) {
      const auto& digit = (*digits)[i_digit];
      if (digit.getTimeStamp() < max_time[sector]) {
        counter++;
      }
    }
  }

  digit_map.resize(counter);
  counter = 0;

  for (unsigned int i_digit = 0; i_digit < digits->size(); i_digit++) {
    const auto& digit = (*digits)[i_digit];

    current_time = digit.getTimeStamp();
    current_pad = digit.getPad();

    if (overwrite_max_time) {
      digit_map[counter] = customCluster{sector, digit.getRow(), current_pad, current_time, (float)current_pad, (float)current_time, 0.f, 0.f, digit.getChargeFloat(), digit.getChargeFloat(), (uint8_t)0, -1, -1, -1, (int)counter, 0.f};
      if (current_time > max_time[sector]){
        max_time[sector] = current_time + 1;
      }
      counter++;
    } else {
      if (current_time < max_time[sector]) {
        digit_map[counter] = customCluster{sector, digit.getRow(), current_pad, current_time, (float)current_pad, (float)current_time, 0.f, 0.f, digit.getChargeFloat(), digit.getChargeFloat(), (uint8_t)0, -1, -1, -1, (int)counter, 0.f};
        counter++;
      }
    }
  }

  digitFile->Close();
}

// ---------------------------------
void qaCluster::read_native(int sector, std::vector<customCluster>& digit_map, std::vector<customCluster>& native_map)
{

  ClusterNativeHelper::Reader tpcClusterReader;
  tpcClusterReader.init((inFileNative).c_str());

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
    if (verbose >= 4) {
      LOG(info) << "Native clusters in sector " << sector << ": " << nClustersSec;
    }
    
    int count_clusters = 0;
    for (int irow = 0; irow < o2::tpc::constants::MAXGLOBALPADROW; ++irow) {
      if (overwrite_max_time) {
        count_clusters += clusterIndex.nClusters[sector][irow];
      } else {
        for (int icl = 0; icl < clusterIndex.nClusters[sector][irow]; ++icl) {
          const auto& cl = *(clusterIndex.clusters[sector][irow] + icl);
          clusters.processCluster(cl, Sector(sector), irow);
          current_time = cl.getTime();
          if (current_time < max_time[sector]) {
            count_clusters++;
          }
        }
      }
    }

    native_map.resize(count_clusters);
    count_clusters = 0;

    for (int irow = 0; irow < o2::tpc::constants::MAXGLOBALPADROW; ++irow) {
      const unsigned long nClusters = clusterIndex.nClusters[sector][irow];
      for (int icl = 0; icl < nClusters; ++icl) {
        const auto& cl = *(clusterIndex.clusters[sector][irow] + icl);
        clusters.processCluster(cl, Sector(sector), irow);
        current_pad = cl.getPad();
        current_time = cl.getTime();

        if (overwrite_max_time) {
          native_map[count_clusters] = customCluster{sector, irow, (int)round(current_pad), (int)round(current_time), current_pad, current_time, cl.getSigmaPad(), cl.getSigmaTime(), (float)cl.getQmax(), (float)cl.getQtot(), cl.getFlags(), -1, -1, -1, count_clusters, 0.f};
          if (current_time > max_time[sector]){
            max_time[sector] = current_time + 1;
          }
          count_clusters++;
        } else {
          if (current_time < max_time[sector]) {
            native_map[count_clusters] = customCluster{sector, irow, (int)round(current_pad), (int)round(current_time), current_pad, current_time, cl.getSigmaPad(), cl.getSigmaTime(), (float)cl.getQmax(), (float)cl.getQtot(), cl.getFlags(), -1, -1, -1, count_clusters, 0.f};
            count_clusters++;
          }
        }
      }
    }
  }
  digit_map = native_map;
}

// ---------------------------------
void qaCluster::read_ideal(int sector, std::vector<customCluster>& ideal_map)
{

  int sec, row, maxp, maxt, pcount, trkid, evid, srcid;
  float cogp, cogt, cogq, maxq, sigmap, sigmat;
  int elements = 0, count_clusters = 0;

  std::stringstream tmp_file;
  tmp_file << simulationPath << "/mclabels_digitizer_" << sector << ".root";
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

  if (overwrite_max_time) {
    count_clusters = digitizerSector->GetEntries();
  } else {
    for (unsigned int j = 0; j < digitizerSector->GetEntries(); j++) {
      digitizerSector->GetEntry(j);
      if (maxt < max_time[sector] && cogt < max_time[sector]) {
        count_clusters++;
      }
    }
  }

  ideal_map.resize(count_clusters);
  count_clusters = 0;

  for (unsigned int j = 0; j < digitizerSector->GetEntries(); j++) {
    try {
      digitizerSector->GetEntry(j);
      auto const mctrk = mctracks[srcid][evid][trkid];
      if (overwrite_max_time) {
        ideal_map[count_clusters] = customCluster{sector, row, maxp, maxt, cogp, cogt, sigmap, sigmat, maxq, cogq, 0, trkid, evid, srcid, (int)count_clusters, 0.f, -1.f, -1.f};
        if (maxt >= max_time[sector]){
          max_time[sector] = maxt + 1;
        }
        if (std::ceil(cogt) >= max_time[sector]){
          max_time[sector] = std::ceil(cogt) + 1;
        }
        count_clusters++;
      } else {
        if (maxt < max_time[sector] && cogt < max_time[sector]) {
          ideal_map[count_clusters] = customCluster{sector, row, maxp, maxt, cogp, cogt, sigmap, sigmat, maxq, cogq, 0, trkid, evid, srcid, (int)count_clusters, 0.f, -1.f, -1.f};
          count_clusters++;
        }
      }

    } catch (...) {
      LOG(info) << "[" << sector << "] (Digitizer) Problem occured in sector " << sector;
    }
  }
  inputFile->Close();
}

// ---------------------------------
void qaCluster::read_tracking_clusters(bool mc){
  // --- tracking clusters & momentum association ---
  const auto& mapper = Mapper::instance();
  auto file = TFile::Open((inFileTracks).c_str());
  auto tree = (TTree*)file->Get("tpcrec");
  if (tree == nullptr) {
    std::cout << "Error getting tree\n";
  }
  std::array<int, 3> mcTrackIDs;
  custom::fill_nested_container(mcTrackIDs, -1);
  std::vector<TrackTPC>* tpcTracks = nullptr;
  std::vector<o2::tpc::TPCClRefElem>* mCluRefVecInp = nullptr; // index to clusters linear structure in ClusterNativeAccess
  std::vector<o2::MCCompLabel>* mMCTruthInp = nullptr;
  tree->SetBranchAddress("TPCTracks", &tpcTracks);
  tree->SetBranchAddress("ClusRefs", &mCluRefVecInp);

  if(mc){
    tree->SetBranchAddress("TPCTracksMCTruth", &mMCTruthInp);
  }

  // Getting the native clusters
  ClusterNativeHelper::Reader tpcClusterReader;
  tpcClusterReader.init((inFileNative).c_str());
  ClusterNativeAccess clusterIndex;
  std::unique_ptr<ClusterNative[]> clusterBuffer;
  memset(&clusterIndex, 0, sizeof(clusterIndex));
  o2::tpc::ClusterNativeHelper::ConstMCLabelContainerViewWithBuffer clusterMCBuffer;
  qc::Clusters clusters;
  for (unsigned long i = 0; i < tpcClusterReader.getTreeSize(); ++i) {
    tpcClusterReader.read(i);
    tpcClusterReader.fillIndex(clusterIndex, clusterBuffer, clusterMCBuffer);
  }

  std::vector<customCluster> track_paths, track_clusters;
  std::vector<std::array<float, 3>> clusterMomenta;
  int tabular_data_counter = 0, track_counter = 0;
  
  // GRPGeomHelper::instance().setRequest(grp_geom);
  // o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj);
  float B_field = -5.00668; //GPUO2InterfaceUtils::getNominalGPUBz(*GRPGeomHelper::instance().getGRPMagField());
  LOG(info) << "Updating solenoid field " << B_field;
  
  std::vector<std::array<float, 12>> misc_track_data;
  std::vector<std::string> misc_track_data_branch_names = {"NClusters", "Chi2", "hasASideClusters", "hasCSideClusters", "P", "dEdxQtot", "dEdxQmax", "AbsCharge", "Eta", "Phi", "Pt"};
  
  // const auto& tpcClusRefs = data.getTPCTracksClusterRefs();
  // const auto& tpcClusAcc = // get from tpc-native-clsuters the flat array

  tree->GetEntry(0);
  size_t nTracks = tpcTracks->size();
  // track_paths.resize(track_paths.size() + nTracks*o2::tpc::constants::MAXGLOBALPADROW);
  misc_track_data.resize(nTracks);
  LOG(info) << "Loaded " << nTracks << " tracks. Processing...";

  // ---| track loop |---
  int cluster_counter = 0;
  for (int k = 0; k < nTracks; k++) {
    auto track = (*tpcTracks)[k];
    if(mc){
      mcTrackIDs = {(*mMCTruthInp)[k].getTrackID(), (*mMCTruthInp)[k].getEventID(), (*mMCTruthInp)[k].getSourceID()};
    }
    std::vector<ClusterNative> assigned_clusters(track.getNClusters());
    std::vector<int> sectors(track.getNClusters()), rows(track.getNClusters()), propagation_status(track.getNClusters(), 0);
    std::vector<GlobalPosition2D> global_positions(track.getNClusters());
    std::vector<LocalPosition2D> local_positions(track.getNClusters());
    std::vector<std::array<float, 3>> momentum_after_propagation(track.getNClusters()), track_point(track.getNClusters());
    float z_shift = 0;
    for(int cl = 0; cl < track.getNClusters(); cl++){
      uint8_t sector = 0, row = 0;
      const auto cluster = track.getCluster(*mCluRefVecInp, cl, clusterIndex, sector, row); // ClusterNative instance
      assigned_clusters[cl] = cluster;
      sectors[cl] = (int)sector;
      rows[cl] = (int)row;
      global_positions[cl] = custom::convertSecRowPadToXY((int)sector, (int)row, cluster.getPad(), tpcmap);
      local_positions[cl] = mapper.GlobalToLocal(global_positions[cl], Sector((int)sector));
    }

    // Sorting for local X to improve propagation (?)
    std::vector<int> index_loc_pos(local_positions.size());
    std::iota(index_loc_pos.begin(), index_loc_pos.end(), 0); // Fill with 0, 1, ..., local_positions.size()-1
    std::sort(index_loc_pos.begin(), index_loc_pos.end(), [&local_positions](int i1, int i2) {
        return local_positions[i1].X() < local_positions[i2].X();
    });
    
    for(int cl = 0; cl < track.getNClusters(); cl++){
      int idx = index_loc_pos[cl];
      const auto cluster = assigned_clusters[idx];
      int sector = sectors[idx], row = rows[idx];
      auto loc_pos = local_positions[idx];
      propagation_status[idx] = track.rotate(Sector(sector).phi()); // Needed for tracks that cross the sector boundaries
      // propagation_status[idx] = track.rotate(constants::math::PI - math::atan(global_positions.Y() / global_positions.X())); // Needed for tracks that cross the sector boundaries
      
      if(propagation_status[idx]){
        propagation_status[idx] = track.propagateTo(loc_pos.X(), B_field);
        if(propagation_status[idx]){
          track.getXYZGlo(track_point[idx]);
          z_shift += (tpcmap.LinearTime2Z(sector, cluster.getTime()) - track_point[idx][2]);
          propagation_status[idx] = propagation_status[idx] && track.getPxPyPzGlo(momentum_after_propagation[idx]);
        } else {
          if(verbose > 2) {
            LOG(warning) << "Track propagation failed! (Track " << k << ", cluster " << cl << ")";
          }
        }
      } else if(verbose > 2) {
        LOG(warning) << "Track rotation failed! (Track " << k << ", cluster " << cl << ")";
      }
    }
    z_shift /= track.getNClusters();

    for(int cl = 0; cl < track.getNClusters(); cl++){
      int idx = index_loc_pos[cl];
      int sector = sectors[idx], row = rows[idx];
      if(propagation_status[idx]){
        const auto cluster = assigned_clusters[idx];
        auto glo_pos = global_positions[idx];
        LocalPosition3D loc_point = mapper.GlobalToLocal(GlobalPosition3D(track_point[idx][0], track_point[idx][1], track_point[idx][2] + z_shift), Sector((int)sector));
        if(cluster.getPad() < 0){ LOG(info) << "Found cluster with cog_pad < 0: " << cluster.getPad() << " (Track " << k << ", cluster " << cl << ")"; }
        customCluster trk_cls{sector, row, (int)round(cluster.getPad()), (int)round(cluster.getTime()), cluster.getPad(), cluster.getTime(), cluster.getSigmaPad(), cluster.getSigmaTime(), (float)cluster.getQmax(), (float)cluster.getQtot(), cluster.getFlags(), mcTrackIDs[0], mcTrackIDs[1], mcTrackIDs[2], cluster_counter, 0.f, glo_pos.X(), glo_pos.Y(), tpcmap.LinearTime2Z(sector, cluster.getTime())};
        customCluster trk_path{sector, row, (int)round(tpcmap.LinearY2Pad(sector, row, loc_point.Y())), (int)round(tpcmap.LinearZ2Time(sector, loc_point.Z())), tpcmap.LinearY2Pad(sector, row, loc_point.Y()), tpcmap.LinearZ2Time(sector, loc_point.Z()), cluster.getSigmaPad(), cluster.getSigmaTime(), (float)cluster.getQmax(), (float)cluster.getQtot(), cluster.getFlags(), mcTrackIDs[0], mcTrackIDs[1], mcTrackIDs[2], cluster_counter, 0.f, track_point[idx][0], track_point[idx][1], track_point[idx][2] + z_shift};
        // LOG(info) << sector << " " << row << " " << tpcmap.LinearY2Pad(sector, row, track_point[idx][1]) << " " << tpcmap.LinearY2Pad(sector, row, loc_point.Y()) << " " << loc_point.Y() << " " << track_point[idx][1];
        track_paths.push_back(trk_path);
        track_clusters.push_back(trk_cls);
        tracking_paths[sector].push_back(trk_path);
        tracking_clusters[sector].push_back(trk_cls);
        clusterMomenta.push_back(momentum_after_propagation[idx]);
        momentum_vectors[sector].push_back(momentum_after_propagation[idx]);
        cluster_counter++;
        if((std::pow(track_point[idx][0], 2) + std::pow(track_point[idx][1], 2)) > std::pow(250,2)){
          GlobalPosition3D point(track_point[idx][0], track_point[idx][1], track_point[idx][2]);
          LOG(warning) << "[" << (int)sector << "] Found TPC track cluster extrapolated outside the TPC boundaries! Track path (XYZ): (" << point.X() << ", " << point.Y() << ", " << point.Z() << ") -> (local XY) (" << mapper.GlobalToLocal(point, Sector((int)sector)).X() << ", " << mapper.GlobalToLocal(point, Sector((int)sector)).Y() << "), Cluster position (XYZ): (" << glo_pos.X() << ", " << glo_pos.Y() << ", " << tpcmap.LinearTime2Z(sector, cluster.getTime()) << ") -> (local XY): (" << local_positions[idx].X() << ", " << local_positions[idx].Y() << ").";
        }
      } else if(verbose > 3) {
        LOG(warning) << "[" << (int)sector << "] Propagation failed for track " << k << ", cluster " << cl << " (sector " << sector << ", row " << row << ")!";
      }
    }

    misc_track_data[k][0] = track.getNClusters();
    misc_track_data[k][1] = track.getChi2();
    misc_track_data[k][2] = track.hasASideClusters();
    misc_track_data[k][3] = track.hasCSideClusters();
    misc_track_data[k][4] = track.getP();
    misc_track_data[k][5] = track.getdEdx().dEdxTotTPC;
    misc_track_data[k][6] = track.getdEdx().dEdxMaxTPC;
    misc_track_data[k][7] = track.getAbsCharge(); // TPC inner param = P / AbsCharge
    misc_track_data[k][8] = track.getEta();
    misc_track_data[k][9] = track.getPhi();
    misc_track_data[k][10] = track.getPt();
  }

  // Writing some data in between
  std::string outfile = custom::splitString(outFileCustomClusters, ".root")[0];
  custom::writeStructToRootFile(outputPath + "/" + outfile + "_track_paths.root", "data", track_paths);
  custom::writeStructToRootFile(outputPath + "/" + outfile + "_track_clusters.root", "data", track_clusters);
  custom::writeTabularToRootFile({"momentum_X", "momentum_Y", "momentum_Z"}, clusterMomenta, outputPath + "/" + outfile + "_momenta.root", "momenta", "Momentum infromation from tracking.");
  custom::writeTabularToRootFile(misc_track_data_branch_names, misc_track_data, outputPath + "/tpc_tracks_tabular_information.root", "tpc_tracks_info", "TPC track information");
  LOG(info) << "Clusters of tracks written!";
}

// ---------------------------------
void qaCluster::read_kinematics(std::vector<std::vector<std::vector<o2::MCTrack>>>& tracks)
{

  o2::steer::MCKinematicsReader reader((simulationPath + "/collisioncontext.root").c_str());
  std::vector<std::array<float,7>> track_ideal_info;
  std::vector<std::string> track_ideal_branches = {"SourceID", "EventID", "TrackID", "Eta", "Phi", "P", "Pt"};

  tracks.resize(reader.getNSources());
  for (int src = 0; src < reader.getNSources(); src++) {
    tracks[src].resize(reader.getNEvents(src));
    for (int ev = 0; ev < reader.getNEvents(src); ev++) {
      tracks[src][ev] = reader.getTracks(src, ev);
      int track_counter = 0;
      for(auto trk : tracks[src][ev]){
        track_ideal_info.push_back({(float)src, (float)ev, (float)track_counter, (float)trk.GetEta(), (float)trk.GetPhi(), (float)trk.GetP(), (float)trk.GetPt()});
        track_counter++;
      }
    }
  }

  custom::writeTabularToRootFile(track_ideal_branches, track_ideal_info, outputPath + "/mc_tracks_tabular_information.root", "mc_tracks_info", "MC track information");

  // LOG(info) << "Done reading kinematics, exporting to file (for python readout)";
}

// ---------------------------------
void qaCluster::write_custom_native(ProcessingContext& pc, std::vector<customCluster>& native_writer_map, bool perSector)
{

  // using MCLabelContainer = o2::dataformats::MCTruthContainer<o2::MCCompLabel>;

  std::vector<ClusterNativeContainer> cont(o2::tpc::constants::MAXSECTOR * o2::tpc::constants::MAXGLOBALPADROW);
  std::vector<o2::dataformats::MCLabelContainer> mcTruth(o2::tpc::constants::MAXSECTOR * o2::tpc::constants::MAXGLOBALPADROW);

  std::array<std::array<int, o2::tpc::constants::MAXGLOBALPADROW>, o2::tpc::constants::MAXSECTOR> cluster_sector_counter;
  custom::fill_nested_container(cluster_sector_counter, 0);

  for (auto cls : native_writer_map) {
    cluster_sector_counter[cls.sector][cls.row]++;
  }
  for (int sec = 0; sec < o2::tpc::constants::MAXSECTOR; sec++) {
    for (int row = 0; row < o2::tpc::constants::MAXGLOBALPADROW; row++) {
      cont[sec * o2::tpc::constants::MAXGLOBALPADROW + row].sector = sec;
      cont[sec * o2::tpc::constants::MAXGLOBALPADROW + row].globalPadRow = row;
      cont[sec * o2::tpc::constants::MAXGLOBALPADROW + row].clusters.resize(cluster_sector_counter[sec][row]);
    }
  }

  int total_clusters = 0;
  o2::dataformats::MCLabelContainer mcTruthBuffer;
  std::vector<o2::dataformats::MCLabelContainer> sorted_mc_labels(36);
  std::array<int, 36> sector_counter;
  custom::fill_nested_container(sector_counter, 0);
  custom::fill_nested_container(cluster_sector_counter, 0);
  o2::MCCompLabel dummyMcLabel(0,0,0,true);
  for (auto const cls : native_writer_map) {
    int sec = cls.sector;
    int row = cls.row;
    // cont[sec*o2::tpc::constants::MAXGLOBALPADROW + row].clusters[cluster_sector_counter[sec][row]].setTime(cls[3]);
    cont[sec * o2::tpc::constants::MAXGLOBALPADROW + row].clusters[cluster_sector_counter[sec][row]].setTimeFlags(cls.cog_time, cls.flag);
    cont[sec * o2::tpc::constants::MAXGLOBALPADROW + row].clusters[cluster_sector_counter[sec][row]].setPad(cls.cog_pad);
    cont[sec * o2::tpc::constants::MAXGLOBALPADROW + row].clusters[cluster_sector_counter[sec][row]].setSigmaTime(cls.sigmaTime);
    cont[sec * o2::tpc::constants::MAXGLOBALPADROW + row].clusters[cluster_sector_counter[sec][row]].setSigmaPad(cls.sigmaPad);
    cont[sec * o2::tpc::constants::MAXGLOBALPADROW + row].clusters[cluster_sector_counter[sec][row]].qMax = cls.qMax;
    cont[sec * o2::tpc::constants::MAXGLOBALPADROW + row].clusters[cluster_sector_counter[sec][row]].qTot = cls.qTot;
    if(cls.mcTrkId != -1){
      mcTruth[sec * o2::tpc::constants::MAXGLOBALPADROW + row].addElement(cluster_sector_counter[sec][row], o2::MCCompLabel(cls.mcTrkId, cls.mcEvId, cls.mcSrcId, false));
      sorted_mc_labels[sec].addElement(sector_counter[sec], o2::MCCompLabel(cls.mcTrkId, cls.mcEvId, cls.mcSrcId, false));
      // mcTruthBuffer.addElement(total_clusters, o2::MCCompLabel(cls.mcTrkId, cls.mcEvId, cls.mcSrcId, false));
    } else {
      mcTruth[sec * o2::tpc::constants::MAXGLOBALPADROW + row].addElement(cluster_sector_counter[sec][row], dummyMcLabel);
      sorted_mc_labels[sec].addElement(sector_counter[sec], dummyMcLabel);
      // mcTruthBuffer.addElement(total_clusters, dummyMcLabel);
    }
    cluster_sector_counter[sec][row]++;
    sector_counter[sec]++;
    total_clusters++;
  }

  std::unique_ptr<ClusterNative[]> clusterBuffer;
  std::unique_ptr<ClusterNativeAccess> clusters = ClusterNativeHelper::createClusterNativeIndex(clusterBuffer, cont, &mcTruthBuffer, &mcTruth);

  LOG(info) << "ClusterNativeAccess structure created.";
  // std::vector<char> buffer;
  // mcTruthBuffer.flatten_to(buffer);
  // o2::dataformats::IOMCTruthContainerView tmp_container(buffer);
  // o2::dataformats::ConstMCTruthContainer<o2::MCCompLabel> constMcLabelContainer;
  // tmp_container.copyandflatten(constMcLabelContainer);
  // o2::dataformats::ConstMCTruthContainerView containerView(constMcLabelContainer);
  // clusters.get()->clustersMCTruth = &containerView;

  o2::tpc::ClusterNativeAccess const& clusterIndex = *(clusters.get());

  if(perSector){
    // Clusters are shipped by sector, we are copying into per-sector buffers (anyway only for ROOT output)
    o2::tpc::TPCSectorHeader clusterOutputSectorHeader{0};
    for (unsigned int i : tpc_sectors) {
      unsigned int subspec = i;
      clusterOutputSectorHeader.sectorBits = (1ul << i);
      char* buffer = pc.outputs().make<char>({o2::header::gDataOriginTPC, "CLUSTERNATIVE", subspec, {clusterOutputSectorHeader}}, clusterIndex.nClustersSector[i] * sizeof(*clusterIndex.clustersLinear) + sizeof(o2::tpc::ClusterCountIndex)).data();
      o2::tpc::ClusterCountIndex* outIndex = reinterpret_cast<o2::tpc::ClusterCountIndex*>(buffer);
      memset(outIndex, 0, sizeof(*outIndex));
      for (int j = 0; j < o2::tpc::constants::MAXGLOBALPADROW; j++) {
        outIndex->nClusters[i][j] = clusterIndex.nClusters[i][j];
      }
      memcpy(buffer + sizeof(*outIndex), clusterIndex.clusters[i][0], clusterIndex.nClustersSector[i] * sizeof(*clusterIndex.clustersLinear));

      // o2::dataformats::MCLabelContainer cont;
      // for (unsigned int j = 0; j < clusterIndex.nClustersSector[i]; j++) {
      //   const auto& labels = clusterIndex.clustersMCTruth->getLabels(clusterIndex.clusterOffset[i][0] + j);
      //   for (const auto& label : labels) {
      //     cont.addElement(j, label);
      //   }
      // }
      o2::dataformats::ConstMCLabelContainer contflat;
      sorted_mc_labels[i].flatten_to(contflat);
      pc.outputs().snapshot({o2::header::gDataOriginTPC, "CLNATIVEMCLBL", subspec, {clusterOutputSectorHeader}}, contflat);
    }
  }

  LOG(info) << "------- Native clusters structure written -------";
}

// ---------------------------------
tpc2d qaCluster::init_map2d(int sector)
{
  tpc2d map2d;
  for (int i = 0; i < 2; i++) {
    map2d[i].resize(max_time[sector] + 1 + (2 * global_shift[1]));
    for (int time_size = 0; time_size < (max_time[sector] + 1 + 2 * global_shift[1]); time_size++) {
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
void qaCluster::fill_map2d(int sector, tpc2d& map2d, std::vector<customCluster>& digit_map, std::vector<customCluster>& ideal_map, int fillmode)
{

  int* map_ptr = nullptr;
  if (use_max_cog == 0) {
    // Storing the indices
    if (fillmode == 0 || fillmode == -1) {
      for (auto dig : digit_map) {
        map2d[1][dig.max_time + global_shift[1]][dig.row + rowOffset(dig.row) + global_shift[2]][dig.max_pad + global_shift[0] + padOffset(dig.row)] = dig.index;
      }
    }
    if (fillmode == 1 || fillmode == -1) {
      std::vector<customCluster> new_ideal_map;
      int overwrite_index = 0, found_overwrites = 0;
      for (auto idl : ideal_map) {
        overwrite_index = idl.index - found_overwrites;
        map_ptr = &map2d[0][idl.max_time + global_shift[1]][idl.row + rowOffset(idl.row) + global_shift[2]][idl.max_pad + global_shift[0] + padOffset(idl.row)];
        if (*map_ptr != -1) {
          for(auto& cls : new_ideal_map){
            if((idl.max_time == cls.max_time) && (idl.max_pad == cls.max_pad)) {
              cls.cog_pad = (cls.cog_pad*cls.qTot + idl.cog_pad*idl.qTot)/(cls.qTot + idl.qTot);
              cls.cog_time = (cls.cog_time*cls.qTot + idl.cog_time*idl.qTot)/(cls.qTot + idl.qTot);
              cls.qTot += idl.qTot;
              cls.qMax += idl.qMax;
              overwrite_index = cls.index;
              found_overwrites++;
              break;
            }
          }
          if(verbose >= 3) {
            LOG(warning) << "[" << sector << "] Conflict detected! Current MaxQ : " << ideal_map[*map_ptr].qMax << "; New MaxQ: " << idl.qMax << "; Index " << idl.index << "/" << ideal_map.size();
          }
        } else {
          idl.index -= found_overwrites;
          new_ideal_map.push_back(idl);
          *map_ptr = overwrite_index;
        }
      }
      if(ideal_map.size() != new_ideal_map.size() && verbose >= 1){
        LOG(info) << "[" << sector << "] New ideal map size is " << new_ideal_map.size() << ", old size was " << ideal_map.size();
      }
      ideal_map = new_ideal_map;
    }
    if (fillmode < -1 || fillmode > 1) {
      LOG(info) << "[" << sector << "] Fillmode unknown! No fill performed!";
    }
  } else if (use_max_cog == 1) {
    // Storing the indices
    if (fillmode == 0 || fillmode == -1) {
      for (auto dig : digit_map) {
        map2d[1][dig.max_time + global_shift[1]][dig.row + rowOffset(dig.row) + global_shift[2]][dig.max_pad + global_shift[0] + padOffset(dig.row)] = dig.index;
      }
    }
    if (fillmode == 1 || fillmode == -1) {
      std::vector<customCluster> new_ideal_map;
      int overwrite_index = 0, found_overwrites = 0;
      for (auto idl : ideal_map) {
        overwrite_index = idl.index - found_overwrites;
        map_ptr = &map2d[0][round(idl.cog_time) + global_shift[1]][idl.row + rowOffset(idl.row) + global_shift[2]][round(idl.cog_pad) + global_shift[0] + padOffset(idl.row)];
        if (*map_ptr != -1) {
          for(auto& cls : new_ideal_map){
            if((round(idl.cog_time) == round(cls.cog_time)) && (round(idl.cog_pad) == round(cls.cog_pad))) {
              cls.cog_pad = (cls.cog_pad*cls.qTot + idl.cog_pad*idl.qTot)/(cls.qTot + idl.qTot);
              cls.cog_time = (cls.cog_time*cls.qTot + idl.cog_time*idl.qTot)/(cls.qTot + idl.qTot);
              cls.qTot += idl.qTot;
              cls.qMax += idl.qMax;
              overwrite_index = cls.index;
              found_overwrites++;
              break;
            }
          }
          if(verbose >= 3) {
            LOG(warning) << "[" << sector << "] Conflict detected! Current MaxQ : " << ideal_map[*map_ptr].qMax << "; New MaxQ: " << idl.qMax << "; Index " << idl.index << "/" << ideal_map.size();
          }
        } else {
          idl.index -= found_overwrites;
          new_ideal_map.push_back(idl);
          *map_ptr = overwrite_index;
        }
      }
      if(ideal_map.size() != new_ideal_map.size() && verbose >= 1){
        LOG(info) << "[" << sector << "] New ideal map size is " << new_ideal_map.size() << ", old size was " << ideal_map.size();
      }
      ideal_map = new_ideal_map;
    }
    if (fillmode < -1 || fillmode > 1) {
      LOG(info) << "[" << sector << "] Fillmode unknown! No fill performed!";
    }
  }
}

// ---------------------------------
void qaCluster::find_maxima(int sector, tpc2d& map2d, std::vector<customCluster>& digit_map, std::vector<int>& maxima_digits)
{

  bool is_max = true;
  float current_charge = 0;
  int row_offset = 0, pad_offset = 0;
  for (int row = 0; row < o2::tpc::constants::MAXGLOBALPADROW; row++) {
    row_offset = rowOffset(row);
    pad_offset = padOffset(row);
    for (int pad = 0; pad < TPC_GEOM[row][2] + 1; pad++) {
      for (int time = 0; time < max_time[sector]; time++) {

        int current_idx = map2d[1][time + global_shift[1]][row + row_offset + global_shift[2]][pad + global_shift[0] + pad_offset];

        if (checkIdx(current_idx)) {

          current_charge = digit_map[current_idx].qMax;

          if (map2d[1][time + global_shift[1]][row + row_offset + global_shift[2]][pad + global_shift[0] + pad_offset + 1] != -1) {
            is_max = (current_charge >= digit_map[map2d[1][time + global_shift[1]][row + row_offset + global_shift[2]][pad + global_shift[0] + pad_offset + 1]].qMax);
          }

          if (is_max && map2d[1][time + global_shift[1] + 1][row + row_offset + global_shift[2]][pad + global_shift[0] + pad_offset] != -1) {
            is_max = (current_charge >= digit_map[map2d[1][time + global_shift[1] + 1][row + row_offset + global_shift[2]][pad + global_shift[0] + pad_offset]].qMax);
          }

          if (is_max && map2d[1][time + global_shift[1]][row + row_offset + global_shift[2]][pad + global_shift[0] + pad_offset - 1] != -1) {
            is_max = (current_charge >= digit_map[map2d[1][time + global_shift[1]][row + row_offset + global_shift[2]][pad + global_shift[0] + pad_offset - 1]].qMax);
          }

          if (is_max && map2d[1][time + global_shift[1] - 1][row + row_offset + global_shift[2]][pad + global_shift[0] + pad_offset] != -1) {
            is_max = (current_charge >= digit_map[map2d[1][time + global_shift[1] - 1][row + row_offset + global_shift[2]][pad + global_shift[0] + pad_offset]].qMax);
          }

          if (is_max && map2d[1][time + global_shift[1] + 1][row + row_offset + global_shift[2]][pad + global_shift[0] + pad_offset + 1] != -1) {
            is_max = (current_charge >= digit_map[map2d[1][time + global_shift[1] + 1][row + row_offset + global_shift[2]][pad + global_shift[0] + pad_offset + 1]].qMax);
          }

          if (is_max && map2d[1][time + global_shift[1] - 1][row + row_offset + global_shift[2]][pad + global_shift[0] + pad_offset + 1] != -1) {
            is_max = (current_charge >= digit_map[map2d[1][time + global_shift[1] - 1][row + row_offset + global_shift[2]][pad + global_shift[0] + pad_offset + 1]].qMax);
          }

          if (is_max && map2d[1][time + global_shift[1] + 1][row + row_offset + global_shift[2]][pad + global_shift[0] + pad_offset - 1] != -1) {
            is_max = (current_charge >= digit_map[map2d[1][time + global_shift[1] + 1][row + row_offset + global_shift[2]][pad + global_shift[0] + pad_offset - 1]].qMax);
          }

          if (is_max && map2d[1][time + global_shift[1] - 1][row + row_offset + global_shift[2]][pad + global_shift[0] + pad_offset - 1] != -1) {
            is_max = (current_charge >= digit_map[map2d[1][time + global_shift[1] - 1][row + row_offset + global_shift[2]][pad + global_shift[0] + pad_offset - 1]].qMax);
          }

          if (is_max) {
            maxima_digits.push_back(current_idx);
            digit_map[current_idx].label = 1; // Preemptive to tag maxima
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
bool qaCluster::is_local_minimum(tpc2d& map2d, std::array<int, 3>& current_position, std::vector<float>& digit_q)
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
int qaCluster::local_saddlepoint(tpc2d& map2d, std::array<int, 3>& current_position, std::vector<float>& digit_q)
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
void qaCluster::native_clusterizer(tpc2d& map2d, std::vector<std::array<int, 3>>& digit_map, std::vector<int>& maxima_digits, std::vector<float>& digit_q, std::vector<std::array<float, 3>>& digit_clusterizer_map, std::vector<float>& digit_clusterizer_q)
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
std::vector<std::vector<std::vector<int>>> qaCluster::looper_tagger(int sector, int counter, std::vector<customCluster>& index_map, std::vector<int>& index_array)
{
  // looper_tagger_granularity[counter] = 1; // to be removed later: Testing for now
  int looper_detector_timesize = std::ceil((float)max_time[sector] / (float)looper_tagger_granularity[counter]);

  std::vector<std::vector<std::vector<float>>> tagger(looper_detector_timesize, std::vector<std::vector<float>>(o2::tpc::constants::MAXGLOBALPADROW)); // time_slice (=std::floor(time/looper_tagger_granularity[counter])), row, pad array -> looper_tagged = 1, else 0
  std::vector<std::vector<std::vector<int>>> tagger_counter(looper_detector_timesize, std::vector<std::vector<int>>(o2::tpc::constants::MAXGLOBALPADROW));
  std::vector<std::vector<std::vector<int>>> looper_tagged_region(max_time[sector] + 1, std::vector<std::vector<int>>(o2::tpc::constants::MAXGLOBALPADROW)); // accumulates all the regions that should be tagged: looper_tagged_region[time_slice][row] = (pad_low, pad_high)
  std::vector<std::vector<std::vector<std::vector<int>>>> sorted_idx(looper_detector_timesize, std::vector<std::vector<std::vector<int>>>(o2::tpc::constants::MAXGLOBALPADROW));

  int operation_mode = 2;
  // op_mode.find(std::string("digit")) != std::string::npos ? operation_mode = 1 : operation_mode = operation_mode;
  // op_mode.find(std::string("ideal")) != std::string::npos ? operation_mode = 2 : operation_mode = operation_mode;

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
    row = index_map[idx].row;
    pad = round(index_map[idx].cog_pad) + pad_offset;
    time_slice = std::floor(round(index_map[idx].cog_time) / (float)looper_tagger_granularity[counter]);

    tagger[time_slice][row][pad]++;
    tagger_counter[time_slice][row][pad]++;
    tagger[time_slice][row][pad] += index_map[idx].qTot; // / landau_approx((array_q[idx] - 25.f) / 17.f);
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
          elementAppearance = custom::hasElementAppearedMoreThanNTimesInVectors(idx_vector, index_map, looper_tagger_threshold_num[counter]);
          accept = (elementAppearance.size() == 0 ? false : true);
        }

        if (accept) {
          // This needs to be modified still
          for (int lbl : elementAppearance) {
            for (std::vector<int> idx_v : idx_vector) {
              for (int idx : idx_v) {
                if (index_map[idx].mcTrkId == lbl) {
                  auto const current_cl = index_map[idx];
                  ideal_pad = std::round(current_cl.cog_pad);
                  ideal_time = std::round(current_cl.cog_time);
                  sigma_pad = std::round(current_cl.sigmaPad);
                  sigma_time = std::round(current_cl.sigmaTime);
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
    file_in << outputPath << "/looper_tagger_" << sector << "_" << counter << ".root";
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
void qaCluster::remove_loopers_digits(int sector, std::vector<std::vector<std::vector<int>>>& looper_map, std::vector<customCluster>& index_map, std::vector<int>& index_array)
{
  std::vector<int> new_index_array;

  for (auto idx : index_array) {
    if (looper_map[round(index_map[idx].cog_time)][index_map[idx].row][round(index_map[idx].cog_pad)] < 0) {
      new_index_array.push_back(idx);
    }
  }

  if (verbose > 2)
    LOG(info) << "[" << sector << "] Old size of maxima index array: " << index_array.size() << "; New size: " << new_index_array.size();

  index_array = new_index_array;
}

// ---------------------------------
void qaCluster::remove_loopers_native(int sector, std::vector<std::vector<std::vector<int>>>& looper_map, std::vector<customCluster>& index_map, std::vector<int>& index_array)
{
  // This function does not remove by using the looper_map because index_array corresponds to maxima_digits which are removed from the digits anyway in a previous step
  std::vector<customCluster> new_map(index_array.size());
  for (auto idx : index_array) {
    new_map[idx] = index_map[idx];
    new_map[idx].index = idx;
  }
  index_map = new_map;
}

// ---------------------------------
void qaCluster::remove_loopers_ideal(int sector, std::vector<std::vector<std::vector<int>>>& looper_map, std::vector<customCluster>& ideal_map)
{

  std::vector<int> pass_index(ideal_map.size(), 0);
  std::vector<customCluster> new_ideal_map;

  int counter_1 = 0, counter_2 = 0;
  for (auto const idl : ideal_map) {
    if (looper_map[round(idl.cog_time)][idl.row][round(idl.cog_pad)] != idl.mcTrkId) { // Compares trkID and only removes ideal clusters with identical ID
      pass_index[counter_1] = 1;
      counter_1++;
    }
  }
  new_ideal_map.resize(counter_1);
  counter_1 = 0;
  for (auto const idl : ideal_map) {
    if (pass_index[counter_1]) {
      new_ideal_map[counter_2] = ideal_map[counter_1];
      new_ideal_map[counter_2].index = counter_2;
      counter_2++;
    }
    counter_1++;
  }
}

// ---------------------------------
std::tuple<std::vector<float>, std::vector<uint8_t>> qaCluster::create_network_input(int sector, tpc2d& map2d, std::vector<int>& maxima_digits, std::vector<customCluster>& digit_map, int network_input_size)
{

  int index_shift_global = (2 * global_shift[0] + 1) * (2 * global_shift[1] + 1) * (2 * global_shift[2] + 1), index_shift_row = (2 * global_shift[0] + 1) * (2 * global_shift[1] + 1), index_shift_pad = (2 * global_shift[1] + 1);

  std::vector<float> input_vector(maxima_digits.size() * network_input_size, 0.f);
  std::vector<float> central_charges(maxima_digits.size(), 0.f);
  std::vector<uint8_t> flags(maxima_digits.size(), 0.f);

  int internal_idx = 0;
  for (unsigned int max = 0; max < maxima_digits.size(); max++) {
    auto const central_digit = digit_map[maxima_digits[max]];
    int row_offset = rowOffset(central_digit.row);
    int pad_offset = padOffset(central_digit.row);
    central_charges[max] = central_digit.qMax;
    bool compromised_charge = (central_charges[max] <= 0);
    if (compromised_charge) {
      LOG(warning) << "[" << sector << "] Central charge < 0 detected at index " << maxima_digits[max] << " = (sector: " << sector << ", row: " << central_digit.row << ", pad: " << central_digit.max_pad << ", time: " << central_digit.max_time << ") ! Continuing with input vector set to -1 everywhere...";
    }
    // reverse order as data is fed by transposing -> Fix in training data feeding?
    for (int row = 0; row < 2 * global_shift[2] + 1; row++) {
      for (int pad = 0; pad < 2 * global_shift[0] + 1; pad++) {
        for (int time = 0; time < 2 * global_shift[1] + 1; time++) {
          // int internal_idx = max * index_shift_global + time * index_shift_row + row * index_shift_pad + pad; // FIXME: THIS NEEDS TO BE CHANGED!!!
          // int internal_idx = max * index_shift_global + row * index_shift_row + pad * index_shift_pad + time;
          if (compromised_charge) {
            input_vector[internal_idx] = -1;
          } else {
            // (?) array_idx = map2d[1][central_digit[2] + 2 * global_shift[1] + 1 - time][central_digit[0]][central_digit[1] + pad + pad_offset];
            if (isBoundary(central_digit.row + row + row_offset - global_shift[2], central_digit.max_pad + pad + pad_offset - global_shift[0])) {
              input_vector[internal_idx] = -1;
            } else {
              int array_idx = map2d[1][central_digit.max_time + time][central_digit.row + row + row_offset][central_digit.max_pad + pad + pad_offset];
              if (array_idx > -1){
                if (normalization_mode == 0) {
                  input_vector[internal_idx] = digit_map[array_idx].qMax / 1024.f;
                } else if (normalization_mode == 1) {
                  input_vector[internal_idx] = digit_map[array_idx].qMax / central_charges[max];
                }
              }
            }
          }
          internal_idx++;
        }
      }
    }
    if(network_input_size > index_shift_global){
      input_vector[internal_idx] = central_digit.sector / o2::tpc::constants::MAXSECTOR;
      input_vector[internal_idx + 1] = central_digit.row / o2::tpc::constants::MAXGLOBALPADROW;
      input_vector[internal_idx + 2] = central_digit.max_pad / TPC_GEOM[central_digit.row][2];
      internal_idx+=3;
    }

    uint8_t flag = 0;
    // Boundary
    flag |= isBoundary(central_digit.row + row_offset - global_shift[2] + 1, central_digit.max_pad + pad_offset - global_shift[0] + 1) ? o2::tpc::ClusterNative::flagEdge : 0;
    flag |= isBoundary(central_digit.row + row_offset - global_shift[2] + 1, central_digit.max_pad + pad_offset - global_shift[0] - 1) ? o2::tpc::ClusterNative::flagEdge : 0;
    flag |= isBoundary(central_digit.row + row_offset - global_shift[2] - 1, central_digit.max_pad + pad_offset - global_shift[0] + 1) ? o2::tpc::ClusterNative::flagEdge : 0;
    flag |= isBoundary(central_digit.row + row_offset - global_shift[2] - 1, central_digit.max_pad + pad_offset - global_shift[0] - 1) ? o2::tpc::ClusterNative::flagEdge : 0;
    // Single pad and time
    flag |= ((map2d[1][central_digit.max_time + global_shift[1] + 1][central_digit.row + row_offset + global_shift[2]][central_digit.max_pad + global_shift[0] + pad_offset] == -1 &&
              map2d[1][central_digit.max_time + global_shift[1] - 1][central_digit.row + row_offset + global_shift[2]][central_digit.max_pad + global_shift[0] + pad_offset] == -1) ||
             (map2d[1][central_digit.max_time + global_shift[1]][central_digit.row + row_offset + global_shift[2]][central_digit.max_pad + global_shift[0] + pad_offset + 1] == -1 &&
              map2d[1][central_digit.max_time + global_shift[1]][central_digit.row + row_offset + global_shift[2]][central_digit.max_pad + global_shift[0] + pad_offset - 1] == -1))
              ? tpc::ClusterNative::flagSingle
              : 0;
    flags[max] = flag;
  }
  return std::make_tuple(input_vector, flags);
}

// ---------------------------------
void qaCluster::run_network_classification(int sector, tpc2d& map2d, std::vector<int>& maxima_digits, std::vector<customCluster>& digit_map, std::vector<customCluster>& network_map)
{

  // Loading the data
  std::vector<int> new_max_dig = maxima_digits;
  std::vector<uint8_t> flags_memory;
  std::vector<int> class_label(maxima_digits.size(), 10000); // some dummy label -> Should lead to segfault for network regression if not changed

  int index_shift_global = (2 * global_shift[0] + 1) * (2 * global_shift[1] + 1) * (2 * global_shift[2] + 1), index_shift_row = (2 * global_shift[0] + 1) * (2 * global_shift[1] + 1), index_shift_pad = (2 * global_shift[1] + 1);
  int network_class_size = 0, index_iroc_oroc_shift = 0;

  std::sort(maxima_digits.begin(), maxima_digits.end(), [&](int a, int b) { return digit_map[a].row < digit_map[b].row; });
  for(auto max : maxima_digits){
    index_iroc_oroc_shift++; // First index at which the row crosses the IROC - OROC1 boundary
    if(digit_map[max].row > 62){
      break;
    }
  }

  int current_max_idx = 0;

  for(int net_counter = 0; net_counter < (networkSplitIrocOroc == 0 ? 1 : 2)*network_classification_paths.size(); net_counter++){

    std::vector<int> eval_idcs;
    if(networkSplitIrocOroc){
      if(net_counter % 2 == 0){
        for(int idx = 0; idx < index_iroc_oroc_shift + 1; idx++){
          eval_idcs.push_back(maxima_digits[idx]);
        }
      } else {
        for(int idx = index_iroc_oroc_shift + 1; idx < maxima_digits.size(); idx++){
          eval_idcs.push_back(maxima_digits[idx]);
        }
      }
    } else {
      eval_idcs = maxima_digits;
    }

    size_t num_output_nodes = network_classification[net_counter].getNumOutputNodes()[0][1];
    std::vector<std::vector<float>> output_network_class;
    custom::resize_nested_container(output_network_class, std::vector<size_t>{maxima_digits.size(), num_output_nodes});

    int network_input_size = 1;
    for(auto v1 : network_classification[net_counter].getNumInputNodes()){
      for(auto v2 : v1){
        if(v2 > 0)
          network_input_size*=v2;
      }
    }

    for (int max_epoch = 0; max_epoch < std::ceil(eval_idcs.size() / (float)networkInputSize); max_epoch++) {

      std::vector<int> investigate_maxima;
      custom::fill_container_by_range(investigate_maxima, eval_idcs, max_epoch * networkInputSize, ((max_epoch + 1) * networkInputSize) - 1);
      auto [input_vector, flags] = create_network_input(sector, map2d, investigate_maxima, digit_map, network_input_size);
      custom::append_to_container(flags_memory, flags);

      int eval_size = input_vector.size() / network_input_size;

      std::vector<float> out_net = network_classification[net_counter].inference<float, float>(input_vector);

      for (int idx = 0; idx < eval_size; idx++) {

        // int current_max_idx = max_epoch * networkInputSize + idx;
        for (int s = 0; s < num_output_nodes; s++) {
          output_network_class[current_max_idx][s] = out_net[idx * num_output_nodes + s];
        }

        if(verbose >= 5 && idx == 100) {
          std::cout << "Corresponding network input:" << std::endl;
          std::cout << "([" << std::endl;
          for (int row = 0; row < 2 * global_shift[2] + 1; row++) {
            for (int pad = 0; pad < 2 * global_shift[0] + 1; pad++) {
              std::cout << "[";
              for (int time = 0; time < 2 * global_shift[1] + 1; time++) {
                std::cout << input_vector[idx*network_input_size + row*index_shift_row + pad*index_shift_pad + time];
                if(time != 2 * global_shift[1]){
                  std::cout << ", ";
                } else if((row+1)*(pad+1)*(time+1) == index_shift_global){
                  std::cout << "]" << std::endl;
                } else {
                  std::cout << "]," << std::endl;
                }
              }
            }
          }
          std::cout << "])" << std::endl;
          std::cout << "Corresponding network output to data above was: ";
          for (int i = 0; i < num_output_nodes; i++) {
            std::cout << output_network_class[current_max_idx][i] << ", ";
          }
          std::cout << std::endl;
        }

        float tmp_class_label = -1;
        if (num_output_nodes == 1) {
          tmp_class_label = (int)(output_network_class[current_max_idx][0] > networkClassThres);
        } else {
          tmp_class_label = std::min((int)std::distance(output_network_class[current_max_idx].begin(), std::max_element(output_network_class[current_max_idx].begin(), output_network_class[current_max_idx].end())), (int)network_regression_paths.size()) - 1;
        }

        if (tmp_class_label > 0 && class_label[current_max_idx] > 0) {
          class_label[current_max_idx] = (int)tmp_class_label;
          new_max_dig[current_max_idx] = maxima_digits[current_max_idx];
          digit_map[maxima_digits[current_max_idx]].label = (int)tmp_class_label;
          network_class_size++;
        } else {
          class_label[current_max_idx] = 0;
          new_max_dig[current_max_idx] = -1;
          digit_map[maxima_digits[current_max_idx]].label = 0;
        }

        current_max_idx++;
      }
    }
  }

  maxima_digits.clear();
  network_map.clear();
  maxima_digits.resize(network_class_size);
  network_map.resize(network_class_size);
  int counter_max_out = 0, counter_max_dig = 0;
  for (int max : new_max_dig) {
    if (max > -1) {
      maxima_digits[counter_max_out] = max; // only change number of digit maxima if regression network is called
      // IDEA:
      // Create regression net for every possible class (e.g. 0 to 5).
      // Create entry for network_map n_class times.
      // Sort entries for maxima_digits based on class label.
      // Eval networks in usual form for each point but choose the network according to class label.
      // Write output to network_map and inflate size of maxima digits for assignments.
      uint8_t tmp_flag = (flags_memory[counter_max_dig] | (class_label[counter_max_dig] > 1 ? tpc::ClusterNative::flagSplitTime : 0));
      network_map[counter_max_out] = digit_map[max];
      network_map[counter_max_out].flag = tmp_flag;
      network_map[counter_max_out].label = class_label[counter_max_dig];
      network_map[counter_max_out].index = counter_max_out;
      counter_max_out++;
    }
    counter_max_dig++;
  }

  LOG(info) << "[" << sector << "] Classification network done!";
}

// ---------------------------------
void qaCluster::run_network_regression(int sector, tpc2d& map2d, std::vector<int>& maxima_digits, std::vector<customCluster>& digit_map, std::vector<customCluster>& network_map, std::vector<std::array<float,2>>& momentum_vector_map)
{

  int index_shift_global = (2 * global_shift[0] + 1) * (2 * global_shift[1] + 1) * (2 * global_shift[2] + 1), index_shift_row = (2 * global_shift[0] + 1) * (2 * global_shift[1] + 1), index_shift_pad = (2 * global_shift[1] + 1);
  int num_output_classes = (int)(network_regression_paths.size() / (networkSplitIrocOroc == 0 ? 1 : 2)), num_output_nodes_regression = network_regression[0].getNumOutputNodes()[0][1]; // Expects regression networks to be sorted by class output and have 5 outputs -> [pad1, pad2, ..., time1, time2, ..., sigma_pad1, ..., sigma_time1, ..., qRatio1, ...]
  std::vector<int> digit_idcs;

  int index_iroc_oroc_shift = 0;
  for(auto max : maxima_digits){
    index_iroc_oroc_shift++; // First index at which the row crosses the IROC - OROC1 boundary; Expects sorting from classificaiton network function
    if(digit_map[max].row > 62){
      break;
    }
  }
  
  int out_net_idx = 0, idx_offset = 0;
  std::vector<customCluster> output_network_reg;
  for(int split_counter = 0; split_counter < 2; split_counter++){

    std::vector<int> eval_idcs;
    if(split_counter % 2 == 0){
      for(int idx = 0; idx < index_iroc_oroc_shift + 1; idx++){
        eval_idcs.push_back(maxima_digits[idx]);
      }
      idx_offset = 0;
    } else {
      for(int idx = index_iroc_oroc_shift + 1; idx < maxima_digits.size(); idx++){
        eval_idcs.push_back(maxima_digits[idx]);
      }
      idx_offset = index_iroc_oroc_shift + 1;
    }

    int total_num_points = 0;
    std::vector<std::vector<int>> sorted_digit_idx(6); // maybe good to change this once to a fixed number of possible classes
    for(int max = 0; max < eval_idcs.size(); max++){
      if(network_map[max + idx_offset].label > 0){
        total_num_points++;
        sorted_digit_idx[network_map[max + idx_offset].label].push_back(max);
      }
    }

    // std::vector<int> corresponding_index_output(total_num_points, 0);
    // total_num_points = 0;
    // int tmp_idx = 0;
    // for(int max = 0; max < eval_idcs.size(); max++){
    //   total_num_points += network_map[max + idx_offset].label;
    //   if(max < (eval_idcs.size()-1) && network_map[max + idx_offset].label > 0){
    //     corresponding_index_output[tmp_idx+1] = corresponding_index_output[tmp_idx] + network_map[max + idx_offset].label;
    //     tmp_idx++;
    //   }
    // }

    for(int class_idx = 1; class_idx < num_output_classes + 1; class_idx++){

      if(sorted_digit_idx[class_idx].size()>0){

        int network_input_size = 1;
        for(auto v1 : network_regression[2*(class_idx-1) + split_counter].getNumInputNodes()){
          for(auto v2 : v1){
            if(v2 > 0)
              network_input_size*=v2;
          }
        }

        for(int max_epoch = 0; max_epoch < std::ceil(sorted_digit_idx[class_idx].size() / (float) networkInputSize); max_epoch++){
          std::vector<int> investigate_maxima;
          custom::fill_container_by_range(investigate_maxima, sorted_digit_idx[class_idx], max_epoch * networkInputSize, ((max_epoch + 1) * networkInputSize) - 1);
          for(int& inv_max : investigate_maxima){
            inv_max = eval_idcs[inv_max];
          }
          auto [input_vector, flags] = create_network_input(sector, map2d, investigate_maxima, digit_map, network_input_size);

          int eval_size = input_vector.size() / network_input_size;
          std::vector<float> out_net = network_regression[2*(class_idx-1) + split_counter].inference<float, float>(input_vector);

          for(int idx = 0; idx < eval_size; idx++){
            int digit_idx = max_epoch * networkInputSize + idx;
            if(digit_idx >= sorted_digit_idx[class_idx].size()){
              break;
            }
            int digit_max_idx = sorted_digit_idx[class_idx][digit_idx];
            customCluster net_cluster = network_map[digit_max_idx + idx_offset];
            for(int subclass = 0; subclass < class_idx; subclass++){
              int net_idx = idx * class_idx * num_output_nodes_regression + subclass;
              customCluster new_net_cluster = net_cluster;
              new_net_cluster.cog_pad += out_net[net_idx + 0 * class_idx];
              new_net_cluster.cog_time += out_net[net_idx + 1 * class_idx];
              new_net_cluster.sigmaPad = out_net[net_idx + 2 * class_idx];
              new_net_cluster.sigmaTime = out_net[net_idx + 3 * class_idx];
              new_net_cluster.qTot = (new_net_cluster.qMax * out_net[net_idx + 4 * class_idx]); // Change for normalization mode
              new_net_cluster.index = out_net_idx;

              if(num_output_nodes_regression > 5){
                momentum_vector_map.push_back({out_net[net_idx + 5 * class_idx], out_net[net_idx + 6 * class_idx]}); //, out_net[net_idx + 7 * class_idx]});
              }

              if(round(new_net_cluster.cog_pad) > TPC_GEOM[new_net_cluster.row][2] || round(new_net_cluster.cog_time) > max_time[sector] || round(new_net_cluster.cog_pad) < 0 || round(new_net_cluster.cog_time) < 0){
                LOG(warning) << "[" << sector << "] Stepping over boundaries! row: " << new_net_cluster.row << "; pad: " << new_net_cluster.cog_pad << " (net: " << out_net[net_idx + 0 * class_idx] << ") / " << TPC_GEOM[new_net_cluster.row][2] << "; time: " << new_net_cluster.cog_time << " (net: " << out_net[net_idx + 1 * class_idx] << ") / " << max_time[sector] << ". Resetting cluster center-of-gravity to maximum position.";
                new_net_cluster.cog_pad = new_net_cluster.max_pad;
                new_net_cluster.cog_time = new_net_cluster.max_time;
                if(verbose >= 5) {
                  std::cout << "Corresponding network input:" << std::endl;
                  std::cout << "([" << std::endl;
                  for (int row = 0; row < 2 * global_shift[2] + 1; row++) {
                    for (int pad = 0; pad < 2 * global_shift[0] + 1; pad++) {
                      std::cout << "[";
                      for (int time = 0; time < 2 * global_shift[1] + 1; time++) {
                        std::cout << input_vector[idx*network_input_size + row*index_shift_row + pad*index_shift_pad + time];
                        if(time != 2 * global_shift[1]){
                          std::cout << ", ";
                        } else if((row+1)*(pad+1)*(time+1) == index_shift_global){
                          std::cout << "]" << std::endl;
                        } else {
                          std::cout << "]," << std::endl;
                        }
                      }
                    }
                  }
                  std::cout << "])" << std::endl;
                  std::cout << "Corresponding network output to data above was: ";
                  for(int i = 0; i < num_output_nodes_regression; i++){
                    if(i != num_output_nodes_regression-1){
                      std::cout << out_net[net_idx + i * class_idx] << ", ";
                    } else {
                      std::cout << out_net[net_idx + i * class_idx] << std::endl;
                    }
                  }
                }
              }

              output_network_reg.push_back(new_net_cluster);
              digit_idcs.push_back(eval_idcs[digit_max_idx]);
              out_net_idx++;
            }
          }
        }
      }
    }
  }
  for(int cls_idx = 0; cls_idx < output_network_reg.size(); cls_idx++){
    output_network_reg[cls_idx].index = cls_idx;
  }

  network_map.clear();
  network_map = output_network_reg;
  
  digit_map.clear();
  digit_map = output_network_reg;

  // digit_map = output_network_reg; // -> This causes huge trouble. Not sure why...
  // maxima_digits.resize(output_network_reg.size());
  // for(int max = 0; max < maxima_digits.size(); max++){
  //   maxima_digits[max] = digit_map[digit_idcs[max]].index;
  // }

  LOG(info) << "[" << sector << "] Regression network done!";
}

// ---------------------------------
void qaCluster::overwrite_map2d(int sector, tpc2d& map2d, std::vector<customCluster>& element_map, std::vector<int>& element_idx, int mode)
{
  custom::fill_nested_container(map2d[mode], -1);
  for (unsigned int id = 0; id < element_idx.size(); id++) {
    // LOG(info) << id << " / " << element_idx.size() << " / " << element_map.size() << "; " << round(element_map[element_idx[id]].cog_time) + global_shift[1] << " / " << max_time[sector] << "; " << element_map[element_idx[id]].row + global_shift[2] + rowOffset(element_map[element_idx[id]].row) << " / " << o2::tpc::constants::MAXGLOBALPADROW + 3*global_shift[2] << "; "<< round(element_map[element_idx[id]].cog_pad) + global_shift[0] + padOffset(element_map[element_idx[id]].row) << " / " << TPC_GEOM[o2::tpc::constants::MAXGLOBALPADROW - 1][2] + 1 + 2 * global_shift[0];
    if(round(element_map[element_idx[id]].cog_time) <= max_time[sector] && element_map[element_idx[id]].row < o2::tpc::constants::MAXGLOBALPADROW && round(element_map[element_idx[id]].cog_pad) <= TPC_GEOM[o2::tpc::constants::MAXGLOBALPADROW - 1][2])
      map2d[mode][round(element_map[element_idx[id]].cog_time) + global_shift[1]][element_map[element_idx[id]].row + global_shift[2] + rowOffset(element_map[element_idx[id]].row)][round(element_map[element_idx[id]].cog_pad) + global_shift[0] + padOffset(element_map[element_idx[id]].row)] = id;
  }
}

// ---------------------------------
int qaCluster::test_neighbour(std::array<int, 3> index, std::array<int, 2> nn, tpc2d& map2d, int mode)
{
  if(index[0] < o2::tpc::constants::MAXGLOBALPADROW && index[1] + nn[0] <= TPC_GEOM[index[0]][2]){
    return map2d[mode][index[2] + global_shift[1] + nn[1]][index[0] + rowOffset(index[0]) + global_shift[2]][index[1] + padOffset(index[0]) + global_shift[0] + nn[0]];
  } else {
    return -1;
  }
}

// ---------------------------------
void qaCluster::runQa(int sector)
{

  std::vector<int> maxima_digits; // , digit_isNoise, digit_isQED, digit_isValid;
  std::vector<std::array<float, 3>> digit_clusterizer_map;
  std::vector<std::array<float,2>> momentum_vector_map;
  std::vector<customCluster> digit_map, ideal_map, network_map, native_map;
  std::vector<std::vector<std::vector<std::vector<int>>>> tagger_maps(looper_tagger_granularity.size());
  std::vector<int> track_cluster_to_ideal_assignment; // First is digit_max index, then 5 possible assignemts of track_clusters

  LOG(info) << "--- Starting process for sector " << sector << " ---";

  if (mode.find(std::string("native")) != std::string::npos) {
    read_native(sector, digit_map, native_map);
  } else {
    read_digits(sector, digit_map);
  }

  if(mode.find(std::string("path")) != std::string::npos){
    std::vector<customCluster> tracking_clusters_sector = tracking_clusters[sector];
    for(int counter = 0; counter < tracking_clusters_sector.size(); counter++){
      tracking_clusters_sector[counter].index = counter;
    }
    ideal_map = tracking_clusters_sector;
  } else {
    if(realData){
      read_native(sector, native_map, ideal_map);
    } else {
      read_ideal(sector, ideal_map);
    }
  }

  num_total_ideal_max += ideal_map.size();

  tpc2d map2d = init_map2d(sector);

  std::vector<int> ideal_idx(ideal_map.size());
  std::iota(ideal_idx.begin(), ideal_idx.end(), 0);

  if (mode.find(std::string("looper_tagger")) != std::string::npos) {
    for (int counter = 0; counter < looper_tagger_granularity.size(); counter++) {
      tagger_maps[counter] = looper_tagger(sector, counter, ideal_map, ideal_idx);
      // remove_loopers_ideal(sector, counter, tagger_maps[counter], ideal_max_map, ideal_cog_map, ideal_max_q, ideal_cog_q, ideal_sigma_map, ideal_mclabels);
    }
  }

  fill_map2d(sector, map2d, digit_map, ideal_map, -1);

  if ((mode.find(std::string("network")) == std::string::npos) && (mode.find(std::string("native")) == std::string::npos)) {
    find_maxima(sector, map2d, digit_map, maxima_digits);
    // if (mode.find(std::string("looper_tagger")) != std::string::npos) {
    //   for (int counter = 0; counter < looper_tagger_granularity.size(); counter++) {
    //     remove_loopers_digits(sector, counter, tagger_maps[counter], digit_map, maxima_digits);
    //   }
    // }
    // if (mode.find(std::string("clusterizer")) != std::string::npos) {
    //   native_clusterizer(map2d, digit_map, maxima_digits, digit_q, digit_clusterizer_map, digit_clusterizer_q);
    // }
    num_total_digit_max += maxima_digits.size();
    overwrite_map2d(sector, map2d, digit_map, maxima_digits, 1);
  } else {
    if (mode.find(std::string("native")) == std::string::npos) {
      find_maxima(sector, map2d, digit_map, maxima_digits);
      // if (mode.find(std::string("looper_tagger")) != std::string::npos) {
      //   for (int counter = 0; counter < looper_tagger_granularity.size(); counter++) {
      //     remove_loopers_digits(sector, counter, tagger_maps[counter], digit_map, maxima_digits);
      //   }
      // }
      if (mode.find(std::string("network_class")) != std::string::npos || mode.find(std::string("network_full")) != std::string::npos) {
        run_network_classification(sector, map2d, maxima_digits, digit_map, network_map); // classification
        // if (mode.find(std::string("clusterizer")) != std::string::npos) {
        //   native_clusterizer(map2d, digit_map, maxima_digits, digit_q, digit_clusterizer_map, digit_clusterizer_q);
        // }
      } 
      if (mode.find(std::string("network_reg")) != std::string::npos || mode.find(std::string("network_full")) != std::string::npos) {
        run_network_regression(sector, map2d, maxima_digits, digit_map, network_map, momentum_vector_map); // classification + regression
      }
      maxima_digits.clear();
      maxima_digits.resize(output_network_reg.size());
      std::iota(maxima_digits.begin(), maxima_digits.end(), 0);
      
      num_total_digit_max += maxima_digits.size();
      overwrite_map2d(sector, map2d, digit_map, maxima_digits, 1);
    } else {
      num_total_digit_max += native_map.size();
      maxima_digits.resize(native_map.size());
      std::iota(std::begin(maxima_digits), std::end(maxima_digits), 0);
      // if (mode.find(std::string("looper_tagger")) != std::string::npos) {
      //   for (int counter = 0; counter < looper_tagger_granularity.size(); counter++) {
      //     remove_loopers_digits(sector, counter, tagger_maps[counter], digit_map, maxima_digits);
      //     remove_loopers_native(sector, counter, tagger_maps[counter], native_map, maxima_digits);
      //   }
      // }
      // overwrite_map2d(sector, map2d, native_map, maxima_digits, 1);
    }
  }

  std::vector<int> assigned_ideal(ideal_map.size(), 0);
  std::vector<std::array<int, 25>> assignments_dig_to_id(ideal_map.size());
  std::vector<int> assigned_digit(maxima_digits.size(), 0);
  std::vector<std::array<int, 25>> assignments_id_to_dig(maxima_digits.size());
  std::vector<float> clone_order(maxima_digits.size(), 0), fractional_clones_vector(maxima_digits.size(), 0);

  custom::fill_nested_container(assignments_dig_to_id, -1);
  custom::fill_nested_container(assignments_id_to_dig, -1);
  custom::fill_nested_container(assigned_digit, 0);
  custom::fill_nested_container(assigned_ideal, 0);
  custom::fill_nested_container(clone_order, 0);

  number_of_digit_max[sector] += maxima_digits.size();
  number_of_ideal_max[sector] += ideal_map.size();

  for (auto const idl : ideal_map) {
    if (idl.qTot >= threshold_cogq && idl.qMax >= threshold_maxq) {
      number_of_ideal_max_findable[sector]++;
    }
  }

  // Level-1 loop: Goes through the layers specified by the adjacency matrix <-> Loop of possible distances
  int layer_count = 0;
  for (int layer = 0; layer < adj_mat.size(); layer++) {

    // Level-2 loop: Goes through the elements of the adjacency matrix at distance d, n times to assign neighbours iteratively
    for (int nn = 0; nn < adj_mat[layer].size(); nn++) {

      // Level-3 loop: Goes through all digit maxima and checks neighbourhood for potential ideal maxima
      for (unsigned int locdigit = 0; locdigit < maxima_digits.size(); locdigit++) {
        int current_neighbour = test_neighbour({digit_map[maxima_digits[locdigit]].row, (int)round(digit_map[maxima_digits[locdigit]].cog_pad), (int)round(digit_map[maxima_digits[locdigit]].cog_time)}, adj_mat[layer][nn], map2d, 0);
        if (current_neighbour > -1 && current_neighbour < (int)ideal_map.size()) {
          assignments_id_to_dig[locdigit][layer_count + nn] = (assigned_digit[locdigit] == 0 && assigned_ideal[current_neighbour] == 0) ? current_neighbour : -1;
        } else if(current_neighbour >= (int)ideal_map.size()){
          LOG(warning) << "[" << sector << "] (assignments_id_to_dig) Invalid map adress at index: " << locdigit << " / " << maxima_digits.size() << ": " << current_neighbour << " (/" << ideal_map.size() << ") - Accessing (row, pad, time) = (" << digit_map[maxima_digits[locdigit]].row << ", " << round(digit_map[maxima_digits[locdigit]].cog_pad) << ", " << round(digit_map[maxima_digits[locdigit]].cog_time) << ")";
        }
      }
      if (verbose >= 4)
        LOG(info) << "[" << sector << "] Done with assignment for digit maxima, layer " << layer;

      // Level-3 loop: Goes through all ideal maxima and checks neighbourhood for potential digit maxima
      for (unsigned int locideal = 0; locideal < ideal_map.size(); locideal++) {
        int current_neighbour = test_neighbour({ideal_map[locideal].row, (int)round(ideal_map[locideal].cog_pad), (int)round(ideal_map[locideal].cog_time)}, adj_mat[layer][nn], map2d, 1);
        if (current_neighbour > -1 && current_neighbour < digit_map.size()) {
          assignments_dig_to_id[locideal][layer_count + nn] = (assigned_ideal[locideal] == 0 && assigned_digit[current_neighbour] == 0) ? current_neighbour : -1;
        } else if(current_neighbour >= (int)digit_map.size()){
          LOG(warning) << "[" << sector << "] (assignments_dig_to_id) Invalid map adress at index: " << locideal << " / " << ideal_map.size() << ": " << current_neighbour << " (/" << digit_map.size() << ") - Accessing (row, pad, time) = (" << ideal_map[locideal].row << ", " << round(ideal_map[locideal].cog_pad) << ", " << round(ideal_map[locideal].cog_time) << ")";
        }
      }
      if (verbose >= 4)
        LOG(info) << "[" << sector << "] Done with assignment for ideal maxima, layer " << layer;
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
    for (unsigned int locideal = 0; locideal < ideal_map.size(); locideal++) {
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
  std::vector<int> ideal_tagged(ideal_map.size(), 0), digit_tagged(maxima_digits.size(), 0);
  std::vector<std::vector<int>> ideal_tag_label((int)ideal_map.size(), std::vector<int>((int)tagger_maps.size(), -1)), digit_tag_label((int)maxima_digits.size(), std::vector<int>((int)tagger_maps.size(), -1));

  if (mode.find(std::string("looper_tagger")) != std::string::npos) {
    int tm_counter = 0, remove_ideal = 0, remove_digit = 0;
    for (auto tm : tagger_maps) {
      int counter = 0;
      for (auto idl : ideal_map) {
        ideal_tag_label[counter][tm_counter] = tm[round(idl.cog_time)][idl.row][round(idl.cog_pad)];
        ideal_tagged[counter] = (int)(((bool)ideal_tagged[counter]) || (tm[round(idl.cog_time)][idl.row][round(idl.cog_pad)] > -1));
        remove_ideal += ideal_tagged[counter];
        counter += 1;
      }
      counter = 0;
      for (int idx_dig : maxima_digits) {
        digit_tag_label[counter][tm_counter] = tm[digit_map[idx_dig].max_time][digit_map[idx_dig].row][digit_map[idx_dig].max_pad];
        digit_tagged[counter] = (int)(((bool)digit_tagged[counter]) || (tm[digit_map[idx_dig].max_time][digit_map[idx_dig].row][digit_map[idx_dig].max_pad] > -1));
        remove_digit += digit_tagged[counter];
        counter += 1;
      }
      tm_counter += 1;
    }
    num_total_ideal_max -= remove_ideal;
    num_total_digit_max -= remove_digit;
  }

  // Checks the number of assignments that have been made with the above loops
  int count_elements_findable = 0, count_elements_dig = 0, count_elements_id = 0;
  for (unsigned int locideal = 0; locideal < assignments_dig_to_id.size(); locideal++) {
    if (!ideal_tagged[locideal]) { // if region is tagged, don't use ideal cluster for ECF calculation
      count_elements_id = 0;
      count_elements_findable = 0;
      for (int idx_dig : assignments_dig_to_id[locideal]) {
        if (checkIdx(idx_dig) && !digit_tagged[idx_dig]) {
          count_elements_id += 1;
          if (ideal_map[locideal].qTot >= threshold_cogq && ideal_map[locideal].qMax >= threshold_maxq) { // FIXME: assignemts to an ideal cluster which are findable? -> Digit maxima which satisfy the criteria not ideal clsuters?!
            count_elements_findable += 1;
          }
        }
      }
      // if (verbose >= 5 && (locideal%10000)==0) LOG(info) << "Count elements: " << count_elements_id << " locideal: " << locideal << " assignments_ideal: " << assignments_ideal[count_elements_id];
      assignments_ideal[sector][count_elements_id] += 1;
      assignments_ideal_findable[sector][count_elements_findable] += 1;
    }
  }
  for (unsigned int locdigit = 0; locdigit < assignments_id_to_dig.size(); locdigit++) {
    if (!digit_tagged[locdigit]) { // if region is tagged, don't use digit maximum for ECF calculation
      count_elements_dig = 0;
      count_elements_findable = 0;
      for (int idx_idl : assignments_id_to_dig[locdigit]) {
        if (checkIdx(idx_idl) && !ideal_tagged[idx_idl]) {
          count_elements_dig += 1;
          if (ideal_map[idx_idl].qTot >= threshold_cogq && ideal_map[idx_idl].qMax >= threshold_maxq) {
            count_elements_findable += 1;
          }
        }
      }
      assignments_digit[sector][count_elements_dig] += 1;
      assignments_digit_findable[sector][count_elements_findable] += 1;
    }
  }

  if (verbose >= 3)
    LOG(info) << "[" << sector << "] Done checking the number of assignments";

  // Clone-rate (Integer)
  for (unsigned int locdigit = 0; locdigit < assignments_id_to_dig.size(); locdigit++) {
    if (!digit_tagged[locdigit]) {
      int count_links = 0;
      float count_weighted_links = 0;
      for (int idx_idl : assignments_id_to_dig[locdigit]) {
        if (checkIdx(idx_idl)) {
          count_links++;
          int count_links_second = 0;
          for (auto elem_dig : assignments_dig_to_id[idx_idl]) {
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
    clones[sector] += clone_order[locdigit];
  }

  // Clone-rate (fractional)
  for (unsigned int locideal = 0; locideal < assignments_dig_to_id.size(); locideal++) {
    if (!ideal_tagged[locideal]) {
      int count_links = 0;
      for (int idx_dig : assignments_dig_to_id[locideal]) {
        if (checkIdx(idx_dig)) {
          count_links += 1;
        }
      }
      for (int idx_dig : assignments_dig_to_id[locideal]) {
        if (checkIdx(idx_dig)) {
          fractional_clones_vector[idx_dig] += 1.f / (float)count_links;
        }
      }
    }
  }
  for (float elem_frac : fractional_clones_vector) {
    if (elem_frac > 1) {
      fractional_clones[sector] += elem_frac - 1;
    }
  }

  if (verbose >= 3)
    LOG(info) << "[" << sector << "] Done determining the clone rate";

  if (verbose >= 4) {
    for (int ass = 0; ass < 25; ass++) {
      LOG(info) << "Number of assignments to one digit maximum (#assignments " << ass << "): " << assignments_digit[sector][ass];
      LOG(info) << "Number of assignments to one ideal maximum (#assignments " << ass << "): " << assignments_ideal[sector][ass] << "\n";
    }
  }

  if (mode.find(std::string("native")) != std::string::npos && create_output == 1) {

    if (verbose >= 3)
      LOG(info) << "Native-Ideal assignment...";

    // creating training data for the neural network
    int data_size = maxima_digits.size();

    std::vector<std::array<customCluster, 2>> native_ideal_assignemnt;
    std::array<customCluster, 2> current_element;

    std::fill(assigned_ideal.begin(), assigned_ideal.end(), 0);
    std::fill(assigned_digit.begin(), assigned_digit.end(), 0);

    // Some useful variables
    int map_dig_idx = 0, map_q_idx = 0, check_assignment = 0, index_assignment = -1, current_idx_id = -1, current_idx_dig = -1, row_offset = 0, pad_offset = 0;
    float distance_assignment = 100000.f, current_distance_dig_to_id = 0, current_distance_id_to_dig = 0;
    bool is_min_dist = true;

    for (int max_point = 0; max_point < data_size; max_point++) {
      auto const dig = digit_map[maxima_digits[max_point]];
      if (!digit_tagged[max_point]) {
        check_assignment = 0;
        index_assignment = -1;
        distance_assignment = 100000.f;
        is_min_dist = true;
        for (int i = 0; i < 25; i++) {
          // Checks all ideal maxima assigned to one digit maximum by calculating mutual distance
          current_idx_id = assignments_id_to_dig[max_point][i];
          if (checkIdx(current_idx_id) && !ideal_tagged[current_idx_id]) {
            if ((ideal_map[current_idx_id].qTot < threshold_cogq && ideal_map[current_idx_id].qMax < threshold_maxq) || (assigned_ideal[current_idx_id] != 0)) {
              is_min_dist = false;
              break;
            } else {
              current_distance_dig_to_id = std::pow((native_map[max_point].cog_time - ideal_map[current_idx_id].cog_time), 2) + std::pow((native_map[max_point].cog_pad - ideal_map[current_idx_id].cog_pad), 2);
              // if the distance is less than the previous one check if update should be made
              if (current_distance_dig_to_id < distance_assignment) {
                for (int j = 0; j < 25; j++) {
                  current_idx_dig = assignments_dig_to_id[current_idx_id][j];
                  if (checkIdx(current_idx_dig) && !digit_tagged[current_idx_dig]) {
                    if (assigned_digit[current_idx_dig] == 0) {
                      // calculate mutual distance from current ideal CoG to all assigned digit maxima. Update if and only if distance is minimal. Else do not assign.
                      current_distance_id_to_dig = std::pow((native_map[current_idx_dig].cog_time - ideal_map[current_idx_id].cog_time), 2) + std::pow((native_map[current_idx_dig].cog_pad - ideal_map[current_idx_id].cog_pad), 2);
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
                  native_map[max_point].mcTrkId = ideal_map[current_idx_id].mcTrkId;
                  native_map[max_point].mcEvId = ideal_map[current_idx_id].mcEvId;
                  native_map[max_point].mcSrcId = ideal_map[current_idx_id].mcSrcId;
                  current_element = {native_map[max_point], ideal_map[current_idx_id]};
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
    file_in << outputPath << "/native_ideal_" << sector << ".root";
    TFile* outputFileNativeIdeal = new TFile(file_in.str().c_str(), "RECREATE");
    TTree* native_ideal = new TTree("native_ideal", "tree");

    // int native_writer_map_size = native_writer_map.size();
    // native_writer_map.resize(native_writer_map_size + native_ideal_assignemnt.size());

    float sec = sector, nat_row = 0, nat_time = 0, nat_pad = 0, nat_sigma_time = 0, nat_sigma_pad = 0,  nat_qTot = 0, nat_qMax = 0, id_sigma_pad = 0, id_sigma_time = 0, id_row = 0, id_time = 0, id_pad = 0, id_qTot = 0, id_qMax = 0;
    native_ideal->Branch("sector", &sec);
    native_ideal->Branch("native_row", &nat_row);
    native_ideal->Branch("native_cog_time", &nat_time);
    native_ideal->Branch("native_cog_pad", &nat_pad);
    native_ideal->Branch("native_sigma_time", &nat_sigma_time);
    native_ideal->Branch("native_sigma_pad", &nat_sigma_pad);
    native_ideal->Branch("native_qMax", &nat_qMax);
    native_ideal->Branch("native_qTot", &nat_qTot);
    native_ideal->Branch("ideal_row", &id_row);
    native_ideal->Branch("ideal_cog_time", &id_time);
    native_ideal->Branch("ideal_cog_pad", &id_pad);
    native_ideal->Branch("ideal_sigma_time", &id_sigma_time);
    native_ideal->Branch("ideal_sigma_pad", &id_sigma_pad);
    native_ideal->Branch("ideal_qMax", &id_qMax);
    native_ideal->Branch("ideal_qTot", &id_qTot);

    int elem_counter = 0;
    for (auto const elem : native_ideal_assignemnt) {
      nat_row = elem[0].row;
      nat_pad = elem[0].cog_pad;
      nat_time = elem[0].cog_time;
      nat_sigma_pad = elem[0].sigmaPad;
      nat_sigma_time = elem[0].sigmaTime;
      nat_qTot = elem[0].qTot;
      nat_qMax = elem[0].qMax;
      id_row = elem[1].row;
      id_pad = elem[1].cog_pad;
      id_time = elem[1].cog_time;
      id_sigma_pad = elem[1].sigmaPad;
      id_sigma_time = elem[1].sigmaTime;
      id_qTot = elem[1].qTot;
      id_qMax = elem[1].qMax;
      native_ideal->Fill();

      // if (write_native_file) {
      //   native_writer_map[native_writer_map_size + elem_counter] = elem[0];
      // }
      // elem_counter++;
    }

    native_ideal->Write();
    outputFileNativeIdeal->Close();

    m.lock();
    if (write_native_file) {
      int native_writer_map_size = native_writer_map.size();
      int cluster_counter = 0, total_counter = 0;
      for(auto const cls : native_map){
        if(!digit_tagged[total_counter]){
          cluster_counter++;
        }
        total_counter++;
      }
      native_writer_map.resize(native_writer_map_size + cluster_counter);
      cluster_counter = 0;
      total_counter = 0;
      for(auto const cls : native_map){
        if(!digit_tagged[total_counter]){
          native_writer_map[native_writer_map_size + cluster_counter] = cls;
          GlobalPosition2D conv_pos = custom::convertSecRowPadToXY(cls.sector, cls.row, cls.cog_pad, tpcmap);
          native_writer_map[native_writer_map_size + cluster_counter].X = conv_pos.X();
          native_writer_map[native_writer_map_size + cluster_counter].Y = conv_pos.Y();
          cluster_counter++;
        }
        total_counter++;
      }
    }
    m.unlock();

  }

  if (mode.find(std::string("network")) != std::string::npos && create_output == 1) {

    if (verbose >= 3)
      LOG(info) << "[" << sector << "] Network-Ideal assignment...";

    bool addMomentumData = (mode.find(std::string("track_cluster")) != std::string::npos);

    if(addMomentumData){
      float precision = 1.f/64; // Defined by ClusterNative.h: unpackPad and unpackTime: scalePadPacked = scaleTimePacked = 64
      track_cluster_to_ideal_assignment.resize(ideal_map.size());
      custom::fill_nested_container(track_cluster_to_ideal_assignment, -1);
      int cluster_counter = -1; // Starting at -1 due to continue statement in the following loop
      for(auto cls : tracking_clusters[sector]){
        int check_pad = round(cls.cog_pad), check_time = round(cls.cog_time);
        int idl_idx = map2d[0][check_time + global_shift[1]][cls.row + global_shift[2] + rowOffset(cls.row)][check_pad + global_shift[0] + padOffset(cls.row)];
        if(idl_idx == -1 && ((std::abs(std::abs(cls.cog_pad - (int)cls.cog_pad) - 0.5) < precision) || (std::abs(std::abs(cls.cog_time - (int)cls.cog_time) - 0.5) < precision))){
          if(std::abs(std::abs(check_pad - cls.cog_pad) - 0.5) < precision){
            if(std::abs(check_pad - cls.cog_pad) < 0.5){
              check_pad += 1; // Check the adjacent pad just to be sure (if you chose the lower pad before because (round(val) - val) < 0.5, then choose the upper one now)
            } else {
              check_pad -= 1;
            }
          }
          if(std::abs(std::abs(check_time - cls.cog_time) - 0.5) < precision){
            if(std::abs(check_time - cls.cog_time) < 0.5){
              check_time += 1;
            } else {
              check_time -= 1;
            }
          }
          idl_idx = map2d[0][check_time + global_shift[1]][cls.row + global_shift[2] + rowOffset(cls.row)][check_pad + global_shift[0] + padOffset(cls.row)];
        }
        cluster_counter++;
        if(idl_idx == -1){
          continue;
        }
        if(ideal_map[idl_idx].index!=-1){
          track_cluster_to_ideal_assignment[ideal_map[idl_idx].index] = cluster_counter;
        }
        //if ((cls.row == ideal_map[idl_idx].row) && (std::abs(cls.cog_time - ideal_map[idl_idx].cog_time) < precision) && (std::abs(cls.cog_pad - ideal_map[idl_idx].cog_pad) < precision)){
        //  track_cluster_to_ideal_assignment[idl_idx] = cluster_counter;
        //}
      }
    }

    // creating training data for the neural network
    int data_size = maxima_digits.size();

    std::vector<std::array<customCluster, 2>> network_ideal_assignment;
    std::array<customCluster, 2> current_element;

    std::fill(assigned_ideal.begin(), assigned_ideal.end(), 0);
    std::fill(assigned_digit.begin(), assigned_digit.end(), 0);

    // Checks if digit is assigned / has non-looper assignments
    std::vector<int> digit_has_non_looper_assignments(maxima_digits.size(), -1); // -1 = has no assignments, 0 = has assignment but is looper, n = has n non-looper assignments
    std::vector<std::vector<int>> digit_non_looper_assignment_labels(maxima_digits.size());
    for (int dig_max = 0; dig_max < maxima_digits.size(); dig_max++) {
      bool digit_has_assignment = false;
      for (int ass : assignments_id_to_dig[dig_max]) {
        if (ass != -1) {
          digit_has_assignment = true;
          bool is_tagged = false;
          if (ideal_tagged[ass]) {
            for (int lbl : ideal_tag_label[ass]) {
              is_tagged |= (lbl != -1 ? (ideal_map[ass].mcTrkId == lbl) : false);
            }
          }
          if (!is_tagged) {
            digit_has_non_looper_assignments[dig_max] == -1 ? digit_has_non_looper_assignments[dig_max] = 1 : digit_has_non_looper_assignments[dig_max] += 1;
            digit_non_looper_assignment_labels[dig_max].push_back(ass);
          }
        }
      }
      if (digit_has_non_looper_assignments[dig_max] == -1 && digit_has_assignment) {
        digit_has_non_looper_assignments[dig_max] = 0;
      }
    }

    // Some useful variables
    int map_dig_idx = 0, map_q_idx = 0, check_assignment = 0, index_assignment = -1, current_idx_id = -1, current_idx_dig = -1;
    float distance_assignment = 100000.f, current_distance_dig_to_id = 0, current_distance_id_to_dig = 0;
    bool is_min_dist = true;

    for (int max_point = 0; max_point < data_size; max_point++) {
      auto const dig = digit_map[maxima_digits[max_point]];
      if (!digit_tagged[max_point]) {
        check_assignment = 0;
        index_assignment = -1;
        distance_assignment = 100000.f;
        is_min_dist = true;
        for (int i = 0; i < 25; i++) {
          // Checks all ideal maxima assigned to one digit maximum by calculating mutual distance
          current_idx_id = assignments_id_to_dig[max_point][i];
          if (checkIdx(current_idx_id) && !ideal_tagged[current_idx_id]) {
            auto const idl = ideal_map[current_idx_id];
            if ((idl.qTot < threshold_cogq && idl.qMax < threshold_maxq) || (assigned_ideal[current_idx_id] != 0)) {
              is_min_dist = false;
              break;
            } else {
              current_distance_dig_to_id = std::pow((network_map[max_point].cog_time - idl.cog_time), 2) + std::pow((network_map[max_point].cog_pad - idl.cog_pad), 2);
              // if the distance is less than the previous one check if update should be made
              if (current_distance_dig_to_id < distance_assignment) {
                for (int j = 0; j < 25; j++) {
                  current_idx_dig = assignments_dig_to_id[current_idx_id][j];
                  if (checkIdx(current_idx_dig) && !digit_tagged[current_idx_dig]) {
                    if (assigned_digit[current_idx_dig] == 0) {
                      // calculate mutual distance from current ideal CoG to all assigned digit maxima. Update if and only if distance is minimal. Else do not assign.
                      current_distance_id_to_dig = std::pow((network_map[current_idx_dig].cog_time - idl.cog_time), 2) + std::pow((network_map[current_idx_dig].cog_pad - idl.cog_pad), 2);
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
                  network_map[max_point].mcTrkId = ideal_map[current_idx_id].mcTrkId;
                  network_map[max_point].mcEvId = ideal_map[current_idx_id].mcEvId;
                  network_map[max_point].mcSrcId = ideal_map[current_idx_id].mcSrcId;
                  current_element = {network_map[max_point], ideal_map[current_idx_id]};
                }
              }
            }
          }
        }

        if (check_assignment > 0 && is_min_dist) {
          network_ideal_assignment.push_back(current_element);
        }
      }
    }
    
    if (verbose >= 3)
      LOG(info) << "Done performing network-ideal assignment. Writing to file...";

    std::stringstream file_in;
    file_in << outputPath << "/network_ideal_" << sector << ".root";
    TFile* outputFileNetworkIdeal = new TFile(file_in.str().c_str(), "RECREATE");
    TTree* network_ideal = new TTree("network_ideal", "tree");

    float sec = sector, id_class = -999, net_row = 0, net_time = 0, net_pad = 0, net_sigma_time = 0, net_sigma_pad = 0, net_qTot = 0, net_qMax = 0, net_momX = 1000, net_momY = 1000, net_momZ = 1000, id_sigma_pad = 0, id_sigma_time = 0, id_row = 0, id_time = 0, id_pad = 0, id_qTot = 0, id_qMax = 0, net_idx = 0, id_idx = 0, id_momX = 1000, id_momY = 1000, id_momZ = 1000, id_mom = 1000, net_momY_X = 1000, net_momZ_X = 1000;
    network_ideal->Branch("sector", &sec);
    network_ideal->Branch("network_row", &net_row);
    network_ideal->Branch("network_cog_time", &net_time);
    network_ideal->Branch("network_cog_pad", &net_pad);
    network_ideal->Branch("network_sigma_time", &net_sigma_time);
    network_ideal->Branch("network_sigma_pad", &net_sigma_pad);
    network_ideal->Branch("network_qMax", &net_qMax);
    network_ideal->Branch("network_qTot", &net_qTot);
    network_ideal->Branch("network_index", &net_idx);
    // network_ideal->Branch("network_momentumX", &net_momX);
    // network_ideal->Branch("network_momentumY", &net_momY);
    // network_ideal->Branch("network_momentumZ", &net_momZ);
    network_ideal->Branch("network_momentumY_X", &net_momY_X);
    network_ideal->Branch("network_momentumZ_X", &net_momZ_X);
    network_ideal->Branch("ideal_class", &id_class);
    network_ideal->Branch("ideal_row", &id_row);
    network_ideal->Branch("ideal_cog_time", &id_time);
    network_ideal->Branch("ideal_cog_pad", &id_pad);
    network_ideal->Branch("ideal_sigma_time", &id_sigma_time);
    network_ideal->Branch("ideal_sigma_pad", &id_sigma_pad);
    network_ideal->Branch("ideal_qMax", &id_qMax);
    network_ideal->Branch("ideal_qTot", &id_qTot);
    network_ideal->Branch("ideal_index", &id_idx);
    network_ideal->Branch("ideal_momentum", &id_mom);
    network_ideal->Branch("ideal_momentumX", &id_momX);
    network_ideal->Branch("ideal_momentumY", &id_momY);
    network_ideal->Branch("ideal_momentumZ", &id_momZ);

    // int netive_writer_map_size = netive_writer_map.size();
    // netive_writer_map.resize(native_writer_map_size + network_ideal_assignment.size());

    bool momentum_vector_estimate = (network_regression[0].getNumOutputNodes()[0][1] > 5);
    int elem_counter = 0;
    for (auto elem : network_ideal_assignment) {
      id_mom = 1000;
      id_momX = 1000;
      id_momY = 1000;
      id_momZ = 1000;
      id_class = -999;
      net_row = elem[0].row;
      net_pad = elem[0].cog_pad;
      net_time = elem[0].cog_time;
      net_sigma_pad = elem[0].sigmaPad;
      net_sigma_time = elem[0].sigmaTime;
      net_qTot = elem[0].qTot;
      net_qMax = elem[0].qMax;
      net_idx = elem[0].index;
      id_class = elem[1].mcTrkId;
      id_row = elem[1].row;
      id_pad = elem[1].cog_pad;
      id_time = elem[1].cog_time;
      id_sigma_pad = elem[1].sigmaPad;
      id_sigma_time = elem[1].sigmaTime;
      id_qTot = elem[1].qTot;
      id_qMax = elem[1].qMax;
      id_idx = elem[1].index;
      id_class = digit_has_non_looper_assignments[elem[0].index];
      if(momentum_vector_estimate){
        net_momY_X = momentum_vector_map[net_idx][0];
        net_momZ_X = momentum_vector_map[net_idx][1];
      }
      if(momentum_vector_estimate && addMomentumData && (id_idx != -1)){
        // net_momX = momentum_vector_map[net_idx][0];
        // net_momY = momentum_vector_map[net_idx][1];
        // net_momZ = momentum_vector_map[net_idx][2];
        if(track_cluster_to_ideal_assignment[id_idx] != -1){
          id_mom = std::sqrt(std::pow(momentum_vectors[sector][track_cluster_to_ideal_assignment[id_idx]][0],2) + std::pow(momentum_vectors[sector][track_cluster_to_ideal_assignment[id_idx]][1],2) + std::pow(momentum_vectors[sector][track_cluster_to_ideal_assignment[id_idx]][2],2));
          id_momX = momentum_vectors[sector][track_cluster_to_ideal_assignment[id_idx]][0];
          id_momY = momentum_vectors[sector][track_cluster_to_ideal_assignment[id_idx]][1];
          id_momZ = momentum_vectors[sector][track_cluster_to_ideal_assignment[id_idx]][2];
        }
      }
      network_ideal->Fill();

      // if (write_native_file) {
      //   elem[0].mcTrkId = elem[1].mcTrkId;
      //   elem[0].mcSrcId = elem[1].mcSrcId;
      //   elem[0].mcEvId = elem[1].mcEvId;
      //   native_writer_map[native_writer_map_size + elem_counter] = elem[0];
      // }
      // elem_counter++;
    }

    network_ideal->Write();
    outputFileNetworkIdeal->Close();

    m.lock();
    if (write_native_file) {
      int native_writer_map_size = native_writer_map.size();
      int cluster_counter = 0, total_counter = 0;
      for(auto const cls : network_map){
        if(!digit_tagged[total_counter]){
          cluster_counter++;
        }
        total_counter++;
      }
      native_writer_map.resize(native_writer_map_size + cluster_counter);
      cluster_counter = 0;
      total_counter = 0;
      for(auto const cls : network_map){
        if(!digit_tagged[total_counter]){
          native_writer_map[native_writer_map_size + cluster_counter] = cls;
          GlobalPosition2D conv_pos = custom::convertSecRowPadToXY(cls.sector, cls.row, cls.cog_pad, tpcmap);
          native_writer_map[native_writer_map_size + cluster_counter].X = conv_pos.X();
          native_writer_map[native_writer_map_size + cluster_counter].Y = conv_pos.Y();
          cluster_counter++;
        }
        total_counter++;
      }
    }
    m.unlock();

  }

  if (mode.find(std::string("training_data")) != std::string::npos && create_output == 1) {

    // If momentum data is present assign momenta of tracking clusters to respective ideal clusters in map2d
    bool addMomentumData = (mode.find(std::string("training_data_mom")) != std::string::npos);

    if(addMomentumData){
      float precision = 1.f/64; // Defined by ClusterNative.h: unpackPad and unpackTime: scalePadPacked = scaleTimePacked = 64
      track_cluster_to_ideal_assignment.resize(ideal_map.size());
      custom::fill_nested_container(track_cluster_to_ideal_assignment, -1);
      int cluster_counter = -1; // Starting at -1 due to continue statement in the following loop
      for(auto cls : tracking_clusters[sector]){
        int check_pad = round(cls.cog_pad), check_time = round(cls.cog_time);
        int idl_idx = map2d[0][check_time + global_shift[1]][cls.row + global_shift[2] + rowOffset(cls.row)][check_pad + global_shift[0] + padOffset(cls.row)];
        if(idl_idx == -1 && ((std::abs(std::abs(cls.cog_pad - (int)cls.cog_pad) - 0.5) < precision) || (std::abs(std::abs(cls.cog_time - (int)cls.cog_time) - 0.5) < precision))){
          if(std::abs(std::abs(check_pad - cls.cog_pad) - 0.5) < precision){
            if(std::abs(check_pad - cls.cog_pad) < 0.5){
              check_pad += 1; // Check the adjacent pad just to be sure (if you chose the lower pad before because (round(val) - val) < 0.5, then choose the upper one now)
            } else {
              check_pad -= 1;
            }
          }
          if(std::abs(std::abs(check_time - cls.cog_time) - 0.5) < precision){
            if(std::abs(check_time - cls.cog_time) < 0.5){
              check_time += 1;
            } else {
              check_time -= 1;
            }
          }
          idl_idx = map2d[0][check_time + global_shift[1]][cls.row + global_shift[2] + rowOffset(cls.row)][check_pad + global_shift[0] + padOffset(cls.row)];
        }
        cluster_counter++;
        if(idl_idx == -1){
          continue;
        }
        track_cluster_to_ideal_assignment[idl_idx] = cluster_counter;
        // if ((cls.row == ideal_map[idl_idx].row) && (std::abs(cls.cog_time - ideal_map[idl_idx].cog_time) < precision) && (std::abs(cls.cog_pad - ideal_map[idl_idx].cog_pad) < precision)){
        //   track_cluster_to_ideal_assignment[idl_idx] = cluster_counter;
        // } else {
        //   LOG(warning) << "Ideal cluster at same index but outside precision (ideal) (sector: " << ideal_map[idl_idx].sector << "; row: " << ideal_map[idl_idx].row << "; pad: " << ideal_map[idl_idx].cog_pad << "; time: " << ideal_map[idl_idx].cog_time << ") ; (tracking) (sector: " << cls.sector << "; row: " << cls.row << "; pad: " << cls.cog_pad << "; time: " << cls.cog_time << ")";
        // }
      }
    }

    // Checks if digit is assigned / has non-looper assignments
    std::vector<int> digit_has_non_looper_assignments(maxima_digits.size(), -1); // -1 = has no assignments, 0 = has assignment but is looper, n = has n non-looper assignments
    std::vector<std::vector<int>> digit_non_looper_assignment_labels(maxima_digits.size());
    for (int dig_max = 0; dig_max < maxima_digits.size(); dig_max++) {
      bool digit_has_assignment = false;
      for (int ass : assignments_id_to_dig[dig_max]) {
        if (ass != -1) {
          digit_has_assignment = true;
          bool is_tagged = false;
          if (ideal_tagged[ass]) {
            for (int lbl : ideal_tag_label[ass]) {
              is_tagged |= (lbl != -1 ? (ideal_map[ass].mcTrkId == lbl) : false);
            }
          }
          if (!is_tagged) {
            digit_has_non_looper_assignments[dig_max] == -1 ? digit_has_non_looper_assignments[dig_max] = 1 : digit_has_non_looper_assignments[dig_max] += 1;
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
    overwrite_map2d(sector, map2d, digit_map, index_digits, 0);

    if (verbose >= 3)
      LOG(info) << "[" << sector << "] Creating training data...";

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

    std::vector<int> tr_data_Y_class(data_size, -1);
    std::vector<std::array<std::array<float, 8>, 5>> tr_data_Y_reg(data_size); // for each data element: for all possible assignments: {trY_time, trY_pad, trY_sigma_pad, trY_sigma_time, trY_q}
    custom::fill_nested_container(tr_data_Y_reg, -999);

    std::fill(assigned_ideal.begin(), assigned_ideal.end(), 0);
    std::fill(assigned_digit.begin(), assigned_digit.end(), 0);

    // For MC info, only if MC data is used
    o2::MCTrack current_track;
    std::vector<float> cluster_pT, cluster_eta, cluster_mass, cluster_p;
    std::vector<int> cluster_isPrimary, cluster_isTagged;

    if(!realData){
      cluster_pT.resize(data_size, -1);
      cluster_eta.resize(data_size, -1);
      cluster_mass.resize(data_size, -1);
      cluster_p.resize(data_size, -1);
      cluster_isPrimary.resize(data_size, -1);
      cluster_isTagged.resize(data_size, -1);
    }

    // Some useful variables
    int map_dig_idx = 0, map_q_idx = 0, check_assignment = 0, index_assignment = -1, current_idx_id = -1, current_idx_dig = -1, row_offset = 0, pad_offset = 0, class_label = 0;
    float distance_assignment = 100000.f, current_distance_dig_to_id = 0, current_distance_id_to_dig = 0;
    bool is_min_dist = true, is_tagged = false, find_track_path = (mode.find(std::string("training_data_mom_path")) != std::string::npos);
    float distance_cluster_track_path = -1.f, cog_tr_pad_offset = (find_track_path ? -0.5 : 0);

    for (int max_point = 0; max_point < data_size; max_point++) {
      // is_tagged = (bool)digit_tagged[max_point];
      auto const dig = digit_map[maxima_digits[max_point]];
      row_offset = rowOffset(dig.row);
      pad_offset = padOffset(dig.row);
      map_dig_idx = map2d[1][dig.max_time + global_shift[1]][dig.row + row_offset + global_shift[2]][dig.max_pad + pad_offset + global_shift[0]];
      if (checkIdx(map_dig_idx)) {
        float q_max = dig.qMax;
        for (int row = 0; row < mat_size_row; row++) {
          for (int pad = 0; pad < mat_size_pad; pad++) {
            for (int time = 0; time < mat_size_time; time++) {
              map_q_idx = map2d[0][dig.max_time + time][dig.row + row + row_offset][dig.max_pad + pad + pad_offset];
              if (map_q_idx == -1) {
                if (isBoundary(dig.row + row + row_offset - global_shift[2], dig.max_pad + pad + pad_offset - global_shift[0])) {
                  tr_data_X[max_point][row][pad][time] = -1;
                } else {
                  tr_data_X[max_point][row][pad][time] = 0;
                }
              } else {
                if (normalization_mode == 0) {
                  tr_data_X[max_point][row][pad][time] = digit_map[map_q_idx].qMax / 1024.f;
                } else if (normalization_mode == 1) {
                  tr_data_X[max_point][row][pad][time] = digit_map[map_q_idx].qMax / q_max;
                }
              }
            }
          }
        }

        tr_data_Y_class[max_point] = digit_has_non_looper_assignments[max_point];
        
        if(!realData){
          cluster_isTagged[max_point] = (bool)digit_tagged[max_point];
        }

        std::vector<int> sorted_idcs;
        if (tr_data_Y_class[max_point] > 0) {
          customCluster idl;
          std::vector<float> distance_array(digit_has_non_looper_assignments[max_point], -1);
          for (int counter = 0; counter < digit_has_non_looper_assignments[max_point]; counter++) {
            int ideal_idx = digit_non_looper_assignment_labels[max_point][counter];
            if(find_track_path && track_cluster_to_ideal_assignment[ideal_idx] != -1){
              idl = tracking_paths[sector][track_cluster_to_ideal_assignment[ideal_idx]];
            } else {
              idl = ideal_map[ideal_idx];
            }
            distance_array[counter] = std::pow((dig.max_time - idl.cog_time), 2) + std::pow((dig.max_pad - idl.cog_pad), 2);
          }

          distance_array.size() > 1 ? sorted_idcs = custom::sort_indices(distance_array) : sorted_idcs = {0};
          for (int counter = 0; counter < digit_has_non_looper_assignments[max_point]; counter++) {
            int ideal_idx = digit_non_looper_assignment_labels[max_point][sorted_idcs[counter]];
            if(find_track_path && track_cluster_to_ideal_assignment[ideal_idx] != -1){
              idl = tracking_paths[sector][track_cluster_to_ideal_assignment[ideal_idx]];
            } else {
              idl = ideal_map[ideal_idx];
            }
            tr_data_Y_reg[max_point][counter][0] = idl.cog_pad - dig.max_pad + cog_tr_pad_offset;       // pad: Matching is done on integers for digit maxima, but track paths and clusters are offset by 0.5 (center of pad)
            tr_data_Y_reg[max_point][counter][1] = idl.cog_time - dig.max_time;                         // time
            tr_data_Y_reg[max_point][counter][2] = idl.sigmaPad;                                        // sigma pad
            tr_data_Y_reg[max_point][counter][3] = idl.sigmaTime;                                       // sigma time
            if (normalization_mode == 0) {
              tr_data_Y_reg[max_point][counter][4] = idl.qTot / 1024.f;
            } else if (normalization_mode == 1) {
              tr_data_Y_reg[max_point][counter][4] = idl.qTot / q_max;
            }

            if (counter == 0 && !realData && idl.mcSrcId != -1) {
              current_track = mctracks[idl.mcSrcId][idl.mcEvId][idl.mcTrkId];
              cluster_pT[max_point] = current_track.GetPt();
              cluster_eta[max_point] = current_track.GetEta();
              cluster_mass[max_point] = current_track.GetMass();
              cluster_p[max_point] = current_track.GetP();
              cluster_isPrimary[max_point] = (int)current_track.isPrimary();
            }

            if(addMomentumData){
              if(track_cluster_to_ideal_assignment[ideal_idx] != -1 && distance_array[sorted_idcs[counter]] < training_data_distance_cluster_path){ // Distance measurement to check if track path is even close to assigned track cluster
                tr_data_Y_reg[max_point][counter][5] = momentum_vectors[sector][track_cluster_to_ideal_assignment[ideal_idx]][0];
                tr_data_Y_reg[max_point][counter][6] = momentum_vectors[sector][track_cluster_to_ideal_assignment[ideal_idx]][1];
                tr_data_Y_reg[max_point][counter][7] = momentum_vectors[sector][track_cluster_to_ideal_assignment[ideal_idx]][2];
              }
            }
          }
        }
      }
    }
    if (verbose >= 3)
      LOG(info) << "[" << sector << "] Done creating training data. Writing to file...";

    std::stringstream file_in;
    file_in << outputPath << "/training_data_" << sector << ".root";
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
    std::array<std::array<float, 8>, 5> trY; // [branch][data]; data = {trY_time, trY_pad, trY_sigma_pad, trY_sigma_time, trY_q}
    custom::fill_nested_container(trY, -1);
    std::array<std::string, 8> branches{"out_reg_pad", "out_reg_time", "out_reg_sigma_pad", "out_reg_sigma_time", "out_reg_qTotOverqMax", "momentumX", "momentumY", "momentumZ"};
    for (int reg_br = 0; reg_br < 8; reg_br++) {
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

    if(!realData){
      tr_data->Branch("cluster_pT", &pT);
      tr_data->Branch("cluster_eta", &eta);
      tr_data->Branch("cluster_mass", &mass);
      tr_data->Branch("cluster_p", &p);
      tr_data->Branch("cluster_isPrimary", &isPrimary);
      tr_data->Branch("cluster_isTagged", &isTagged);
    }

    // Filling elements
    for (int element = 0; element < data_size; element++) {
      atomic_unit = tr_data_X[element];
      trY = tr_data_Y_reg[element];
      class_val = tr_data_Y_class[element];
      idx_sector = sector;
      idx_row = digit_map[maxima_digits[element]].row;
      idx_pad = digit_map[maxima_digits[element]].max_pad;
      idx_time = digit_map[maxima_digits[element]].max_time;
      if(!realData){
        pT = cluster_pT[element];
        eta = cluster_eta[element];
        mass = cluster_mass[element];
        p = cluster_p[element];
        isPrimary = cluster_isPrimary[element];
        isTagged = cluster_isTagged[element];
      }
      tr_data->Fill();
    }
    tr_data->Write();
    outputFileTrData->Close();

  }

  if (mode.find(std::string("write_ideal")) != std::string::npos && create_output == 1) {
    m.lock();
    if (write_native_file) {
      int native_writer_map_size = native_writer_map.size();
      int cluster_counter = 0, total_counter = 0;
      for(auto const cls : ideal_map){
        if(!ideal_tagged[total_counter]){
          cluster_counter++;
        }
        total_counter++;
      }
      native_writer_map.resize(native_writer_map_size + cluster_counter);
      cluster_counter = 0;
      total_counter = 0;
      for(auto const cls : ideal_map){
        if(!ideal_tagged[total_counter]){
          native_writer_map[native_writer_map_size + cluster_counter] = cls;
          GlobalPosition2D conv_pos = custom::convertSecRowPadToXY(cls.sector, cls.row, cls.cog_pad, tpcmap);
          native_writer_map[native_writer_map_size + cluster_counter].X = conv_pos.X();
          native_writer_map[native_writer_map_size + cluster_counter].Y = conv_pos.Y();
          cluster_counter++;
        }
        total_counter++;
      }
    }
    m.unlock();
  }

  LOG(info) << "--- Done with sector " << sector << " ---\n";

}

// ---------------------------------
void qaCluster::run(ProcessingContext& pc)
{

  if (mode.find(std::string("looper_tagger")) != std::string::npos) {
    for (int i = 0; i < looper_tagger_granularity.size(); i++) {
      LOG(info) << "Looper tagger active, settings: granularity " << looper_tagger_granularity[i] << ", time-window: " << looper_tagger_timewindow[i] << ", pad-window: " << looper_tagger_padwindow[i] << ", threshold (num. points): " << looper_tagger_threshold_num[i] << ", threshold (Q): " << looper_tagger_threshold_q[i] << ", operational mode: " << looper_tagger_opmode;
    }
  }

  if(!realData){
    LOG(info) << "Reading kinematics information...";
    read_kinematics(mctracks);
  }

  if(mode.find(std::string("training_data_mom")) != std::string::npos || mode.find(std::string("track_cluster")) != std::string::npos || realData){
    LOG(info) << "Reading tracking information...";
    read_tracking_clusters(!realData);
  }

  if (mode.find(std::string("matcher")) != std::string::npos){

    number_of_ideal_max.fill(0);
    number_of_digit_max.fill(0);
    number_of_ideal_max_findable.fill(0);
    clones.fill(0);
    fractional_clones.fill(0.f);

    // init array
    custom::fill_nested_container(assignments_ideal, 0);
    custom::fill_nested_container(assignments_digit, 0);
    custom::fill_nested_container(assignments_ideal_findable, 0);
    custom::fill_nested_container(assignments_digit_findable, 0);

    numThreads = std::min(numThreads, 36);
    thread_group group;
    if (numThreads > 1) {
      int loop_counter = 0;
      for (int sector : tpc_sectors) {
        loop_counter++;
        group.create_thread(boost::bind(&qaCluster::runQa, this, sector));
        if ((sector + 1) % numThreads == 0 || sector + 1 == o2::tpc::constants::MAXSECTOR || loop_counter == tpc_sectors.size()) {
          group.join_all();
        }
      }
    } else {
      for (int sector : tpc_sectors) {
        runQa(sector);
      }
    }
    LOG(info) << "Per sector QA done. Creating ECF values.";

    unsigned int number_of_ideal_max_sum = 0, number_of_digit_max_sum = 0, number_of_ideal_max_findable_sum = 0;
    float clones_sum = 0, fractional_clones_sum = 0;
    custom::sum_nested_container(assignments_ideal, number_of_ideal_max_sum);
    custom::sum_nested_container(assignments_digit, number_of_digit_max_sum);
    custom::sum_nested_container(assignments_ideal_findable, number_of_ideal_max_findable_sum);
    custom::sum_nested_container(clones, clones_sum);
    custom::sum_nested_container(fractional_clones, fractional_clones_sum);

    LOG(info) << "------- RESULTS -------\n";
    LOG(info) << "Number of digit maxima (after exclusions): " << num_total_digit_max;
    LOG(info) << "Number of ideal maxima (after exclusions): " << num_total_ideal_max << "\n";
    LOG(info) << "Number of digit maxima: (before exclusion): " << number_of_digit_max_sum;
    LOG(info) << "Number of ideal maxima: (before exclusion): " << number_of_ideal_max_sum;

    unsigned int efficiency_normal = 0;
    unsigned int efficiency_findable = 0;
    for (int ass = 0; ass < 25; ass++) {
      int ass_dig = 0, ass_id = 0;
      for (int s : tpc_sectors) {
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
      for (int s : tpc_sectors) {
        ass_dig += assignments_digit_findable[s][ass];
        ass_id += assignments_ideal_findable[s][ass];
      }
      if (ass == 0) {
        ass_id -= (number_of_ideal_max_sum - number_of_ideal_max_findable_sum);
      }
      LOG(info) << "Number of finable assigned digit maxima (#assignments " << ass << "): " << ass_dig;
      LOG(info) << "Number of finable assigned ideal maxima (#assignments " << ass << "): " << ass_id << "\n";
      if (ass > 0) {
        for (int s : tpc_sectors) {
          efficiency_findable += assignments_ideal_findable[s][ass];
        }
      }
    }

    int fakes_dig = 0;
    for (int s : tpc_sectors) {
      fakes_dig += assignments_digit[s][0];
    }
    int fakes_id = 0;
    for (int s : tpc_sectors) {
      fakes_id += assignments_ideal[s][0];
    }

    LOG(info) << "Efficiency - Number of assigned (ideal -> digit) clusters: " << efficiency_normal << " (" << (float)efficiency_normal * 100 / (float)num_total_ideal_max << "% of ideal maxima)";
    LOG(info) << "Efficiency (findable) - Number of assigned (ideal -> digit) clusters: " << efficiency_findable << " (" << (float)efficiency_findable * 100 / (float)number_of_ideal_max_findable_sum << "% of ideal maxima)";
    LOG(info) << "Clones (Int, clone-order >= 2 for ideal cluster): " << clones_sum << " (" << (float)clones_sum * 100 / (float)num_total_digit_max << "% of digit maxima)";
    LOG(info) << "Clones (Float, fractional clone-order): " << fractional_clones_sum << " (" << (float)fractional_clones_sum * 100 / (float)num_total_digit_max << "% of digit maxima)";
    LOG(info) << "Fakes for digits (number of digit hits that can't be assigned to any ideal hit): " << fakes_dig << " (" << (float)fakes_dig * 100 / (float)num_total_digit_max << "% of digit maxima)";
    LOG(info) << "Fakes for ideal (number of ideal hits that can't be assigned to any digit hit): " << fakes_id << " (" << (float)fakes_id * 100 / (float)num_total_ideal_max << "% of ideal maxima)";

    if (mode.find(std::string("looper_tagger")) != std::string::npos && create_output == 1) {
      LOG(info) << "------- Merging looper tagger regions -------";
      std::stringstream command;
      command << "hadd -k -f " << outputPath << "/looper_tagger.root " << outputPath << "/looper_tagger_*.root";
      gSystem->Exec(command.str().c_str());
    }

    if (mode.find(std::string("training_data")) != std::string::npos && create_output == 1) {
      LOG(info) << "------- Merging training data -------";
      std::stringstream command;
      command << "hadd -k -f " << outputPath << "/training_data.root " << outputPath << "/training_data_*.root";
      gSystem->Exec(command.str().c_str());
    }

    if (mode.find(std::string("native")) != std::string::npos && create_output == 1) {
      LOG(info) << "------- Merging native-ideal assignments -------";
      std::stringstream command;
      command << "hadd -k -f " << outputPath << "/native_ideal.root " << outputPath << "/native_ideal_*.root";
      gSystem->Exec(command.str().c_str());
    }

    if (mode.find(std::string("network")) != std::string::npos && create_output == 1) {
      LOG(info) << "------- Merging network-ideal assignments -------";
      std::stringstream command;
      command << "hadd -k -f " << outputPath << "/network_ideal.root " << outputPath << "/network_ideal_*.root";
      gSystem->Exec(command.str().c_str());
    }

    if (create_output == 1 && write_native_file == 1) {
      custom::writeStructToRootFile(outputPath + "/" + outFileCustomClusters, "data", native_writer_map);
      if ((mode.find(std::string("network")) != std::string::npos) || (mode.find(std::string("native")) != std::string::npos) || (mode.find(std::string("write_ideal")) != std::string::npos)) {
        write_custom_native(pc, native_writer_map);
      }
    }

    if (remove_individual_files > 0) {
      LOG(info) << "!!! Removing sector-individual files !!!";
      std::stringstream command;
      command << "rm -rf " << outputPath;
      gSystem->Exec((command.str() + "/looper_tagger_*.root").c_str());
      gSystem->Exec((command.str() + "/training_data_*.root").c_str());
      gSystem->Exec((command.str() + "/native_ideal_*.root").c_str());
      gSystem->Exec((command.str() + "/network_ideal_*.root").c_str());
    }
  }

  if(mode.find(std::string("tpc_geometry")) != std::string::npos) {
    std::vector<std::array<float, 4>> sector_xy_coords = tpcmap.getSectorsXY();
    std::vector<std::string> coords_branch_names = {"X1", "Y1", "X2", "Y2"};
    custom::writeTabularToRootFile(coords_branch_names, sector_xy_coords, outputPath + "/tpc_geometry.root", "tpc_geomtery", "TPC sector boundary geometry");
  }

  pc.services().get<ControlService>().endOfStream();
  pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
}

// ----- Input and output processors -----

// customize clusterers and cluster decoders to process immediately what comes in
void customize(std::vector<o2::framework::CompletionPolicy>& policies)
{
  policies.push_back(o2::framework::CompletionPolicyHelpers::consumeWhenAllOrdered(".*(?:TPC|tpc).*[w,W]riter.*"));
}

void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  std::vector<ConfigParamSpec> options{
    {"output-path", VariantType::String, ".", {"Path to the folder where output files should be written"}},
    {"simulation-path", VariantType::String, ".", {"Path to the folder where simulation files are taken from"}},
    {"create-output", VariantType::Int, 1, {"Create output, specific to any given mode."}},
    {"write-native-file", VariantType::Int, 0, {"Whether or not to write a custom native file"}},
    {"outfile-native", VariantType::String, "tpc-native-cluster-custom.root", {"Path to native file"}},
    {"outfile-clusters", VariantType::String, "custom_clusters.root", {"Path to custom clusters file"}},
    {"native-file-single-branch", VariantType::Int, 1, {"Whether or not to write a single branch in the custom native file"}},
    {"tpc-sectors", VariantType::String, "0-35", {"TPC sector range, e.g. 5-7,8,9"}}
  };
  std::swap(workflowOptions, options);
}

// ---------------------------------
#include "Framework/runDataProcessing.h"

DataProcessorSpec processIdealClusterizer(ConfigContext const& cfgc, std::vector<InputSpec>& inputs, std::vector<OutputSpec>& outputs)
{

  // A copy of the global workflow options from customize() to pass to the task
  std::unordered_map<std::string, std::string> options_map{
    {"create-output", cfgc.options().get<std::string>("create-output")},
    {"output-path", cfgc.options().get<std::string>("output-path")},
    {"simulation-path", cfgc.options().get<std::string>("simulation-path")},
    {"write-native-file", cfgc.options().get<std::string>("write-native-file")},
    {"outfile-native", cfgc.options().get<std::string>("outfile-native")},
    {"outfile-clusters", cfgc.options().get<std::string>("outfile-clusters")},
    {"native-file-single-branch", cfgc.options().get<std::string>("native-file-single-branch")},
    {"tpc-sectors", cfgc.options().get<std::string>("tpc-sectors")}
  };

  if (cfgc.options().get<int>("write-native-file")) {
    // setOutputAllocator("CLUSTERNATIVE", true, outputRegions.clustersNative, std::make_tuple(gDataOriginTPC, mSpecConfig.sendClustersPerSector ? (DataDescription) "CLUSTERNATIVETMP" : (DataDescription) "CLUSTERNATIVE", NSectors, clusterOutputSectorHeader), sizeof(o2::tpc::ClusterCountIndex));
    for (int i : o2::RangeTokenizer::tokenize<int>(cfgc.options().get<std::string>("tpc-sectors"))) {
      outputs.emplace_back(o2::header::gDataOriginTPC, "CLUSTERNATIVE", i, Lifetime::Timeframe); // Dropping incomplete Lifetime::Transient?
      outputs.emplace_back(o2::header::gDataOriginTPC, "CLNATIVEMCLBL", i, Lifetime::Timeframe); // Dropping incomplete Lifetime::Transient?
    }
  }

  return DataProcessorSpec{
    "tpc-qa-ideal",
    inputs,
    outputs,
    adaptFromTask<qaCluster>(options_map),
    Options{
      {"verbose", VariantType::Int, 0, {"Verbosity level"}},
      {"mode", VariantType::String, "matcher,training_data", {"Enables different settings (e.g. creation of training data for NN, running with tpc-native clusters). Options are: training_data, native, network_classification, network_regression, network_full, clusterizer"}},
      {"real-data", VariantType::Int, 0, {"If real data is used. MC kinematics will not be read then"}},
      {"normalization-mode", VariantType::Int, 1, {"Normalization: 0 = normalization by 1024.f; 1 = normalization by q_center"}},
      {"use-max-cog", VariantType::Int, 1, {"Use maxima for assignment = 0, use CoG's = 1"}},
      {"max-time", VariantType::Int, -1, {"Maximum time allowed for reading data."}},
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
      {"infile-tracks", VariantType::String, "tpctracks.root", {"Input file name (tracks)"}},
      {"network-data-output", VariantType::String, "network_out.root", {"Input file for the network output"}},
      // {"network-regression-paths", VariantType::ArrayString, std::vector<std::string>{""}, {"Absolute path(s) to the network file(s) (cluster regression)"}}, // Change to array
      // {"network-classification-paths", VariantType::ArrayString, std::vector<std::string>{""}, {"Absolute path(s) to the network file(s) (cluster classification)"}}, // Change to array
      {"network-regression-paths", VariantType::String, "", {"Absolute path(s) to the network file(s) (cluster regression), semicolon-delimited"}}, // Change to array
      {"network-classification-paths", VariantType::String, "", {"Absolute path(s) to the network file(s) (cluster classification), semicolon-delimited"}}, // Change to array
      {"network-split-iroc-oroc", VariantType::Int, 0, {"Whether to use different networks for IROC and OROC's. If 1: network-path = network_iroc1;network_oroc1;network_iroc2;network_oroc2; ..."}},
      {"network-input-size", VariantType::Int, 1000, {"Size of the vector to be fed through the neural network"}},
      {"network-class-threshold", VariantType::Float, 0.5f, {"Threshold for classification network: Keep or reject maximum (default: 0.5)"}},
      {"network-device", VariantType::String, "cpu", {"Device on which to execute the NNs"}},
      {"network-dtype", VariantType::String, "FP32", {"Dtype for which the execution is done (FP32, FP16)"}},
      {"enable-network-optimizations", VariantType::Int, 1, {"Enable ONNX network optimizations"}},
      {"network-num-threads", VariantType::Int, 1, {"Set the number of CPU threads for network execution"}},
      {"network-threshold-sigmoid-trafo", VariantType::Int, 0, {"If 1, convert network-class-threshold to sigmoid^-1(threshold)"}},
      {"remove-individual-files", VariantType::Int, 0, {"Remove sector-individual files that are created during the task and only keep merged files"}},
      {"training-data-distance-cluster-path", VariantType::Float, 2.f, {"When creating the training data with momentum information, this defines the distance between an assigned (native) cluster and the track path until which a momentum vector is used. If track is too far away, no momentum vector infromation is written"}}}};
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
  if (cfgc.options().get<int>("write-native-file")) {

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
          def.keys = mergeInputs(def.keys, laneConfiguration.size(), amendInput);
          // the branch is disabled if set to 0
          def.nofBranches = tpcSectors.size();
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
                                   (cfgc.options().get<std::string>("output-path") + "/" + cfgc.options().get<std::string>("outfile-native")).c_str(),
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
