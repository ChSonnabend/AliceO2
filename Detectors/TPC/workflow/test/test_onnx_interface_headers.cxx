#include <iostream>
#include <vector>
#include <fstream>
#include <thread>

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
#include <chrono>

#include "Algorithm/RangeTokenizer.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/ConstMCTruthContainer.h"
#include "SimulationDataFormat/LabelContainer.h"
#include "SimulationDataFormat/IOMCTruthContainerView.h"
#include "SimulationDataFormat/MCTruthContainer.h"

#include "Headers/DataHeader.h"

#include "ML/OrtInterface.h"
#include "ML/3rdparty/GPUORTFloat16.h"

#include "Steer/MCKinematicsReader.h"

#include "DPLUtils/RootTreeReader.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"

#include "DataFormatsTPC/WorkflowHelper.h"
#include "DataFormatsTPC/ClusterNativeHelper.h"
#include "DataFormatsTPC/ClusterNative.h"
#include "DataFormatsTPC/ClusterGroupAttribute.h"
#include "DataFormatsTPC/Constants.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "DataFormatsGlobalTracking/TrackTuneParams.h"
#include "DataFormatsTPC/Defs.h"

#include "TPCWorkflow/ProcessingHelpers.h"
#include "TPCQC/Clusters.h"
#include "TPCBase/Painter.h"
#include "TPCBase/CalDet.h"
#include "TPCBase/Mapper.h"

#include "Framework/Logger.h"
#include "Framework/Task.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/CompletionPolicyHelpers.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/CallbacksPolicy.h"

#include "DetectorsRaw/HBFUtils.h"

#include <onnx/onnx_pb.h>

using namespace o2;
using namespace o2::ml;
using namespace o2::tpc;
using namespace o2::framework;

namespace o2
{
namespace tpc
{
class onnxInference : public Task
{
 public:
  onnxInference(std::unordered_map<std::string, std::string> optionsMap)
  {
    options_map = optionsMap;
    models = std::vector<OrtModel>(std::stoi(options_map["execution-threads"]));
    for (int thrd = 0; thrd < std::stoi(options_map["execution-threads"]); thrd++) {
      models[thrd].init(options_map);
    }
  };

  template <class I, class O>
  void runONNXGPUModel(std::vector<std::vector<I>>& input, int execution_threads)
  {
    std::vector<std::thread> threads(execution_threads);
    for (int thrd = 0; thrd < execution_threads; thrd++) {
      threads[thrd] = std::thread([&, thrd] {
        auto outputTensors = models[thrd].inference<I, O>(input[thrd]);
      });
    }
    for (auto& thread : threads) {
      thread.join();
    }
  };

  template <class I, class O>
  void runONNXGPUModel(std::vector<std::vector<std::vector<I>>>& input, int execution_threads)
  {
    std::vector<std::thread> threads(execution_threads);
    for (int thrd = 0; thrd < execution_threads; thrd++) {
      threads[thrd] = std::thread([&, thrd] {
        auto outputTensors = models[thrd].inference<I, O>(input[thrd]);
      });
    }
    for (auto& thread : threads) {
      thread.join();
    }
  };

  void add_concat_to_input(onnx::ModelProto& model, std::vector<ONNXAdaptorSpec> originals)
  {
    auto* graph = model.mutable_graph();

    // 1. Save original input name and remove it
    std::string old_input_name = graph->input(0).name();
    graph->mutable_input()->DeleteSubrange(0, 1);

    // 2. Create N new input tensors: input_0, ..., input_N-1
    std::vector<std::string> input_names;
    for (int i = 0; i < originals.size(); ++i) {
      std::string name = fmt::format("input_{}", originals[i].name);
      input_names.push_back(name);

      auto* input = graph->add_input();
      input->set_name(name);
      auto* tensor_type = input->mutable_type()->mutable_tensor_type();
      tensor_type->set_elem_type(onnx::TensorProto_DataType_FLOAT);
      auto* shape = tensor_type->mutable_shape();
      shape->add_dim()->set_dim_param("N");
      shape->add_dim()->set_dim_value(originals[i].numColumns);
    }

    // 3. Add Concat node
    auto* concat = graph->add_node();
    concat->set_op_type("Concat");
    concat->set_name("concat_inputs");
    for (const auto& name : input_names) {
      concat->add_input(name);
    }
    concat->add_output(old_input_name); // Output replaces the original input
    concat->add_attribute()->CopyFrom([] {
      onnx::AttributeProto attr;
      attr.set_name("merged_output");
      attr.set_type(onnx::AttributeProto_AttributeType_FLOAT);
      attr.set_i(1); // Concatenate on feature dimension
      return attr;
    }());
  }
  void print_shape(const onnx::TensorShapeProto& shape)
  {
    std::cout << "[";
    for (int i = 0; i < shape.dim_size(); ++i) {
      if (i > 0) {
        std::cout << ", ";
      }
      if (shape.dim(i).has_dim_value()) {
        std::cout << shape.dim(i).dim_value();
      } else if (shape.dim(i).has_dim_param()) {
        std::cout << shape.dim(i).dim_param();
      } else {
        std::cout << "?";
      }
    }
    std::cout << "]";
  }

  void testModelSurgery(){
    onnx::ModelProto modelProto;
    std::ifstream input(options_map["model-path"].c_str(), std::ios::in | std::ios::binary);

    if (!input) {
      throw std::runtime_error("Failed to open model file: " + options_map["model-path"]);
    }
    if (!modelProto.ParseFromIstream(&input)) {
      throw std::runtime_error("Failed to parse ONNX model from stream.");
    }

    model.set_ir_version(onnx::IR_VERSION);
    model.set_producer_name("example_linear");

    onnx::GraphProto* graph = model.mutable_graph();
    graph->set_name("LinearGraph");

    // onnx::ValueInfoProto* input = graph->add_input();
    // input->set_name("input");
    // onnx::TypeProto* input_type = input->mutable_type();
    // auto* input_tensor_type = input_type->mutable_tensor_type();
    // input_tensor_type->set_elem_type(onnx::TensorProto_DataType_FLOAT);
    // auto* input_shape = input_tensor_type->mutable_shape();
    // input_shape->add_dim()->set_dim_param("N");
    // input_shape->add_dim()->set_dim_value(7);

    // const int input_dim = 7;
    // const int output_dim = 4;

    // // Weights (W): shape [4, 7]
    // onnx::TensorProto* weights = graph->add_initializer();
    // weights->set_name("W");
    // weights->set_data_type(onnx::TensorProto_DataType_FLOAT);
    // weights->add_dims(output_dim);
    // weights->add_dims(input_dim);
    // for (int i = 0; i < output_dim * input_dim; ++i) {
    //   weights->add_float_data(ONNXHelpers::rand_float());
    // }

    // // Bias (B): shape [4]
    // onnx::TensorProto* bias = graph->add_initializer();
    // bias->set_name("B");
    // bias->set_data_type(onnx::TensorProto_DataType_FLOAT);
    // bias->add_dims(output_dim);
    // for (int i = 0; i < output_dim; ++i) {
    //   bias->add_float_data(ONNXHelpers::rand_float());
    // }

    // // ==== MatMul Node ====
    // onnx::NodeProto* matmul = graph->add_node();
    // matmul->set_op_type("MatMul");
    // matmul->add_input("input");
    // matmul->add_input("W");
    // matmul->add_output("matmul_out");

    // // ==== Add Node ====
    // onnx::NodeProto* add = graph->add_node();
    // add->set_op_type("Add");
    // add->add_input("matmul_out");
    // add->add_input("B");
    // add->add_output("output");

    // // ==== Output Tensor ====
    // onnx::ValueInfoProto* output = graph->add_output();
    // output->set_name("output");
    // onnx::TypeProto* output_type = output->mutable_type();
    // auto* output_tensor_type = output_type->mutable_tensor_type();
    // output_tensor_type->set_elem_type(onnx::TensorProto_DataType_FLOAT);
    // auto* output_shape = output_tensor_type->mutable_shape();
    // output_shape->add_dim()->set_dim_param("N");
    // output_shape->add_dim()->set_dim_value(output_dim);

    // std::stringstream out;
    // if (!model.SerializeToOstream(&out)) {
    //   std::cerr << "Failed to write ONNX model.\n";
    //   return 1;
    // }
    // onnx::ModelProto modelIn;
    // if (!modelIn.ParseFromString(out.str())) {
    //   std::cerr << "Failed to parse ONNX model.\n";
    //   return 1;
    // }

    std::vector<ONNXAdaptorSpec> specs = {
      {"1", 1},
      {"2", 1},
      {"3", 1},
      {"4", 1},
      {"5", 1},
      {"6", 1},
      {"7", 1},
    };
    add_concat_to_input(model, specs);

    std::cout << "=== ONNX Model Summary ===\n";
    std::cout << "Producer: " << model.producer_name() << "\n";
    std::cout << "IR version: " << model.ir_version() << "\n";
    std::cout << "Graph name: " << model.graph().name() << "\n";

    // Inputs
    {
      onnx::GraphProto const& graph = model.graph();
      std::cout << "\nInputs:\n";
      for (const auto& input : graph.input()) {
        std::cout << "  " << input.name() << " : ";
        if (input.has_type() && input.type().has_tensor_type()) {
          const auto& shape = input.type().tensor_type().shape();
          print_shape(shape);
        }
        std::cout << "\n";
      }

      // Outputs
      std::cout << "\nOutputs:\n";
      for (const auto& output : graph.output()) {
        std::cout << "  " << output.name() << " : ";
        if (output.has_type() && output.type().has_tensor_type()) {
          const auto& shape = output.type().tensor_type().shape();
          print_shape(shape);
        }
        std::cout << "\n";
      }

      // Nodes
      std::cout << "\nNodes:\n";
      for (const auto& node : graph.node()) {
        std::cout << "  [" << node.op_type() << "] ";
        for (const auto& input : node.input()) {
          std::cout << input << " ";
        }
        std::cout << "-> ";
        for (const auto& output : node.output()) {
          std::cout << output << " ";
        }
        std::cout << "\n";
      }
    }

    // Export the modified model to ./model.onnx
    std::ofstream outModel("./model.onnx", std::ios::out | std::ios::binary);
    if (!outModel) {
      std::cerr << "Failed to open output file for writing ONNX model.\n";
      return;
    }
    if (!model.SerializeToOstream(&outModel)) {
      std::cerr << "Failed to serialize ONNX model to file.\n";
      return;
    }
    std::cout << "Modified ONNX model exported to ./model.onnx\n";
  }

  void init(InitContext& ic) final {};

  template <typename I, typename O>
  void run_models() {

    double time = 0;

    int test_size_tensor = std::stoi(options_map["size-tensor"]);
    int epochs_measure = std::stoi(options_map["measure-cycle"]);
    int execution_threads = std::stoi(options_map["execution-threads"]);
    int test_num_tensors = std::stoi(options_map["num-tensors"]);
    int test_size_iter = std::stoi(options_map["num-iter"]);

    LOG(info) << "Preparing input data";
    // Prepare input data
    std::vector<int64_t> inputShape{test_size_tensor, models[0].getNumInputNodes()[0][1]};

    LOG(info) << "Creating ONNX tensor";
    std::vector<std::vector<I>> input_tensor(execution_threads);
    std::vector<I> input_data(models[0].getNumInputNodes()[0][1] * test_size_tensor, I(1.0f)); // Example input
    for (int i = 0; i < execution_threads; i++) {
      input_tensor[i] = input_data;
      // input_tensor[i].resize(test_num_tensors);
      // for(int j = 0; j < test_num_tensors; j++){
      // 	input_tensor[i][j] = input_data;
      // }
    }

    LOG(info) << "Starting inference";
    auto start_network_eval = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < test_size_iter; i++) {
      runONNXGPUModel<I, O>(input_tensor, execution_threads);
      if ((i % epochs_measure == 0) && (i != 0)) {
        auto end_network_eval = std::chrono::high_resolution_clock::now();
        time = std::chrono::duration<double, std::ratio<1, (unsigned long)1e9>>(end_network_eval - start_network_eval).count() / 1e9;
        LOG(info) << "Total time: " << time << "s. Timing: " << uint64_t((double)test_size_tensor * epochs_measure * execution_threads / time) << " elements / s";
        time = 0;
        start_network_eval = std::chrono::high_resolution_clock::now();
      }
    }
  }
  void run(ProcessingContext& pc) final
  {
    if(options_map["mode"] == "surgery"){
      testModelSurgery();
    } else if (options_map["mode"] == "run") {
      if (options_map["dtype"] == "FP16") {
        run_models<OrtDataType::Float16_t, OrtDataType::Float16_t>();
      } else if (options_map["dtype"] == "INT8") {
        run_models<float, int8_t>();
      } else {
        run_models<float, float>();
      }
    }

    pc.services().get<ControlService>().endOfStream();
    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
  };

 private:
  std::vector<OrtModel> models;
  std::unordered_map<std::string, std::string> options_map;
};
} // namespace tpc
} // namespace o2

void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  std::vector<ConfigParamSpec> options{
    {"mode", VariantType::String, "run", {"Mode of operation: run or surgery"}},
    {"path", VariantType::String, "./model.onnx", {"Path to ONNX model"}},
    {"device", VariantType::String, "CPU", {"Device on which the ONNX model is run"}},
    {"device-id", VariantType::Int, 0, {"Device ID on which the ONNX model is run"}},
    {"dtype", VariantType::String, "-", {"Dtype in which the ONNX model is run (FP16 or FP32)"}},
    {"size-tensor", VariantType::Int, 100, {"Size of the input tensor"}},
    {"execution-threads", VariantType::Int, 1, {"If > 1 will run session->Run() with multiple threads as execution providers"}},
    {"intra-op-num-threads", VariantType::Int, 0, {"Number of threads per session for CPU execution provider"}},
    {"num-tensors", VariantType::Int, 1, {"Number of tensors on which execution is being performed"}},
    {"num-iter", VariantType::Int, 100, {"Number of iterations"}},
    {"measure-cycle", VariantType::Int, 10, {"Epochs in which to measure"}},
    {"enable-profiling", VariantType::Int, 0, {"Enable profiling"}},
    {"profiling-output-path", VariantType::String, "/scratch/csonnabe/O2_new", {"Path to save profiling output"}},
    {"logging-level", VariantType::Int, 1, {"Logging level"}},
    {"enable-optimizations", VariantType::Int, 0, {"Enable optimizations"}},
    {"allocate-device-memory", VariantType::Int, 0, {"Allocate the memory on device"}}};
  std::swap(workflowOptions, options);
}

// ---------------------------------
#include "Framework/runDataProcessing.h"

DataProcessorSpec testProcess(ConfigContext const& cfgc, std::vector<InputSpec>& inputs, std::vector<OutputSpec>& outputs)
{

  // A copy of the global workflow options from customize() to pass to the task
  std::unordered_map<std::string, std::string> options_map{
    {"model-path", cfgc.options().get<std::string>("path")},
    {"device", cfgc.options().get<std::string>("device")},
    {"device-id", std::to_string(cfgc.options().get<int>("device-id"))},
    {"dtype", cfgc.options().get<std::string>("dtype")},
    {"size-tensor", std::to_string(cfgc.options().get<int>("size-tensor"))},
    {"intra-op-num-threads", std::to_string(cfgc.options().get<int>("intra-op-num-threads"))},
    {"execution-threads", std::to_string(cfgc.options().get<int>("execution-threads"))},
    {"num-tensors", std::to_string(cfgc.options().get<int>("num-tensors"))},
    {"num-iter", std::to_string(cfgc.options().get<int>("num-iter"))},
    {"measure-cycle", std::to_string(cfgc.options().get<int>("measure-cycle"))},
    {"enable-profiling", std::to_string(cfgc.options().get<int>("enable-profiling"))},
    {"profiling-output-path", cfgc.options().get<std::string>("profiling-output-path")},
    {"logging-level", std::to_string(cfgc.options().get<int>("logging-level"))},
    {"enable-optimizations", std::to_string(cfgc.options().get<int>("enable-optimizations"))},
    {"allocate-device-memory", std::to_string(cfgc.options().get<int>("allocate-device-memory"))}};

  return DataProcessorSpec{
    "test-onnx-interface",
    inputs,
    outputs,
    adaptFromTask<onnxInference>(options_map),
    Options{
      {"somethingElse", VariantType::String, "-", {"Something else"}}}};
}

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{

  WorkflowSpec specs;

  static std::vector<InputSpec> inputs;
  static std::vector<OutputSpec> outputs;

  specs.push_back(testProcess(cfgc, inputs, outputs));

  return specs;
}
