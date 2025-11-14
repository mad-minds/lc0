/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2024 The LCZero Authors

  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "neural/keras/converter.h"

#include <Python.h>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

#include "neural/loader.h"
#include "neural/network.h"
#include "neural/network_legacy.h"
#include "neural/onnx/adapters.h"
#include "neural/tables/activation_function.h"
#include "neural/tables/attention_policy_map.h"
#include "neural/tables/policy_map.h"
#include "proto/net.pb.h"

#include "utils/bf16_utils.h"
#include "utils/exception.h"
#include "utils/fp16_utils.h"
#include "utils/transpose.h"

namespace lczero {
namespace {

// Helper class to manage Python interpreter and execute Python code
class PythonInterpreter {
 public:
  PythonInterpreter(bool track_code = false, const std::string& weights_dir = "") 
      : track_code_(track_code), weights_dir_(weights_dir) {
    Py_Initialize();
    if (!Py_IsInitialized()) {
      throw Exception("Failed to initialize Python interpreter. "
                      "Make sure Python and TensorFlow are installed.");
    }

    // Add site-packages paths after initialization
    // This ensures Python can find TensorFlow and other installed packages
    AddSitePackagesToPath();
    
    // Add the current working directory to Python path to find lc0_keras_layers
    AddCurrentDirToPath();

    // Build import statements for the script
    script_ << "import sys\n";
    script_ << "import os\n";
    script_ << "\n";
    script_ << "import tensorflow as tf\n";
    script_ << "from tensorflow import keras\n";
    script_ << "from tensorflow.keras import layers\n";
    script_ << "from keras import ops\n";
    script_ << "import numpy as np\n";
    script_ << "\n";
    script_ << "# Import custom LCZero Keras layers\n";
    script_ << "# Install: uv add git+https://github.com/LeelaChessZero/lc0#subdirectory=src/python\n";
    script_ << "import lc0keras\n";
    script_ << "from lc0keras import (\n";
    script_ << "    MatMul, Gather, DynamicTile, StaticRepeat, SliceLayer,\n";
    script_ << "    FlattenBatchSpatial, UnflattenBatchSpatial, DynamicReshape\n";
    script_ << ")\n";
    script_ << "\n";
    
    // Also add imports to code_log_ if tracking (for --python-output file)
    if (track_code_) {
      code_log_ += "#!/usr/bin/env python3\n";
      code_log_ += "\"\"\"LCZero Keras model generation script.\n";
      code_log_ += "\n";
      code_log_ += "This script generates a Keras model from LCZero network weights.\n";
      code_log_ += "\n";
      code_log_ += "Requirements:\n";
      code_log_ += "    pip install lc0-keras\n";
      code_log_ += "    # or\n";
      code_log_ += "    uv add git+https://github.com/LeelaChessZero/lc0#subdirectory=src/python\n";
      code_log_ += "\"\"\"\n";
      code_log_ += "import tensorflow as tf\n";
      code_log_ += "from tensorflow import keras\n";
      code_log_ += "from tensorflow.keras import layers\n";
      code_log_ += "from keras import ops\n";
      code_log_ += "import numpy as np\n";
      code_log_ += "\n";
      code_log_ += "# Import custom LCZero Keras layers\n";
      code_log_ += "import lc0keras\n";
      code_log_ += "from lc0keras import (\n";
      code_log_ += "    MatMul, Gather, DynamicTile, StaticRepeat, SliceLayer,\n";
      code_log_ += "    FlattenBatchSpatial, UnflattenBatchSpatial, DynamicReshape\n";
      code_log_ += ")\n";
      code_log_ += "\n";
    }
  }

  ~PythonInterpreter() {
    if (Py_IsInitialized()) {
      Py_Finalize();
    }
  }

  // Append Python code to the script (deferred execution)
  void AppendPython(const std::string& code) {
    script_ << code << "\n";
    if (track_code_) {
      code_log_ += code + "\n";
    }
  }

  // Helper to append a formatted line (reduces boilerplate with ostringstream)
  template<typename... Args>
  void AppendLine(const std::string& format_str) {
    AppendPython(format_str);
  }

  // Legacy method for immediate execution (only used for initialization now)
  void ExecutePython(const std::string& code) {
    AppendPython(code);
  }

  // Execute the entire accumulated script
  void ExecuteAll() {
    std::string full_script = script_.str();
    int result = PyRun_SimpleString(full_script.c_str());
    if (result != 0) {
      PyObject* ptype, *pvalue, *ptraceback;
      PyErr_Fetch(&ptype, &pvalue, &ptraceback);
      std::string error_msg = "Python execution error: ";
      if (pvalue) {
        PyObject* str = PyObject_Str(pvalue);
        if (str) {
          const char* msg = PyUnicode_AsUTF8(str);
          if (msg) error_msg += msg;
          Py_DECREF(str);
        }
      }
      Py_XDECREF(ptype);
      Py_XDECREF(pvalue);
      Py_XDECREF(ptraceback);
      // Find the line that caused the error (rough approximation)
      std::string code_preview;
      if (full_script.length() > 200) {
        code_preview = full_script.substr(0, 200) + "... (truncated)";
      } else {
        code_preview = full_script;
      }
      throw Exception(error_msg + "\nPython code: " + code_preview);
    }
  }

  std::string GetCodeLog() const { return code_log_.empty() ? script_.str() : code_log_; }
  void ClearCodeLog() { code_log_.clear(); script_.str(""); script_.clear(); }

  // Save numpy array to .npy file directly in C++ (much faster than Python string conversion)
  void SaveNumpyArray(const std::string& file_path,
                      const std::vector<float>& data,
                      const std::vector<int>& shape) {
    // Create directory if needed
    std::filesystem::path path(file_path);
    std::filesystem::create_directories(path.parent_path());
    
    // Open file for binary writing
    std::ofstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
      throw Exception("Failed to open file for writing: " + file_path);
    }
    
    // Write .npy header (version 1.0 format)
    // Magic string
    file.write("\x93NUMPY", 6);
    
    // Version
    file.put(1);  // major
    file.put(0);  // minor
    
    // Build header dict
    std::ostringstream header;
    header << "{'descr': '<f4', 'fortran_order': False, 'shape': (";
    for (size_t i = 0; i < shape.size(); ++i) {
      if (i > 0) header << ", ";
      header << shape[i];
    }
    if (shape.size() == 1) header << ",";  // Trailing comma for 1D arrays
    header << "), }";
    
    // Pad header to multiple of 64 bytes (including magic + version + header_len)
    std::string header_str = header.str();
    size_t total_header_size = 6 + 2 + 2 + header_str.size() + 1;  // +1 for newline
    size_t padding = (64 - (total_header_size % 64)) % 64;
    for (size_t i = 0; i < padding; ++i) {
      header_str += ' ';
    }
    header_str += '\n';
    
    // Write header length (little-endian uint16)
    uint16_t header_len = static_cast<uint16_t>(header_str.size());
    file.write(reinterpret_cast<const char*>(&header_len), 2);
    
    // Write header
    file.write(header_str.c_str(), header_str.size());
    
    // Write data (float32, little-endian)
    file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
    
    file.close();
    if (!file.good()) {
      throw Exception("Failed to write numpy array to file: " + file_path);
    }
  }

  std::string CreateNumpyArray(const std::string& var_name,
                               const std::vector<float>& data,
                               const std::vector<int>& shape) {
    // Validate shape before proceeding
    size_t expected_size = 1;
    for (int dim : shape) {
      if (dim < 0) {
        throw Exception("Invalid shape dimension < 0 for " + var_name);
      }
      expected_size *= dim;
    }
    if (expected_size != data.size()) {
      throw Exception("Shape size mismatch for " + var_name + ": expected " + 
                      std::to_string(expected_size) + ", got " + 
                      std::to_string(data.size()));
    }
    if (!weights_dir_.empty()) {
      // Save to .npy file using C++ directly
      std::filesystem::path weights_path(weights_dir_);
      std::filesystem::create_directories(weights_path);
      
      std::string filename = var_name + ".npy";
      std::filesystem::path file_path = weights_path / filename;
      std::filesystem::path abs_file_path = std::filesystem::absolute(file_path);
      std::string file_path_str = abs_file_path.string();
      
      // Save the numpy array file
      SaveNumpyArray(file_path_str, data, shape);
      
      // In the script, just load the file
      std::ostringstream code;
      // Escape backslashes for Python string
      std::string escaped_path = file_path_str;
      size_t pos = 0;
      while ((pos = escaped_path.find('\\', pos)) != std::string::npos) {
        escaped_path.replace(pos, 1, "\\\\");
        pos += 2;
      }
      code << var_name << " = np.load(r'" << escaped_path << "')";
      
      AppendPython(code.str());
      return var_name;
    } else {
      // Original behavior: embed in Python code
      std::ostringstream code;
      code << var_name << " = np.array([";

      for (size_t i = 0; i < data.size(); ++i) {
        if (i > 0) code << ", ";
        code << std::setprecision(9) << std::fixed << data[i];
      }

      code << "], dtype=np.float32).reshape((";
      for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) code << ", ";
        code << shape[i];
      }
      code << "))";

      AppendPython(code.str());
      return var_name;
    }
  }

 private:
  bool track_code_;
  std::string code_log_;
  std::ostringstream script_;  // Accumulates the entire Python script for deferred execution
  std::string weights_dir_;  // Directory to save weight .npy files
  
  void AddSitePackagesToPath() {
    // After initialization, add additional paths to sys.path programmatically
    // This ensures Python can find TensorFlow and other packages installed in the environment
    std::ostringstream code;
    
    // Use sysconfig to get the correct site-packages path for this Python installation
    // This works for both regular Python installations and conda/mamba environments
    code << "import sys\n";
    code << "import sysconfig\n";
    code << "import os\n";
    code << "\n";
    
    // Get the purelib path (where packages are installed)
    code << "# Add site-packages from sysconfig (handles conda/venv/system Python)\n";
    code << "try:\n";
    code << "    purelib = sysconfig.get_path('purelib')\n";
    code << "    if purelib and os.path.exists(purelib) and purelib not in sys.path:\n";
    code << "        sys.path.insert(0, purelib)\n";
    code << "except:\n";
    code << "    pass\n";
    code << "\n";
    
    // Also add based on sys.prefix (works for conda environments)
    code << "# Add site-packages based on sys.prefix (for conda/mamba)\n";
    code << "try:\n";
    code << "    prefix = sys.prefix\n";
    code << "    version = str(sys.version_info.major) + '.' + str(sys.version_info.minor)\n";
    code << "    site_pkg = os.path.join(prefix, 'lib', 'python' + version, 'site-packages')\n";
    code << "    if os.path.exists(site_pkg) and site_pkg not in sys.path:\n";
    code << "        sys.path.insert(0, site_pkg)\n";
    code << "except:\n";
    code << "    pass\n";
    code << "\n";
    
    // Add base environment site-packages (important for conda + venv setups)
    code << "# Add base environment site-packages (for conda base when using venv)\n";
    code << "try:\n";
    code << "    if hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix:\n";
    code << "        base_prefix = sys.base_prefix\n";
    code << "        version = str(sys.version_info.major) + '.' + str(sys.version_info.minor)\n";
    code << "        base_site_pkg = os.path.join(base_prefix, 'lib', 'python' + version, 'site-packages')\n";
    code << "        if os.path.exists(base_site_pkg) and base_site_pkg not in sys.path:\n";
    code << "            sys.path.insert(0, base_site_pkg)\n";
    code << "except:\n";
    code << "    pass\n";
    code << "\n";
    
    // Check VIRTUAL_ENV if set
    code << "# Add virtual environment site-packages if VIRTUAL_ENV is set\n";
    code << "venv = os.environ.get('VIRTUAL_ENV')\n";
    code << "if venv:\n";
    code << "    try:\n";
    code << "        venv_site = sysconfig.get_path('purelib', vars={'base': venv})\n";
    code << "        if venv_site and os.path.exists(venv_site) and venv_site not in sys.path:\n";
    code << "            sys.path.insert(0, venv_site)\n";
    code << "    except:\n";
    code << "        # Fallback: construct path manually\n";
    code << "        version = str(sys.version_info.major) + '.' + str(sys.version_info.minor)\n";
    code << "        venv_site = os.path.join(venv, 'lib', 'python' + version, 'site-packages')\n";
    code << "        if os.path.exists(venv_site) and venv_site not in sys.path:\n";
    code << "            sys.path.insert(0, venv_site)\n";
    
    // Execute immediately (not deferred) - needed for SaveNumpyArray to work
    PyRun_SimpleString(code.str().c_str());
  }
  
  void AddCurrentDirToPath() {
    // Add src/python to Python path for local development
    // This allows importing lc0keras when running from the lc0 repo
    std::ostringstream code;
    code << "import sys\n";
    code << "import os\n";
    code << "# Add src/python for local development (lc0keras package)\n";
    code << "cwd = os.getcwd()\n";
    code << "src_python = os.path.join(cwd, 'src', 'python')\n";
    code << "if os.path.exists(src_python) and src_python not in sys.path:\n";
    code << "    sys.path.insert(0, src_python)\n";
    code << "\n";
    
    // Execute immediately
    PyRun_SimpleString(code.str().c_str());
  }
};

class KerasConverter {
 public:
  KerasConverter(const pblczero::Net& net,
                 const WeightsToKerasConverterOptions& options)
      : src_(net),
        options_(options),
        py_(!options.python_output_file.empty(), options.weights_dir),
        default_activation_(
            net.format().network_format().default_activation() ==
                    pblczero::NetworkFormat::DEFAULT_ACTIVATION_MISH
                ? ACTIVATION_MISH
                : ACTIVATION_RELU),
        default_eps_(net.format().network_format().input_embedding() ==
                             pblczero::NetworkFormat::INPUT_EMBEDDING_PE_DENSE
                         ? 1e-3f
                         : 1e-6f) {
    // Initialize input layer in Python
    std::ostringstream code;
    if (options_.batch_size > 0) {
      // Input shape matches ONNX format: (channels, height, width) = (112, 8, 8)
      code << "input_planes = layers.Input(shape=(" << kInputPlanes << ", 8, 8"
           << "), name='" << SanitizeLayerName(options_.input_name) << "', batch_size=" << options_.batch_size << ")";
    } else {
      // For dynamic batch size, omit batch_size parameter (defaults to None)
      code << "input_planes = layers.Input(shape=(" << kInputPlanes << ", 8, 8"
           << "), name='" << SanitizeLayerName(options_.input_name) << "')";
    }
    py_.AppendPython(code.str());
  }

  void ConvertToKeras(const std::string& output_path);

 private:
  int NumFilters() const {
    return LayerAdapter(src_.weights().input().weights()).size() /
           kInputPlanes / 9;
  }

  size_t NumResBlocks() const { return src_.weights().residual_size(); }
  size_t NumEncBlocks() const { return src_.weights().encoder().size(); }

  std::string MakeConvLayer(const std::string& input_var,
                           const MultiHeadWeights::ConvBlock& weights,
                           int input_channels, int output_channels,
                           const std::string& layer_name,
                           int kernel_size = 3,
                           bool activation = true);

  std::string MakeResidualBlock(const std::string& input_var,
                               const MultiHeadWeights::Residual& res,
                               const std::string& block_name);

  std::string MakeActivation(const std::string& input_var,
                            const std::string& name,
                            ActivationFunction activation);

  std::string MakeSqueezeAndExcite(const std::string& input_var,
                                  const MultiHeadWeights::SEunit& se_unit,
                                  const std::string& name);

  std::string MakeLayerNorm(const std::string& input_var,
                           const std::string& name,
                           const std::vector<float>& gammas,
                           const std::vector<float>& betas,
                           float eps = 1e-6f);

  std::string MakeFFN(const std::string& ffn_in,
                     const MultiHeadWeights::FFN& ffn,
                     int embedding_size,
                     const std::string& name,
                     ActivationFunction activation,
                     float alpha = 1.0f);

  std::string MakeEncoderLayer(const std::string& encoder_in,
                              const MultiHeadWeights::EncoderLayer& layer,
                              int embedding_size, int heads,
                              const std::string& name,
                              ActivationFunction activation,
                              const MultiHeadWeights& weights,
                              float alpha = 1.0f);

  std::string MakeSmolgen(const MultiHeadWeights::EncoderLayer& layer,
                         int embedding_size, int heads,
                         const std::string& encoder_in,
                         const std::string& name,
                         const MultiHeadWeights& weights);

  std::string AttentionBodyMapEmbedding(const std::string& input);
  std::string AttentionBodyDenseEmbedding(const std::string& input,
                                         const MultiHeadWeights& weights,
                                         int embedding_dense_size);
  std::string MakeAttentionBody(const std::string& input,
                               const MultiHeadWeights& weights);

  std::string MakeAttentionPolicy(const std::string& input,
                                 const MultiHeadWeights& weights,
                                 const MultiHeadWeights::PolicyHead& head);

  std::string MakePolicyHead(const MultiHeadWeights& weights,
                            const std::string& input_var);
  std::string MakeValueHead(const MultiHeadWeights& weights,
                           const std::string& input_var);
  std::string MakeMovesLeftHead(const MultiHeadWeights& weights,
                               const std::string& input_var);

  std::string WeightsToNumpyArray(const std::string& var_name,
                                 const std::vector<float>& weights,
                                 const std::vector<int>& dims,
                                 const std::vector<int>& transpose_order = {});

  // Helper to create a Dense layer with weights (reduces code duplication)
  std::string MakeDenseLayer(const std::string& name,
                            const std::string& input_var,
                            const std::vector<float>& weights,
                            const std::vector<float>& biases,
                            int input_size,
                            int output_size,
                            const std::vector<int>& weight_transpose = {1, 0});

  // Sanitize layer name for Keras: strip leading "/" and replace other "/" with "_"
  std::string SanitizeLayerName(const std::string& name) const;

  const pblczero::Net& src_;
  const WeightsToKerasConverterOptions& options_;
  PythonInterpreter py_;
  const ActivationFunction default_activation_;
  const float default_eps_;
  std::string current_flow_var_;
  std::vector<std::string> output_vars_;
  bool smolgen_w_created_ = false;
};

std::string KerasConverter::WeightsToNumpyArray(
    const std::string& var_name,
    const std::vector<float>& weights,
    const std::vector<int>& dims,
    const std::vector<int>& transpose_order) {
  std::vector<float> transposed = weights;
  if (!transpose_order.empty()) {
    std::vector<float> temp(weights.size());
    TransposeTensor<float>(dims, transpose_order, weights, temp.data());
    transposed = temp;
  }

  // For Keras format conversion if needed (f16/bf16 handled at model level)
  if (options_.data_type != WeightsToKerasConverterOptions::DataType::kFloat32) {
    // Note: Keras model.save() will handle dtype conversion automatically
    // when we set the model dtype, so we keep weights as float32 here
  }

  return py_.CreateNumpyArray(var_name, transposed, dims);
}

std::string KerasConverter::SanitizeLayerName(const std::string& name) const {
  if (name.empty()) return name;
  
  std::string sanitized = name;
  // Strip leading "/"
  if (!sanitized.empty() && sanitized[0] == '/') {
    sanitized = sanitized.substr(1);
  }
  // Replace all other "/" with "_"
  for (size_t i = 0; i < sanitized.size(); ++i) {
    if (sanitized[i] == '/') {
      sanitized[i] = '_';
    }
  }
  return sanitized;
}

std::string KerasConverter::MakeDenseLayer(
    const std::string& name,
    const std::string& input_var,
    const std::vector<float>& weights,
    const std::vector<float>& biases,
    int input_size,
    int output_size,
    const std::vector<int>& weight_transpose) {
  
  // Create weight and bias arrays
  std::string w_var = name + "_w";
  std::string b_var = name + "_b";
  
  WeightsToNumpyArray(w_var, weights, {input_size, output_size}, weight_transpose);
  WeightsToNumpyArray(b_var, biases, {output_size});
  
  // Generate layer code
  std::ostringstream code;
  std::string sanitized_name = SanitizeLayerName(name);
  
  code << name << "_layer = layers.Dense(" << output_size << ", name='" 
       << sanitized_name << "', use_bias=True)";
  py_.AppendPython(code.str());
  
  code.str("");
  code << name << "_layer.build([None, " << input_size << "])";
  py_.AppendPython(code.str());
  
  code.str("");
  code << name << "_layer.set_weights([" << w_var << ", " << b_var << "])";
  py_.AppendPython(code.str());
  
  code.str("");
  code << name << " = " << name << "_layer(" << input_var << ")";
  py_.AppendPython(code.str());
  
  return name;
}

std::string KerasConverter::MakeActivation(const std::string& input_var,
                                          const std::string& name,
                                          ActivationFunction activation) {
  std::ostringstream code;
  
  if (activation == ACTIVATION_DEFAULT) {
    activation = default_activation_;
  }

  switch (activation) {
    case ACTIVATION_RELU:
      code << name << " = layers.ReLU(name='" << name << "')(" << input_var << ")";
      break;
    case ACTIVATION_MISH:
      code << name << " = layers.Activation('mish', name='" << name << "')(" 
           << input_var << ")";
      break;
    case ACTIVATION_SWISH:
      code << name << " = layers.Activation('swish', name='" << name << "')(" 
           << input_var << ")";
      break;
    case ACTIVATION_SELU:
      code << name << " = layers.Activation('selu', name='" << name << "')(" 
           << input_var << ")";
      break;
    case ACTIVATION_TANH:
      code << name << " = layers.Activation('tanh', name='" << name << "')(" 
           << input_var << ")";
      break;
    case ACTIVATION_SIGMOID:
      code << name << " = layers.Activation('sigmoid', name='" << name << "')(" 
           << input_var << ")";
      break;
    case ACTIVATION_RELU_2: {
      // Squared ReLU: ReLU(x) * ReLU(x)
      std::string relu_out = name + "_relu";
      code << relu_out << " = layers.ReLU(name='" << relu_out << "')(" << input_var << ")";
      py_.AppendPython(code.str());
      code.str("");
      code << name << " = layers.Multiply(name='" << name << "')([" << relu_out 
           << ", " << relu_out << "])";
      break;
    }
    case ACTIVATION_NONE:
      return input_var;
    default:
      code << name << " = layers.ReLU(name='" << name << "')(" << input_var << ")";
  }

  py_.AppendPython(code.str());
  return name;
}

std::string KerasConverter::MakeConvLayer(
    const std::string& input_var,
    const MultiHeadWeights::ConvBlock& weights,
    int input_channels, int output_channels,
    const std::string& layer_name,
    int kernel_size,
    bool activation) {
  
  // Create kernel weights
  // Keras Conv2D ALWAYS expects weights in (kernel_h, kernel_w, in_channels, out_channels) format
  // regardless of data_format. ONNX format is (out_channels, in_channels, kernel_h, kernel_w)
  // So we need to transpose: {2, 3, 1, 0}
  std::string kernel_var = layer_name + "_kernel";
  std::vector<int> kernel_dims = {output_channels, input_channels, kernel_size, kernel_size};
  WeightsToNumpyArray(kernel_var, weights.weights, kernel_dims, {2, 3, 1, 0});
  
  // Create bias array
  std::string bias_var = layer_name + "_bias";
  WeightsToNumpyArray(bias_var, weights.biases, {output_channels});
  
  // Create Conv2D layer with channels_first format (NCHW)
  std::ostringstream code;
  code << layer_name << "_layer = layers.Conv2D("
       << "filters=" << output_channels << ", "
       << "kernel_size=" << kernel_size << ", "
       << "padding='same', "
       << "data_format='channels_first', "
       << "use_bias=True, "
       << "name='" << layer_name << "'"
       << ")";
  py_.AppendPython(code.str());
  
  // Apply layer to input
  code.str("");
  code << layer_name << " = " << layer_name << "_layer(" << input_var << ")";
  py_.AppendPython(code.str());
  
  // Set weights
  code.str("");
  code << layer_name << "_layer.set_weights([" << kernel_var << ", " << bias_var << "])";
  py_.AppendPython(code.str());
  
  // Apply activation
  if (activation) {
    return MakeActivation(layer_name, layer_name + "_act", default_activation_);
  }
  
  return layer_name;
}

std::string KerasConverter::MakeSqueezeAndExcite(
    const std::string& input_var,
    const MultiHeadWeights::SEunit& se_unit,
    const std::string& name) {
  const int se_filters = se_unit.b1.size();
  
  std::ostringstream code;
  
  // Global average pooling (with channels_first format)
  code << name << "_pool = layers.GlobalAveragePooling2D(data_format='channels_first', name='" 
       << name << "_pool')(" << input_var << ")";
  py_.AppendPython(code.str());
  
  // First dense layer
  std::string w1_var = name + "_w1";
  WeightsToNumpyArray(w1_var, se_unit.w1, {NumFilters(), se_filters}, {1, 0});
  
  std::string b1_var = name + "_b1";
  WeightsToNumpyArray(b1_var, se_unit.b1, {se_filters});
  
  code.str("");
  code << name << "_dense1_layer = layers.Dense(" << se_filters 
       << ", name='" << name << "_dense1', use_bias=True)";
  py_.AppendPython(code.str());
  
  code.str("");
  code << name << "_dense1_layer.build([None, " << NumFilters() << "])";
  py_.AppendPython(code.str());
  
  code.str("");
  code << name << "_dense1_layer.set_weights([" << w1_var << ", " << b1_var << "])";
  py_.AppendPython(code.str());
  
  code.str("");
  code << name << "_dense1 = " << name << "_dense1_layer(" << name << "_pool)";
  py_.AppendPython(code.str());
  
  std::string dense1_act = MakeActivation(name + "_dense1", name + "_dense1_act", 
                                         default_activation_);
  
  // Second dense layer
  std::string w2_var = name + "_w2";
  WeightsToNumpyArray(w2_var, se_unit.w2, {se_filters, 2 * NumFilters()}, {1, 0});
  
  std::string b2_var = name + "_b2";
  WeightsToNumpyArray(b2_var, se_unit.b2, {2 * NumFilters()});
  
  code.str("");
  code << name << "_dense2_layer = layers.Dense(" << (2 * NumFilters())
       << ", name='" << name << "_dense2', use_bias=True)";
  py_.AppendPython(code.str());
  
  code.str("");
  code << name << "_dense2_layer.build([None, " << se_filters << "])";
  py_.AppendPython(code.str());
  
  code.str("");
  code << name << "_dense2_layer.set_weights([" << w2_var << ", " << b2_var << "])";
  py_.AppendPython(code.str());
  
  code.str("");
  code << name << "_dense2 = " << name << "_dense2_layer(" << dense1_act << ")";
  py_.AppendPython(code.str());
  
  // Reshape to match spatial dimensions (channels_first: (batch, channels, 1, 1))
  code.str("");
  code << name << "_reshape = layers.Reshape((" << (2 * NumFilters())
       << ", 1, 1), name='" << name << "_reshape')(" << name << "_dense2)";
  py_.AppendPython(code.str());
  
  // Split into two parts (sigmoid and additive) - use slicing on channel axis (axis=1 in NCHW)
  int half_filters = NumFilters();
  code.str("");
  code << name << "_split_0 = " << name << "_reshape[:, :" << half_filters << ", :, :]";
  py_.AppendPython(code.str());
  
  code.str("");
  code << name << "_split_1 = " << name << "_reshape[:, " << half_filters << ":, :, :]";
  py_.AppendPython(code.str());
  
  code.str("");
  code << name << "_split = [" << name << "_split_0, " << name << "_split_1]";
  py_.AppendPython(code.str());
  code.str("");
  code << name << "_split_0 = " << name << "_split[0]";
  py_.AppendPython(code.str());
  code.str("");
  code << name << "_split_1 = " << name << "_split[1]";
  py_.AppendPython(code.str());
  
  // Apply sigmoid and multiply
  code.str("");
  code << name << "_sigmoid = layers.Activation('sigmoid', name='" 
       << name << "_sigmoid')(" << name << "_split[0])";
  py_.AppendPython(code.str());
  
  code.str("");
  code << name << "_mul = layers.Multiply(name='" << name << "_mul')("
       << "[" << input_var << ", " << name << "_sigmoid])";
  py_.AppendPython(code.str());
  
  // Add the second part
  code.str("");
  code << name << "_out = layers.Add(name='" << name << "_out')("
       << "[" << name << "_mul, " << name << "_split[1]])";
  py_.AppendPython(code.str());
  
  return name + "_out";
}

std::string KerasConverter::MakeResidualBlock(
    const std::string& input_var,
    const MultiHeadWeights::Residual& res,
    const std::string& block_name) {
  
  // First conv block
  std::string block1 = MakeConvLayer(input_var, res.conv1, NumFilters(), NumFilters(),
                                     block_name + "_conv1", 3, true);
  
  // Second conv block (without activation - will add residual first)
  std::string block2 = MakeConvLayer(block1, res.conv2, NumFilters(), NumFilters(),
                                     block_name + "_conv2", 3, false);
  
  // Apply SE unit if present
  if (res.has_se) {
    block2 = MakeSqueezeAndExcite(block2, res.se, block_name + "_se");
  }
  
  // Add residual connection
  std::ostringstream code;
  code << block_name << "_add = layers.Add(name='" << block_name 
       << "_add')([" << input_var << ", " << block2 << "])";
  py_.AppendPython(code.str());
  
  std::string output = block_name + "_add";
  
  // Apply activation
  output = MakeActivation(output, block_name + "_out", default_activation_);
  
  return output;
}

std::string KerasConverter::MakeLayerNorm(
    const std::string& input_var,
    const std::string& name,
    const std::vector<float>& gammas,
    const std::vector<float>& betas,
    float eps) {
  
  std::string gamma_var = name + "_gamma";
  WeightsToNumpyArray(gamma_var, gammas, {static_cast<int>(gammas.size())});
  
  std::string beta_var = name + "_beta";
  WeightsToNumpyArray(beta_var, betas, {static_cast<int>(betas.size())});
  
  std::ostringstream code;
  code << name << "_layer = layers.LayerNormalization(epsilon=" << eps
       << ", name='" << name << "')";
  py_.AppendPython(code.str());
  
  // Build the layer - LayerNormalization needs input shape
  // The input shape is dynamic, so we use None for batch dimension
  int norm_size = static_cast<int>(gammas.size());
  code.str("");
  code << name << "_layer.build([None, " << norm_size << "])";
  py_.AppendPython(code.str());
  
  code.str("");
  code << name << "_layer.set_weights([" << gamma_var << ", " << beta_var << "])";
  py_.AppendPython(code.str());
  
  code.str("");
  code << name << " = " << name << "_layer(" << input_var << ")";
  py_.AppendPython(code.str());
  
  return name;
}

std::string KerasConverter::MakeFFN(
    const std::string& ffn_in,
    const MultiHeadWeights::FFN& ffn,
    int embedding_size,
    const std::string& name,
    ActivationFunction activation,
    float alpha) {
  
  const int dff_size = ffn.dense1_b.size();
  
  // First dense layer
  std::string w1_var = name + "_ffn_w1";
  WeightsToNumpyArray(w1_var, ffn.dense1_w, {embedding_size, dff_size}, {1, 0});
  
  std::string b1_var = name + "_ffn_b1";
  WeightsToNumpyArray(b1_var, ffn.dense1_b, {dff_size});
  
  std::ostringstream code;
  code << name << "_ffn_dense1_layer = layers.Dense(" << dff_size
       << ", name='" << name << "_ffn_dense1', use_bias=True)";
  py_.AppendPython(code.str());
  
  code.str("");
  code << name << "_ffn_dense1_layer.build([None, " << embedding_size << "])";
  py_.AppendPython(code.str());
  
  code.str("");
  code << name << "_ffn_dense1_layer.set_weights([" << w1_var << ", " << b1_var << "])";
  py_.AppendPython(code.str());
  
  code.str("");
  code << name << "_ffn_dense1 = " << name << "_ffn_dense1_layer(" << ffn_in << ")";
  py_.AppendPython(code.str());
  
  std::string dense1_act = MakeActivation(name + "_ffn_dense1", name + "_ffn_dense1_act", 
                                         activation);
  
  // Second dense layer
  std::string w2_var = name + "_ffn_w2";
  WeightsToNumpyArray(w2_var, ffn.dense2_w, {dff_size, embedding_size}, {1, 0});
  
  std::string b2_var = name + "_ffn_b2";
  WeightsToNumpyArray(b2_var, ffn.dense2_b, {embedding_size});
  
  code.str("");
  code << name << "_ffn_dense2_layer = layers.Dense(" << embedding_size
       << ", name='" << name << "_ffn_dense2', use_bias=True)";
  py_.AppendPython(code.str());
  
  code.str("");
  code << name << "_ffn_dense2_layer.build([None, " << dff_size << "])";
  py_.AppendPython(code.str());
  
  code.str("");
  code << name << "_ffn_dense2_layer.set_weights([" << w2_var << ", " << b2_var << "])";
  py_.AppendPython(code.str());
  
  code.str("");
  code << name << "_ffn_dense2 = " << name << "_ffn_dense2_layer(" << dense1_act << ")";
  py_.AppendPython(code.str());
  
  // Apply alpha scaling if needed
  if (alpha != 1.0f) {
    code.str("");
    code << name << "_ffn_dense2 = layers.Rescaling(" << std::setprecision(9)
         << std::fixed << alpha << ", name='" << name << "_ffn_alpha')("
         << name << "_ffn_dense2)";
    py_.AppendPython(code.str());
  }
  
  // Skip connection
  code.str("");
  code << name << "_ffn_out = layers.Add(name='" << name << "_ffn_skip')("
       << "[" << ffn_in << ", " << name << "_ffn_dense2])";
  py_.AppendPython(code.str());
  
  return name + "_ffn_out";
}

std::string KerasConverter::MakeSmolgen(
    const MultiHeadWeights::EncoderLayer& layer,
    int embedding_size, int heads,
    const std::string& encoder_in,
    const std::string& name,
    const MultiHeadWeights& weights) {
  
  const auto smolgen_activation = static_cast<ActivationFunction>(
      src_.format().network_format().smolgen_activation());
  const auto activation = smolgen_activation == ACTIVATION_DEFAULT
                              ? default_activation_
                              : smolgen_activation;
  
  const int smolgen_hidden_channels =
      layer.mha.smolgen.compress.size() / embedding_size;
  const int smolgen_hidden_sz = layer.mha.smolgen.dense1_b.size();
  const int smolgen_gen_sz = layer.mha.smolgen.dense2_b.size() / heads;
  
  // Compress
  std::string compress_w_var = name + "_smolgen_compress_w";
  WeightsToNumpyArray(compress_w_var, layer.mha.smolgen.compress,
                     {embedding_size, smolgen_hidden_channels}, {1, 0});
  
  std::ostringstream code;
  std::string sanitized_name = SanitizeLayerName(name);
  code << name << "_smolgen_compress_layer = layers.Dense(" << smolgen_hidden_channels
       << ", name='" << sanitized_name << "_smolgen_compress', use_bias=False)";
  py_.AppendPython(code.str());
  
  code.str("");
  code << name << "_smolgen_compress_layer.build([None, " << embedding_size << "])";
  py_.AppendPython(code.str());
  
  code.str("");
  code << name << "_smolgen_compress_layer.set_weights([" << compress_w_var << "])";
  py_.AppendPython(code.str());
  
  code.str("");
  code << name << "_smolgen_compress = " << name << "_smolgen_compress_layer(" 
       << encoder_in << ")";
  py_.AppendPython(code.str());
  
  // Unflatten from (batch*64, hidden_channels) to (batch, 64, hidden_channels)
  // then flatten to (batch, 64*hidden_channels)
  code.str("");
  code << name << "_smolgen_compress = UnflattenBatchSpatial(" << smolgen_hidden_channels
       << ", name='" << sanitized_name << "_smolgen_unflatten')(" << name << "_smolgen_compress)";
  py_.AppendPython(code.str());
  
  code.str("");
  code << name << "_smolgen_compress = layers.Reshape((" << (64 * smolgen_hidden_channels) 
       << ",), name='" << sanitized_name << "_smolgen_compress_reshape')(" << name
       << "_smolgen_compress)";
  py_.AppendPython(code.str());
  
  // Dense1
  std::string d1_w_var = name + "_smolgen_d1_w";
  WeightsToNumpyArray(d1_w_var, layer.mha.smolgen.dense1_w,
                     {64 * smolgen_hidden_channels, smolgen_hidden_sz}, {1, 0});
  
  std::string d1_b_var = name + "_smolgen_d1_b";
  WeightsToNumpyArray(d1_b_var, layer.mha.smolgen.dense1_b, {smolgen_hidden_sz});
  
  code.str("");
  code << name << "_smolgen_d1_layer = layers.Dense(" << smolgen_hidden_sz
       << ", name='" << sanitized_name << "_smolgen_d1', use_bias=True)";
  py_.AppendPython(code.str());
  
  code.str("");
  code << name << "_smolgen_d1_layer.build([None, " << (64 * smolgen_hidden_channels) << "])";
  py_.AppendPython(code.str());
  
  code.str("");
  code << name << "_smolgen_d1_layer.set_weights([" << d1_w_var << ", " << d1_b_var << "])";
  py_.AppendPython(code.str());
  
  code.str("");
  code << name << "_smolgen_d1 = " << name << "_smolgen_d1_layer(" 
       << name << "_smolgen_compress)";
  py_.AppendPython(code.str());
  
  std::string d1_act = MakeActivation(name + "_smolgen_d1", name + "_smolgen_d1_act", 
                                     activation);
  
  std::string ln1_out = MakeLayerNorm(d1_act, name + "_smolgen_ln1",
                                      layer.mha.smolgen.ln1_gammas,
                                      layer.mha.smolgen.ln1_betas, 1e-3f);
  
  // Dense2
  std::string d2_w_var = name + "_smolgen_d2_w";
  WeightsToNumpyArray(d2_w_var, layer.mha.smolgen.dense2_w,
                     {smolgen_hidden_sz, smolgen_gen_sz * heads}, {1, 0});
  
  std::string d2_b_var = name + "_smolgen_d2_b";
  WeightsToNumpyArray(d2_b_var, layer.mha.smolgen.dense2_b, {smolgen_gen_sz * heads});
  
  code.str("");
  code << name << "_smolgen_d2_layer = layers.Dense(" << (smolgen_gen_sz * heads)
       << ", name='" << sanitized_name << "_smolgen_d2', use_bias=True)";
  py_.AppendPython(code.str());
  
  code.str("");
  code << name << "_smolgen_d2_layer.build([None, " << smolgen_hidden_sz << "])";
  py_.AppendPython(code.str());
  
  code.str("");
  code << name << "_smolgen_d2_layer.set_weights([" << d2_w_var << ", " << d2_b_var << "])";
  py_.AppendPython(code.str());
  
  code.str("");
  code << name << "_smolgen_d2 = " << name << "_smolgen_d2_layer(" << ln1_out << ")";
  py_.AppendPython(code.str());
  
  std::string d2_act = MakeActivation(name + "_smolgen_d2", name + "_smolgen_d2_act", 
                                     activation);
  
  std::string ln2_out = MakeLayerNorm(d2_act, name + "_smolgen_ln2",
                                      layer.mha.smolgen.ln2_gammas,
                                      layer.mha.smolgen.ln2_betas, 1e-3f);
  
  // Reshape and multiply with smolgen weights - use DynamicReshape for dynamic batch
  code.str("");
  code << name << "_smolgen_reshape = DynamicReshape(target_shape=(-1, " << heads << ", " 
       << smolgen_gen_sz << "), output_shape_tuple=(None, " << heads << ", " 
       << smolgen_gen_sz << "), name='" << sanitized_name << "_smolgen_reshape')(" << ln2_out << ")";
  py_.AppendPython(code.str());
  
  // Create smolgen_w constant if not already created
  if (!smolgen_w_created_) {
    std::string smolgen_w_var = "smolgen_w";
    int smolgen_w_rows = static_cast<int>(weights.smolgen_w.size() / 4096);
    WeightsToNumpyArray(smolgen_w_var, weights.smolgen_w,
                       {smolgen_w_rows, 4096}, {1, 0});
    smolgen_w_created_ = true;
  }
  
  // Use custom MatMul layer instead of Lambda
  code.str("");
  code << name << "_smolgen_weight_gen = MatMul(name='" << name << "_smolgen_weight_gen')([" << name << "_smolgen_reshape, smolgen_w])";
  py_.AppendPython(code.str());
  
  code.str("");
  code << name << "_smolgen_out = layers.Reshape((" << heads << ", 64, 64), name='" 
       << sanitized_name << "_smolgen_out')(" << name << "_smolgen_weight_gen)";
  py_.AppendPython(code.str());
  
  return name + "_smolgen_out";
}

std::string KerasConverter::MakeEncoderLayer(
    const std::string& encoder_in,
    const MultiHeadWeights::EncoderLayer& layer,
    int embedding_size, int heads,
    const std::string& name,
    ActivationFunction activation,
    const MultiHeadWeights& weights,
    float alpha) {
  
  const int d_model = layer.mha.q_b.size();
  const int depth = d_model / heads;
  
  std::string sanitized_name = SanitizeLayerName(name);
  
  // Multi-head attention
  // Q
  std::string q_w_var = name + "_q_w";
  WeightsToNumpyArray(q_w_var, layer.mha.q_w, {embedding_size, d_model}, {1, 0});
  
  std::string q_b_var = name + "_q_b";
  WeightsToNumpyArray(q_b_var, layer.mha.q_b, {d_model});
  
  std::ostringstream code;
  code << name << "_q_layer = layers.Dense(" << d_model
       << ", name='" << name << "_q', use_bias=True)";
  py_.AppendPython(code.str());
  
  code.str("");
  code << name << "_q_layer.build([None, " << embedding_size << "])";
  py_.AppendPython(code.str());
  
  code.str("");
  code << name << "_q_layer.set_weights([" << q_w_var << ", " << q_b_var << "])";
  py_.AppendPython(code.str());
  
  code.str("");
  code << name << "_q = " << name << "_q_layer(" << encoder_in << ")";
  py_.AppendPython(code.str());
  
  // Use DynamicReshape for reshaping with dynamic batch
  code.str("");
  code << name << "_q = DynamicReshape(target_shape=(-1, 64, " << heads << ", " << depth 
       << "), output_shape_tuple=(None, 64, " << heads << ", " << depth 
       << "), name='" << name << "_q_reshape')(" << name << "_q)";
  py_.AppendPython(code.str());
  
  // Use Permute for transpose: [0, 2, 1, 3] -> (2, 1, 3) in 1-indexed
  code.str("");
  code << name << "_q = layers.Permute((2, 1, 3), name='" << name << "_q_transpose')(" << name << "_q)";
  py_.AppendPython(code.str());
  
  // K
  std::string k_w_var = name + "_k_w";
  WeightsToNumpyArray(k_w_var, layer.mha.k_w, {embedding_size, d_model}, {1, 0});
  
  std::string k_b_var = name + "_k_b";
  WeightsToNumpyArray(k_b_var, layer.mha.k_b, {d_model});
  
  code.str("");
  code << name << "_k_layer = layers.Dense(" << d_model
       << ", name='" << name << "_k', use_bias=True)";
  py_.AppendPython(code.str());
  
  code.str("");
  code << name << "_k_layer.build([None, " << embedding_size << "])";
  py_.AppendPython(code.str());
  
  code.str("");
  code << name << "_k_layer.set_weights([" << k_w_var << ", " << k_b_var << "])";
  py_.AppendPython(code.str());
  
  code.str("");
  code << name << "_k = " << name << "_k_layer(" << encoder_in << ")";
  py_.AppendPython(code.str());
  
  // Use DynamicReshape for reshaping with dynamic batch
  code.str("");
  code << name << "_k = DynamicReshape(target_shape=(-1, 64, " << heads << ", " << depth 
       << "), output_shape_tuple=(None, 64, " << heads << ", " << depth 
       << "), name='" << name << "_k_reshape')(" << name << "_k)";
  py_.AppendPython(code.str());
  
  // Use Permute for transpose: [0, 2, 3, 1] -> (2, 3, 1) in 1-indexed
  code.str("");
  code << name << "_k = layers.Permute((2, 3, 1), name='" << name << "_k_transpose')(" << name << "_k)";
  py_.AppendPython(code.str());
  
  // V
  std::string v_w_var = name + "_v_w";
  WeightsToNumpyArray(v_w_var, layer.mha.v_w, {embedding_size, d_model}, {1, 0});
  
  std::string v_b_var = name + "_v_b";
  WeightsToNumpyArray(v_b_var, layer.mha.v_b, {d_model});
  
  code.str("");
  code << name << "_v_layer = layers.Dense(" << d_model
       << ", name='" << name << "_v', use_bias=True)";
  py_.AppendPython(code.str());
  
  code.str("");
  code << name << "_v_layer.build([None, " << embedding_size << "])";
  py_.AppendPython(code.str());
  
  code.str("");
  code << name << "_v_layer.set_weights([" << v_w_var << ", " << v_b_var << "])";
  py_.AppendPython(code.str());
  
  code.str("");
  code << name << "_v = " << name << "_v_layer(" << encoder_in << ")";
  py_.AppendPython(code.str());
  
  // Use DynamicReshape for reshaping with dynamic batch
  code.str("");
  code << name << "_v = DynamicReshape(target_shape=(-1, 64, " << heads << ", " << depth 
       << "), output_shape_tuple=(None, 64, " << heads << ", " << depth 
       << "), name='" << name << "_v_reshape')(" << name << "_v)";
  py_.AppendPython(code.str());
  
  // Use Permute for transpose: [0, 2, 1, 3] -> (2, 1, 3) in 1-indexed
  code.str("");
  code << name << "_v = layers.Permute((2, 1, 3), name='" << name << "_v_transpose')(" << name << "_v)";
  py_.AppendPython(code.str());
  
  // QK matmul - use custom MatMul layer
  code.str("");
  code << name << "_qk = MatMul(name='" << name << "_qk_matmul')([" << name << "_q, " << name << "_k])";
  py_.AppendPython(code.str());
  
  // Scale - use Rescaling layer for scalar multiplication
  float scale = 1.0f / sqrtf(static_cast<float>(depth));
  
  code.str("");
  code << name << "_qk = layers.Rescaling(scale=" << std::setprecision(9) << std::fixed << scale 
       << ", name='" << name << "_qk_scale')(" << name << "_qk)";
  py_.AppendPython(code.str());
  
  // Add smolgen weights if present
  if (layer.mha.has_smolgen) {
    std::string smolgen_weights = MakeSmolgen(layer, embedding_size, heads, encoder_in, name, weights);
    code.str("");
    code << name << "_qk = layers.Add(name='" << sanitized_name << "_smolgen_add')("
         << "[" << name << "_qk, " << smolgen_weights << "])";
    py_.AppendPython(code.str());
  }
  
  // Softmax
  code.str("");
  code << name << "_qk = layers.Softmax(axis=-1, name='" << sanitized_name << "_qk_softmax')(" 
       << name << "_qk)";
  py_.AppendPython(code.str());
  
  // QKV matmul - use custom MatMul layer
  code.str("");
  code << name << "_qkv = MatMul(name='" << name << "_qkv_matmul')([" << name << "_qk, " << name << "_v])";
  py_.AppendPython(code.str());
  
  // Transpose and reshape if multi-head
  if (heads > 1) {
    // Use Permute for transpose: [0, 2, 1, 3] -> (2, 1, 3) in 1-indexed
    code.str("");
    code << name << "_qkv = layers.Permute((2, 1, 3), name='" << name << "_qkv_transpose')(" << name << "_qkv)";
    py_.AppendPython(code.str());
  }

  // Use DynamicReshape for flattening with dynamic batch
  code.str("");
  code << name << "_qkv = DynamicReshape(target_shape=(-1, "
       << d_model << "), output_shape_tuple=(None, " << d_model 
       << "), name='" << sanitized_name << "_qkv_reshape')("
       << name << "_qkv)";
  py_.AppendPython(code.str());
  
  // Output dense
  std::string dense_w_var = name + "_mha_dense_w";
  WeightsToNumpyArray(dense_w_var, layer.mha.dense_w, {d_model, embedding_size}, {1, 0});
  
  std::string dense_b_var = name + "_mha_dense_b";
  WeightsToNumpyArray(dense_b_var, layer.mha.dense_b, {embedding_size});
  
  code.str("");
  code << name << "_mha_dense_layer = layers.Dense(" << embedding_size
       << ", name='" << name << "_mha_dense', use_bias=True)";
  py_.AppendPython(code.str());
  
  code.str("");
  code << name << "_mha_dense_layer.build([None, " << d_model << "])";
  py_.AppendPython(code.str());
  
  code.str("");
  code << name << "_mha_dense_layer.set_weights([" << dense_w_var << ", " 
       << dense_b_var << "])";
  py_.AppendPython(code.str());
  
  code.str("");
  code << name << "_mha_dense = " << name << "_mha_dense_layer(" << name << "_qkv)";
  py_.AppendPython(code.str());
  
  // Apply alpha scaling if needed
  if (alpha != 1.0f) {
    // Use Rescaling layer for alpha multiplication
    code.str("");
    code << name << "_mha_dense = layers.Rescaling(scale=" << std::setprecision(9) << std::fixed << alpha 
         << ", name='" << sanitized_name << "_alpha_mul')(" << name << "_mha_dense)";
    py_.AppendPython(code.str());
  }
  
  // Skip connection
  code.str("");
  code << name << "_mha_out = layers.Add(name='" << name << "_mha_skip')("
       << "[" << encoder_in << ", " << name << "_mha_dense])";
  py_.AppendPython(code.str());
  
  // Layer norm 1
  std::string ln1_out = MakeLayerNorm(name + "_mha_out", name + "_ln1",
                                     layer.ln1_gammas, layer.ln1_betas, default_eps_);
  
  // FFN
  const auto ffn_activation = static_cast<ActivationFunction>(
      src_.format().network_format().ffn_activation());
  auto final_ffn_activation = ffn_activation == ACTIVATION_DEFAULT 
                                  ? activation 
                                  : ffn_activation;
  
  std::string ffn_out = MakeFFN(ln1_out, layer.ffn, embedding_size, name,
                               final_ffn_activation, alpha);
  
  // Layer norm 2
  std::string ln2_out = MakeLayerNorm(ffn_out, name + "_ln2",
                                     layer.ln2_gammas, layer.ln2_betas, default_eps_);
  
  return ln2_out;
}

std::string KerasConverter::AttentionBodyMapEmbedding(const std::string& input) {
  // Reshape to (batch, 64, 112) - use DynamicReshape for dynamic batch
  std::ostringstream code;
  code << "attn_body_reshape = DynamicReshape(target_shape=(-1, 64, 112), output_shape_tuple=(None, 64, 112), name='attn_body_reshape')(" 
       << input << ")";
  py_.AppendPython(code.str());
  
  // Position encoding padding logic
  // Create position encoding array from kPosEncoding constant (64x64 matrix)
  std::vector<float> pos_encoding;
  pos_encoding.reserve(64 * 64);
  for (int i = 0; i < 64; ++i) {
    for (int j = 0; j < kNumPosEncodingChannels; ++j) {
      pos_encoding.push_back(kPosEncoding[i][j]);
    }
  }
  
  std::string pos_enc_var = "pos_encoding_const";
  py_.CreateNumpyArray(pos_enc_var, pos_encoding, {64, 64});  // Save as (64, 64) instead of (1, 64, 64)
  
  if (options_.batch_size > 0) {
    // Known batch size - use custom StaticRepeat layer
    code.str("");
    code << "pos_encoding_expanded = StaticRepeat(repeats=" << options_.batch_size << ", axis=0, name='pos_encoding_repeat')(" << pos_enc_var << ")";
    py_.AppendPython(code.str());
  } else {
    // Dynamic batch - use custom DynamicTile layer
    code.str("");
    code << "# Expand position encoding dynamically based on batch size\n";
    code << "pos_encoding_expanded = DynamicTile(name='pos_encoding_broadcast')([attn_body_reshape, " << pos_enc_var << "])";
    py_.AppendPython(code.str());
  }
  
  code.str("");
  code << "attn_body_padded = layers.Concatenate(axis=-1, name='attn_body_concat')([" << "attn_body_reshape, pos_encoding_expanded])";
  py_.AppendPython(code.str());
  
  // Reshape from (batch, 64, 176) to (batch*64, 176) by flattening first two dimensions
  code.str("");
  code << "attn_body_out = FlattenBatchSpatial(176, name='attn_body_out')(attn_body_padded)";
  py_.AppendPython(code.str());
  
  return "attn_body_out";
}

std::string KerasConverter::AttentionBodyDenseEmbedding(
    const std::string& input,
    const MultiHeadWeights& weights,
    int embedding_dense_size) {
  
  std::ostringstream code;
  
  // Reshape to (batch, 64, 112) - use DynamicReshape for dynamic batch
  code << "attn_body_reshape = DynamicReshape(target_shape=(-1, 64, 112), output_shape_tuple=(None, 64, 112), name='attn_body_reshape')(" 
       << input << ")";
  py_.AppendPython(code.str());
  
  // Slice position info (first 12 channels) - use custom SliceLayer
  code.str("");
  code << "pos_info = SliceLayer(start_indices=[0, 0, 0], sizes=[-1, 64, 12], name='pos_info_slice')(attn_body_reshape)";
  py_.AppendPython(code.str());
  
  // Reshape
  code.str("");
  code << "pos_info_flat = layers.Reshape((64 * 12,), name='pos_info_flat')(" 
       << "pos_info)";
  py_.AppendPython(code.str());
  
  // Preprocess dense layer
  std::string prep_w_var = "attn_emb_preproc_w";
  WeightsToNumpyArray(prep_w_var, weights.ip_emb_preproc_w,
                     {64 * 12, 64 * embedding_dense_size}, {1, 0});
  
  std::string prep_b_var = "attn_emb_preproc_b";
  WeightsToNumpyArray(prep_b_var, weights.ip_emb_preproc_b,
                     {64 * embedding_dense_size});
  
  code.str("");
  code << "attn_emb_preproc_layer = layers.Dense(" << (64 * embedding_dense_size)
       << ", name='attn_emb_preproc', use_bias=True)";
  py_.AppendPython(code.str());
  
  code.str("");
  code << "attn_emb_preproc_layer.build([None, " << (64 * 12) << "])";
  py_.AppendPython(code.str());
  
  code.str("");
  code << "attn_emb_preproc_layer.set_weights([" << prep_w_var << ", " << prep_b_var << "])";
  py_.AppendPython(code.str());
  
  code.str("");
  code << "pos_info_proc = attn_emb_preproc_layer(pos_info_flat)";
  py_.AppendPython(code.str());
  
  // Reshape
  code.str("");
  code << "pos_info_proc = layers.Reshape((64, " << embedding_dense_size 
       << "), name='pos_info_proc')(" << "pos_info_proc)";
  py_.AppendPython(code.str());
  
  // Concat with input
  code.str("");
  code << "attn_body_concat = layers.Concatenate(axis=-1, name='attn_body_concat')([attn_body_reshape, pos_info_proc])";
  py_.AppendPython(code.str());
  
  // Reshape to (batch*64, 112 + embedding_dense_size)
  // ONNX Reshape with -1 flattens batch and spatial dimensions: (batch, 64, C) -> (batch*64, C)
  code.str("");
  code << "attn_body_out = FlattenBatchSpatial(" << (112 + embedding_dense_size) 
       << ", name='attn_body_out')(attn_body_concat)";
  py_.AppendPython(code.str());
  
  return "attn_body_out";
}

std::string KerasConverter::MakeAttentionBody(
    const std::string& input,
    const MultiHeadWeights& weights) {
  
  // Create smolgen_w constant if needed (will be used by MakeSmolgen if present)
  if (weights.has_smolgen && weights.smolgen_w.size() > 0 && !smolgen_w_created_) {
    std::string smolgen_w_var = "smolgen_w";
    int smolgen_w_rows = static_cast<int>(weights.smolgen_w.size() / 4096);
    WeightsToNumpyArray(smolgen_w_var, weights.smolgen_w,
                       {smolgen_w_rows, 4096}, {1, 0});
    smolgen_w_created_ = true;
  }
  
  // Transpose from NCHW to NHWC using Permute (1-indexed, batch dim excluded)
  // ops.transpose(x, [0, 2, 3, 1]) -> Permute([2, 3, 1])
  std::ostringstream code;
  code << "attn_body_transpose = layers.Permute((2, 3, 1), name='attn_body_transpose')(" << input << ")";
  py_.AppendPython(code.str());
  
  auto input_embedding = src_.format().network_format().input_embedding();
  using network_format = pblczero::NetworkFormat;
  std::string flow = "attn_body_transpose";
  int first_stage_out_C = 0;
  
  if (NumResBlocks() > 0) {
    // Reshape from residual output: (batch, 8, 8, filters) -> (batch*64, filters)
    code.str("");
    code << "attn_body_reshape = FlattenBatchSpatial(" << NumFilters() 
         << ", name='attn_body_reshape')(" << flow << ")";
    py_.AppendPython(code.str());
    flow = "attn_body_reshape";
    first_stage_out_C = NumFilters();
  } else if (input_embedding == network_format::INPUT_EMBEDDING_PE_MAP) {
    flow = AttentionBodyMapEmbedding(flow);
    first_stage_out_C = 176;
  } else if (input_embedding == network_format::INPUT_EMBEDDING_PE_DENSE) {
    int embedding_dense_size = weights.ip_emb_preproc_b.size() / 64;
    flow = AttentionBodyDenseEmbedding(flow, weights, embedding_dense_size);
    first_stage_out_C = 112 + embedding_dense_size;
  } else {
    throw Exception("Attention body missing input embedding.");
  }
  
  // Embedding projection
  int embedding_size = weights.ip_emb_b.size();
  std::string emb_w_var = "attn_ip_emb_w";
  WeightsToNumpyArray(emb_w_var, weights.ip_emb_w,
                     {first_stage_out_C, embedding_size}, {1, 0});
  
  std::string emb_b_var = "attn_ip_emb_b";
  WeightsToNumpyArray(emb_b_var, weights.ip_emb_b, {embedding_size});
  
  code.str("");
  code << "attn_ip_emb_layer = layers.Dense(" << embedding_size
       << ", name='attn_ip_emb', use_bias=True)";
  py_.AppendPython(code.str());
  
  // Build the layer - need to know input shape, which depends on flow
  // For attn_body_out, it's (batch*64, first_stage_out_C) after reshape
  code.str("");
  code << "attn_ip_emb_layer.build([None, " << first_stage_out_C << "])";
  py_.AppendPython(code.str());
  
  code.str("");
  code << "attn_ip_emb_layer.set_weights([" << emb_w_var << ", " << emb_b_var << "])";
  py_.AppendPython(code.str());
  
  code.str("");
  code << "attn_body_emb = attn_ip_emb_layer(" << flow << ")";
  py_.AppendPython(code.str());
  
  flow = MakeActivation("attn_body_emb", "attn_body_emb_act", default_activation_);
  
  // Layer norm if dense embedding
  if (input_embedding == network_format::INPUT_EMBEDDING_PE_DENSE) {
    flow = MakeLayerNorm(flow, "attn_body_ln",
                        weights.ip_emb_ln_gammas,
                        weights.ip_emb_ln_betas, 1e-3f);
  }
  
  // Multi-gate if present
  // Note: ONNX reshapes to (-1, 64, embedding_size) first, then applies gates
  if (weights.ip_mult_gate.size() > 0 || weights.ip_add_gate.size() > 0) {
    // Reshape from (batch*64, embedding_size) to (batch, 64, embedding_size)
    code.str("");
    code << flow << " = UnflattenBatchSpatial(" << embedding_size 
         << ", name='attn_gating_reshape')(" << flow << ")";
    py_.AppendPython(code.str());
    
    if (weights.ip_mult_gate.size() > 0) {
      std::string mult_gate_var = "attn_mult_gate";
      WeightsToNumpyArray(mult_gate_var, weights.ip_mult_gate, {64, embedding_size}, {1, 0});
      
      // Expand dims to (1, 64, embedding_size) for proper broadcasting with (batch, 64, embedding_size)
      code.str("");
      code << "attn_mult_gate_const = ops.expand_dims(" << mult_gate_var << ", axis=0)";
      py_.AppendPython(code.str());
      
      code.str("");
      code << flow << " = layers.Multiply(name='attn_mult_gate')("
           << "[" << flow << ", attn_mult_gate_const])";
      py_.AppendPython(code.str());
    }
    
    if (weights.ip_add_gate.size() > 0) {
      std::string add_gate_var = "attn_add_gate";
      WeightsToNumpyArray(add_gate_var, weights.ip_add_gate, {64, embedding_size}, {1, 0});
      
      // Expand dims to (1, 64, embedding_size) for proper broadcasting with (batch, 64, embedding_size)
      code.str("");
      code << "attn_add_gate_const = ops.expand_dims(" << add_gate_var << ", axis=0)";
      py_.AppendPython(code.str());
      
      code.str("");
      code << flow << " = layers.Add(name='attn_add_gate')("
           << "[" << flow << ", attn_add_gate_const])";
      py_.AppendPython(code.str());
    }
    
    // Reshape back to (batch*64, embedding_size)
    code.str("");
    code << flow << " = FlattenBatchSpatial(" << embedding_size 
         << ", name='attn_gating_reshape_back')(" << flow << ")";
    py_.AppendPython(code.str());
  }
  
  // Encoder layers
  int heads = weights.encoder_head_count;
  float alpha = std::pow(2.0f * NumEncBlocks(), -0.25f);
  for (size_t i = 0; i < NumEncBlocks(); ++i) {
    std::string enc_name = "enc_layer_" + std::to_string(i);
    flow = MakeEncoderLayer(flow, weights.encoder[i], embedding_size, heads,
                           enc_name, default_activation_, weights, alpha);
  }
  
  return flow;
}

namespace {
std::vector<int> MakePolicyMap(const short* map, int size) {
  std::vector<int> policy_map(1858);
  int idx = 0;
  for (int i = 0; i < size; i++) {
    if (map[i] > -1) policy_map[map[i]] = idx;
    idx++;
  }
  return policy_map;
}
}  // namespace

std::string KerasConverter::MakeAttentionPolicy(
    const std::string& input,
    const MultiHeadWeights& weights,
    const MultiHeadWeights::PolicyHead& head) {
  
  if (head.ip2_pol_b.empty()) {
    throw Exception("The policy head selected '" + options_.policy_head + "'"
                    " is empty.");
  }
  
  const int embedding_size = weights.ip_emb_b.size();
  const int policy_embedding_size = head.ip_pol_b.size();
  const int policy_d_model = head.ip2_pol_b.size();
  
  std::ostringstream code;
  std::string flow = input;
  
  auto activation = src_.format().network_format().network() >=
                            pblczero::NetworkFormat::
                                NETWORK_ATTENTIONBODY_WITH_HEADFORMAT
                        ? default_activation_
                        : ACTIVATION_SELU;
  
  // Transpose and reshape if coming from residual blocks
  if (NumEncBlocks() == 0) {
    code << "policy_transpose = layers.Permute((2, 3, 1), name='policy_transpose')(" << input << ")";
    py_.AppendPython(code.str());
    
    code.str("");
    code << "policy_reshape = layers.Reshape((" << NumFilters() 
         << ",), name='policy_reshape')(" << "policy_transpose)";
    py_.AppendPython(code.str());
    flow = "policy_reshape";
  }
  
  // First dense layer
  std::string ip_pol_w_var = "policy_ip_pol_w";
  WeightsToNumpyArray(ip_pol_w_var, head.ip_pol_w,
                     {NumEncBlocks() > 0 ? embedding_size : NumFilters(),
                      policy_embedding_size}, {1, 0});
  
  std::string ip_pol_b_var = "policy_ip_pol_b";
  WeightsToNumpyArray(ip_pol_b_var, head.ip_pol_b, {policy_embedding_size});
  
  code.str("");
  code << "policy_dense1_layer = layers.Dense(" << policy_embedding_size
       << ", name='policy_dense1', use_bias=True)";
  py_.AppendPython(code.str());
  
  code.str("");
  int policy_input_size = NumEncBlocks() > 0 ? embedding_size : NumFilters();
  code << "policy_dense1_layer.build([None, " << policy_input_size << "])";
  py_.AppendPython(code.str());
  
  code.str("");
  code << "policy_dense1_layer.set_weights([" << ip_pol_w_var << ", " 
       << ip_pol_b_var << "])";
  py_.AppendPython(code.str());
  
  code.str("");
  code << "policy_dense1 = policy_dense1_layer(" << flow << ")";
  py_.AppendPython(code.str());
  
  flow = MakeActivation("policy_dense1", "policy_dense1_act", activation);
  
  // Policy encoder layers (no alpha scaling - defaults to 1.0)
  for (size_t i = 0; i < head.pol_encoder.size(); i++) {
    std::string enc_name = "policy_enc_layer_" + std::to_string(i);
    flow = MakeEncoderLayer(flow, head.pol_encoder[i], policy_embedding_size,
                           head.pol_encoder_head_count, enc_name, activation, weights);
  }
  
  std::string encoder_out = flow;
  
  // Q projection
  std::string ip2_pol_w_var = "policy_ip2_pol_w";
  WeightsToNumpyArray(ip2_pol_w_var, head.ip2_pol_w,
                     {policy_embedding_size, policy_d_model}, {1, 0});
  
  std::string ip2_pol_b_var = "policy_ip2_pol_b";
  WeightsToNumpyArray(ip2_pol_b_var, head.ip2_pol_b, {policy_d_model});
  
  code.str("");
  code << "policy_q_layer = layers.Dense(" << policy_d_model
       << ", name='policy_q', use_bias=True)";
  py_.AppendPython(code.str());
  
  code.str("");
  code << "policy_q_layer.build([None, " << policy_embedding_size << "])";
  py_.AppendPython(code.str());
  
  code.str("");
  code << "policy_q_layer.set_weights([" << ip2_pol_w_var << ", " 
       << ip2_pol_b_var << "])";
  py_.AppendPython(code.str());
  
  code.str("");
  code << "policy_q = policy_q_layer(" << encoder_out << ")";
  py_.AppendPython(code.str());
  
  // Use UnflattenBatchSpatial for dynamic batch unflattening
  code.str("");
  code << "policy_q = UnflattenBatchSpatial(" << policy_d_model 
       << ", name='policy_q_reshape')(policy_q)";
  py_.AppendPython(code.str());
  
  // K projection
  std::string ip3_pol_w_var = "policy_ip3_pol_w";
  WeightsToNumpyArray(ip3_pol_w_var, head.ip3_pol_w,
                     {policy_embedding_size, policy_d_model}, {1, 0});
  
  std::string ip3_pol_b_var = "policy_ip3_pol_b";
  WeightsToNumpyArray(ip3_pol_b_var, head.ip3_pol_b, {policy_d_model});
  
  code.str("");
  code << "policy_k_layer = layers.Dense(" << policy_d_model
       << ", name='policy_k', use_bias=True)";
  py_.AppendPython(code.str());
  
  code.str("");
  code << "policy_k_layer.build([None, " << policy_embedding_size << "])";
  py_.AppendPython(code.str());
  
  code.str("");
  code << "policy_k_layer.set_weights([" << ip3_pol_w_var << ", " 
       << ip3_pol_b_var << "])";
  py_.AppendPython(code.str());
  
  code.str("");
  code << "policy_k = policy_k_layer(" << encoder_out << ")";
  py_.AppendPython(code.str());
  
  // Use UnflattenBatchSpatial for dynamic batch unflattening
  code.str("");
  code << "policy_k = UnflattenBatchSpatial(" << policy_d_model 
       << ", name='policy_k_reshape')(policy_k)";
  py_.AppendPython(code.str());
  
  // Promotion handling - slice from K before transpose
  // Slice from (batch, 64, policy_d_model) at [0, 56, 0] with sizes [-1, 8, policy_d_model]
  code.str("");
  code << "policy_prom_slice = SliceLayer(start_indices=[0, 56, 0], sizes=[-1, 8, " 
       << policy_d_model << "], name='policy_prom_slice')(policy_k)";
  py_.AppendPython(code.str());
  
  // Now transpose K for QK matmul
  code.str("");
  code << "policy_k = layers.Permute((2, 1), name='policy_k_transpose')(policy_k)";
  py_.AppendPython(code.str());
  
  // QK matmul
  // Use custom MatMul layer
  code.str("");
  code << "policy_qk = MatMul(name='policy_qk_matmul')([policy_q, policy_k])";
  py_.AppendPython(code.str());
  
  // Scale - use Rescaling layer for scalar multiplication
  float scale = 1.0f / sqrtf(static_cast<float>(policy_d_model));
  
  code.str("");
  code << "policy_qk = layers.Rescaling(scale=" << std::setprecision(9) << std::fixed << scale 
       << ", name='policy_qk_scale')(policy_qk)";
  py_.AppendPython(code.str());
  
  // Reshape from (batch, 8, policy_d_model) to (batch*8, policy_d_model)
  // Use FlattenBatchSpatial for flattening - no transpose needed!
  code.str("");
  code << "policy_prom_reshape = FlattenBatchSpatial(" << policy_d_model
       << ", name='policy_prom_reshape')(policy_prom_slice)";
  py_.AppendPython(code.str());
  
  std::string ip4_pol_w_var = "policy_ip4_pol_w";
  WeightsToNumpyArray(ip4_pol_w_var, head.ip4_pol_w,
                     {policy_d_model, 4}, {1, 0});
  
  code.str("");
  code << "policy_prom_layer = layers.Dense(4, name='policy_prom', use_bias=False)";
  py_.AppendPython(code.str());
  
  code.str("");
  code << "policy_prom_layer.build([None, " << policy_d_model << "])";
  py_.AppendPython(code.str());
  
  code.str("");
  code << "policy_prom_layer.set_weights([" << ip4_pol_w_var << "])";
  py_.AppendPython(code.str());
  
  code.str("");
  code << "policy_prom = policy_prom_layer(policy_prom_reshape)";
  py_.AppendPython(code.str());
  
  // Reshape from (batch*8, 4) to (batch, 8, 4) before transpose - use DynamicReshape
  code.str("");
  code << "policy_prom = DynamicReshape(target_shape=(-1, 8, 4), output_shape_tuple=(None, 8, 4), name='policy_prom_reshape_3d')(policy_prom)";
  py_.AppendPython(code.str());
  
  code.str("");
  code << "policy_prom = layers.Permute((2, 1), name='policy_prom_transpose1')(policy_prom)";
  py_.AppendPython(code.str());
  
  // Split along axis=1 into sizes [3, 1] using SliceLayer
  code.str("");
  code << "policy_prom_split_0 = SliceLayer(start_indices=[0, 0, 0], sizes=[-1, 3, 8], name='policy_prom_split_0')(policy_prom)";
  py_.AppendPython(code.str());
  code.str("");
  code << "policy_prom_split_1 = SliceLayer(start_indices=[0, 3, 0], sizes=[-1, 1, 8], name='policy_prom_split_1')(policy_prom)";
  py_.AppendPython(code.str());
  
  code.str("");
  code << "policy_prom = layers.Add(name='policy_prom_add')([policy_prom_split_0, policy_prom_split_1])";
  py_.AppendPython(code.str());
  
  code.str("");
  code << "policy_prom = layers.Permute((2, 1), name='policy_prom_transpose2')(policy_prom)";
  py_.AppendPython(code.str());
  
  code.str("");
  code << "policy_prom = layers.Reshape((1, 24), name='policy_prom_reshape_1x24')(" 
       << "policy_prom)";
  py_.AppendPython(code.str());
  
  // Promotion slice from qk - use custom SliceLayer
  code.str("");
  code << "policy_qk_slice = SliceLayer(start_indices=[0, 48, 56], sizes=[-1, 8, 8], name='policy_qk_slice')(policy_qk)";
  py_.AppendPython(code.str());
  
  // Use Reshape layer
  code.str("");
  code << "policy_qk_slice = layers.Reshape((64, 1), name='policy_qk_slice_reshape')(policy_qk_slice)";
  py_.AppendPython(code.str());
  
  // Use custom layer for repeat - but this is a repeat along an existing axis, not expand
  // For this, we need a different approach - use ops.repeat directly but in a custom layer
  // Actually, let's create a simple repeat by concatenation
  code.str("");
  code << "policy_qk_slice = layers.Concatenate(axis=-1, name='policy_qk_slice_repeat')([policy_qk_slice, policy_qk_slice, policy_qk_slice])";
  py_.AppendPython(code.str());
  
  code.str("");
  code << "policy_qk_slice = layers.Reshape((8, 24), name='policy_qk_slice_reshape2')(" 
       << "policy_qk_slice)";
  py_.AppendPython(code.str());
  
  code.str("");
  code << "policy_prom = layers.Add(name='policy_prom_add2')([policy_qk_slice, policy_prom])";
  py_.AppendPython(code.str());
  
  code.str("");
  code << "policy_prom = layers.Reshape((3, 64), name='policy_prom_final')(" 
       << "policy_prom)";
  py_.AppendPython(code.str());
  
  // Concat with qk
  code.str("");
  code << "policy_qk_prom = layers.Concatenate(axis=1, name='policy_qk_prom_concat')([policy_qk, policy_prom])";
  py_.AppendPython(code.str());
  
  // Reshape to (batch*64, 67*64) like ONNX does before gather
  // policy_qk_prom is (batch*64, 67, 64) after concat, need to flatten last two dims
  // Use FlattenBatchSpatial for flattening
  code.str("");
  code << "policy_qk_prom = FlattenBatchSpatial(67 * 64, name='policy_qk_prom_reshape')(policy_qk_prom)";
  py_.AppendPython(code.str());
  
  // Create policy mapping and convert to TensorFlow constant
  std::vector<int> policy_map = MakePolicyMap(kAttnPolicyMap, std::size(kAttnPolicyMap));
  std::string map_var = "policy_map";
  WeightsToNumpyArray(map_var, std::vector<float>(policy_map.begin(), policy_map.end()),
                     {1858});
  
  code.str("");
  code << "policy_map_int = ops.cast(ops.convert_to_tensor(" << map_var
       << "), dtype='int32')";
  py_.AppendPython(code.str());
  
  // Gather with axis=1: policy_qk_prom is (batch*64, 67*64), policy_map_int is (1858,)
  // Result: (batch*64, 1858)
  // Use custom Gather layer with output name (matching ONNX: /output/policy -> output_policy)
  code.str("");
  code << "output_policy = Gather(axis=1, name='output_policy')([policy_qk_prom, policy_map_int])";
  py_.AppendPython(code.str());
  
  output_vars_.push_back("output_policy");
  return "output_policy";
}

std::string KerasConverter::MakePolicyHead(const MultiHeadWeights& weights,
                                          const std::string& input_var) {
  if (weights.policy_heads.count(options_.policy_head) == 0) {
    throw Exception("The policy head you specified '" + options_.policy_head +
                    "' does not exist in this net.");
  }
  
  const MultiHeadWeights::PolicyHead& head =
      weights.policy_heads.at(options_.policy_head);
  
  std::ostringstream code;
  
  if (src_.format().network_format().policy() ==
      pblczero::NetworkFormat::POLICY_ATTENTION) {
    return MakeAttentionPolicy(input_var, weights, head);
  }
  
  if (head.policy.weights.empty()) {
    throw Exception("The policy head selected '" + options_.policy_head + "'"
                    " is empty.");
  }
  
  if (!head.policy1.weights.empty()) {
    // Convolutional policy head
    if (NumEncBlocks() > 0) {
      throw Exception("Convolutional policy not supported with attention body.");
    }
    
    std::string conv1 = MakeConvLayer(input_var, head.policy1, NumFilters(), NumFilters(),
                                     "policy_conv1", 3, true);
    
    std::string conv2 = MakeConvLayer(conv1, head.policy, NumFilters(), 80,
                                     "policy_conv2", 3, false);
    
    // Reshape
    code << "policy_flat = layers.Reshape((80 * 8 * 8,), name='policy_flat')(" 
         << conv2 << ")";
    py_.AppendPython(code.str());
    
    // Create policy mapping
    std::vector<int> policy_map(1858);
    for (const auto& mapping : kConvPolicyMap) {
      if (mapping == -1) continue;
      const auto index = &mapping - kConvPolicyMap;
      const auto displacement = index / 64;
      const auto square = index % 64;
      const auto row = square / 8;
      const auto col = square % 8;
      policy_map[mapping] = ((row * 8) + col) * 80 + displacement;
    }
    
    std::string map_var = "policy_map";
    WeightsToNumpyArray(map_var, std::vector<float>(policy_map.begin(), policy_map.end()),
                       {1858});
    
    code.str("");
    code << "policy_map_int = ops.cast(ops.convert_to_tensor(" << map_var << "), dtype='int32')";
    py_.AppendPython(code.str());
    
    // Reshape to (batch*64, conv_size*8*8) before gather (like ONNX does)
    // Use FlattenBatchSpatial for flattening
    code.str("");
    code << "policy_flat_reshaped = FlattenBatchSpatial("
         << (head.policy.biases.size() * 8 * 8) << ", name='policy_flat_reshaped')(policy_flat)";
    py_.AppendPython(code.str());
    
    // Gather with axis=1 - use custom Gather layer
    code.str("");
    code << "policy_output = Gather(axis=1, name='policy_gather')([policy_flat_reshaped, policy_map_int])";
    py_.AppendPython(code.str());
    
  } else {
    // Classical policy head
    int policy_conv_size = head.policy.biases.size();
    std::string conv = MakeConvLayer(input_var, head.policy, NumFilters(), 
                                     policy_conv_size, "policy_conv", 1, true);
    
    // Reshape
    code << "policy_flat = layers.Reshape((" << (policy_conv_size * 8 * 8) 
         << ",), name='policy_flat')(" << conv << ")";
    py_.AppendPython(code.str());
    
    // Dense layer
    std::string ip_pol_w_var = "policy_ip_w";
    WeightsToNumpyArray(ip_pol_w_var, head.ip_pol_w, 
                       {policy_conv_size * 8 * 8, 1858}, {1, 0});
    
    std::string ip_pol_b_var = "policy_ip_b";
    WeightsToNumpyArray(ip_pol_b_var, head.ip_pol_b, {1858});
    
    // Use output_policy as the layer name directly
    code.str("");
    code << "policy_dense_layer = layers.Dense(1858, name='output_policy', use_bias=True)";
    py_.AppendPython(code.str());
    
    code.str("");
    code << "policy_dense_layer.build([None, " << (policy_conv_size * 8 * 8) << "])";
    py_.AppendPython(code.str());
    
    code.str("");
    code << "policy_dense_layer.set_weights([" << ip_pol_w_var << ", " 
         << ip_pol_b_var << "])";
    py_.AppendPython(code.str());
    
    code.str("");
    code << "output_policy = policy_dense_layer(policy_flat)";
    py_.AppendPython(code.str());
  }
  
  output_vars_.push_back("output_policy");
  return "output_policy";
}

std::string KerasConverter::MakeValueHead(const MultiHeadWeights& weights,
                                         const std::string& input_var) {
  if (weights.value_heads.count(options_.value_head) == 0) {
    throw Exception("The value head you specified '" + options_.value_head +
                    "' does not exist in this net.");
  }
  
  const MultiHeadWeights::ValueHead& head =
      weights.value_heads.at(options_.value_head);
  
  if (head.ip1_val_b.empty()) {
    throw Exception("The value head selected '" + options_.value_head + "'"
                    " is empty.");
  }
  
  std::ostringstream code;
  std::string value_flow_var;
  
  const int val_channels = NumEncBlocks() > 0 ? head.ip_val_b.size() : 32;
  
  if (NumEncBlocks() > 0) {
    // Encoder-based value head
    int embedding_size = weights.ip_emb_b.size();
    
    std::string ip_val_w_var = "value_ip_val_w";
    WeightsToNumpyArray(ip_val_w_var, head.ip_val_w,
                       {embedding_size, val_channels}, {1, 0});
    
    std::string ip_val_b_var = "value_ip_val_b";
    WeightsToNumpyArray(ip_val_b_var, head.ip_val_b, {val_channels});
    
    code << "value_embed_layer = layers.Dense(" << val_channels
         << ", name='value_embed', use_bias=True)";
    py_.AppendPython(code.str());
    
    code.str("");
    code << "value_embed_layer.build([None, " << embedding_size << "])";
    py_.AppendPython(code.str());
    
    code.str("");
    code << "value_embed_layer.set_weights([" << ip_val_w_var << ", " 
         << ip_val_b_var << "])";
    py_.AppendPython(code.str());
    
    code.str("");
    code << "value_embed = value_embed_layer(" << input_var << ")";
    py_.AppendPython(code.str());
    
    std::string value_embed_act = MakeActivation("value_embed", "value_embed_act", 
                                                 default_activation_);
    
    // Reshape from (batch*64, val_channels) to (batch, 64, val_channels) then to (batch, val_channels*8*8)
    // Use DynamicReshape for unflattening with dynamic batch
    code.str("");
    code << "value_reshape = DynamicReshape(target_shape=(-1, 64, " << val_channels 
         << "), output_shape_tuple=(None, 64, " << val_channels 
         << "), name='value_reshape_3d')(" << value_embed_act << ")";
    py_.AppendPython(code.str());
    
    code.str("");
    code << "value_flat = layers.Reshape((" << (val_channels * 8 * 8) 
         << ",), name='value_flat')(value_reshape)";
    py_.AppendPython(code.str());
    
    // First dense layer (matches ONNX converter line 1021-1026)
    std::string ip1_val_w_var = "value_ip1_w";
    WeightsToNumpyArray(ip1_val_w_var, head.ip1_val_w,
                       {val_channels * 8 * 8, 128}, {1, 0});
    
    std::string ip1_val_b_var = "value_ip1_b";
    WeightsToNumpyArray(ip1_val_b_var, head.ip1_val_b, {128});
    
    code.str("");
    code << "value_dense1_layer = layers.Dense(128, name='value_dense1', use_bias=True)";
    py_.AppendPython(code.str());
    
    code.str("");
    code << "value_dense1_layer.build([None, " << (val_channels * 8 * 8) << "])";
    py_.AppendPython(code.str());
    
    code.str("");
    code << "value_dense1_layer.set_weights([" << ip1_val_w_var << ", " 
         << ip1_val_b_var << "])";
    py_.AppendPython(code.str());
    
    code.str("");
    code << "value_dense1 = value_dense1_layer(value_flat)";
    py_.AppendPython(code.str());
    
    value_flow_var = MakeActivation("value_dense1", "value_dense1_act", 
                                   default_activation_);
    
  } else {
    // Classical value head
    std::string conv = MakeConvLayer(input_var, head.value, NumFilters(), 
                                     val_channels, "value_conv", 1, true);
    
    // Reshape
    code << "value_flat = layers.Reshape((" << (val_channels * 8 * 8) 
         << ",), name='value_flat')(" << conv << ")";
    py_.AppendPython(code.str());
    
    // First dense layer
    std::string ip1_val_w_var = "value_ip1_w";
    WeightsToNumpyArray(ip1_val_w_var, head.ip1_val_w,
                       {val_channels * 8 * 8, 128}, {1, 0});
    
    std::string ip1_val_b_var = "value_ip1_b";
    WeightsToNumpyArray(ip1_val_b_var, head.ip1_val_b, {128});
    
    code.str("");
    code << "value_dense1_layer = layers.Dense(128, name='value_dense1', use_bias=True)";
    py_.AppendPython(code.str());
    
    code.str("");
    code << "value_dense1_layer.build([None, " << (val_channels * 8 * 8) << "])";
    py_.AppendPython(code.str());
    
    code.str("");
    code << "value_dense1_layer.set_weights([" << ip1_val_w_var << ", " 
         << ip1_val_b_var << "])";
    py_.AppendPython(code.str());
    
    code.str("");
    code << "value_dense1 = value_dense1_layer(value_flat)";
    py_.AppendPython(code.str());
    
    value_flow_var = MakeActivation("value_dense1", "value_dense1_act", 
                                   default_activation_);
  }
  
  // Final dense layer
  const bool wdl = src_.format().network_format().value() ==
                   pblczero::NetworkFormat::VALUE_WDL;
  
  int output_size = wdl ? 3 : 1;
  std::string ip2_val_w_var = "value_ip2_w";
  WeightsToNumpyArray(ip2_val_w_var, head.ip2_val_w, {128, output_size}, {1, 0});
  
  std::string ip2_val_b_var = "value_ip2_b";
  WeightsToNumpyArray(ip2_val_b_var, head.ip2_val_b, {output_size});
  
  code.str("");
  code << "value_dense2_layer = layers.Dense(" << output_size
       << ", name='value_dense2', use_bias=True)";
  py_.AppendPython(code.str());
  
  code.str("");
  code << "value_dense2_layer.build([None, 128])";
  py_.AppendPython(code.str());
  
  code.str("");
  code << "value_dense2_layer.set_weights([" << ip2_val_w_var << ", " 
       << ip2_val_b_var << "])";
  py_.AppendPython(code.str());
  
  code.str("");
  code << "value_output = value_dense2_layer(" << value_flow_var << ")";
  py_.AppendPython(code.str());
  
  // Use output_wdl as the final layer name directly
  if (wdl) {
    code.str("");
    code << "output_wdl = layers.Softmax(name='output_wdl')(" 
         << "value_output)";
    py_.AppendPython(code.str());
  } else {
    code.str("");
    code << "output_wdl = layers.Activation('tanh', name='output_wdl')(" 
         << "value_output)";
    py_.AppendPython(code.str());
  }
  
  output_vars_.push_back("output_wdl");
  return "output_wdl";
}

std::string KerasConverter::MakeMovesLeftHead(const MultiHeadWeights& weights,
                                             const std::string& input_var) {
  if (src_.format().network_format().moves_left() !=
      pblczero::NetworkFormat::MOVES_LEFT_V1) {
    return "";
  }
  
  std::ostringstream code;
  
  // For attention-based networks, mlh_channels comes from ip_mov_b, not moves_left.biases
  const int mlh_channels = NumEncBlocks() > 0
                               ? weights.ip_mov_b.size()
                               : weights.moves_left.biases.size();
  const int mlh_fc1_outputs = weights.ip1_mov_b.size();
  
  std::string flow_var;
  if (NumEncBlocks() > 0) {
    // Attention-based: use embedding layer
    int embedding_size = weights.ip_emb_b.size();
    
    std::string ip_mov_w_var = "mlh_ip_mov_w";
    WeightsToNumpyArray(ip_mov_w_var, weights.ip_mov_w,
                       {embedding_size, mlh_channels}, {1, 0});
    
    std::string ip_mov_b_var = "mlh_ip_mov_b";
    WeightsToNumpyArray(ip_mov_b_var, weights.ip_mov_b, {mlh_channels});
    
    code << "mlh_embed_layer = layers.Dense(" << mlh_channels
         << ", name='mlh_embed', use_bias=True)";
    py_.AppendPython(code.str());
    
    code.str("");
    code << "mlh_embed_layer.build([None, " << embedding_size << "])";
    py_.AppendPython(code.str());
    
    code.str("");
    code << "mlh_embed_layer.set_weights([" << ip_mov_w_var << ", " 
         << ip_mov_b_var << "])";
    py_.AppendPython(code.str());
    
    code.str("");
    code << "mlh_embed = mlh_embed_layer(" << input_var << ")";
    py_.AppendPython(code.str());
    
    std::string mlh_embed_act = MakeActivation("mlh_embed", "mlh_embed_act", 
                                                default_activation_);
    
    // Reshape directly from (batch*64, mlh_channels) to (batch, mlh_channels*8*8)
    // Match ONNX converter which does a direct reshape
    code.str("");
    code << "mlh_flat = DynamicReshape(target_shape=(-1, " << (mlh_channels * 8 * 8) 
         << "), output_shape_tuple=(None, " << (mlh_channels * 8 * 8) 
         << "), name='mlh_flat')(" << mlh_embed_act << ")";
    py_.AppendPython(code.str());
    
    flow_var = "mlh_flat";
  } else {
    // Classical: use convolution
    std::string conv = MakeConvLayer(input_var, weights.moves_left, NumFilters(),
                                     mlh_channels, "mlh_conv", 1, true);
    
    // Reshape
    code.str("");
    code << "mlh_flat = layers.Reshape((" << (mlh_channels * 8 * 8) 
         << ",), name='mlh_flat')(" << conv << ")";
    py_.AppendPython(code.str());
    
    flow_var = "mlh_flat";
  }
  
  // First dense
  std::string ip1_mov_w_var = "mlh_ip1_w";
  WeightsToNumpyArray(ip1_mov_w_var, weights.ip1_mov_w,
                     {mlh_channels * 8 * 8, mlh_fc1_outputs}, {1, 0});
  
  std::string ip1_mov_b_var = "mlh_ip1_b";
  WeightsToNumpyArray(ip1_mov_b_var, weights.ip1_mov_b, {mlh_fc1_outputs});
  
  code.str("");
  code << "mlh_dense1_layer = layers.Dense(" << mlh_fc1_outputs
       << ", name='mlh_dense1', use_bias=True)";
  py_.AppendPython(code.str());
  
  code.str("");
  code << "mlh_dense1_layer.build([None, " << (mlh_channels * 8 * 8) << "])";
  py_.AppendPython(code.str());
  
  code.str("");
  code << "mlh_dense1_layer.set_weights([" << ip1_mov_w_var << ", " 
       << ip1_mov_b_var << "])";
  py_.AppendPython(code.str());
  
  code.str("");
  code << "mlh_dense1 = mlh_dense1_layer(mlh_flat)";
  py_.AppendPython(code.str());
  
  std::string mlh_flow_var = MakeActivation("mlh_dense1", "mlh_dense1_act", 
                                           default_activation_);
  
  // Final dense
  std::string ip2_mov_w_var = "mlh_ip2_w";
  WeightsToNumpyArray(ip2_mov_w_var, weights.ip2_mov_w, {mlh_fc1_outputs, 1}, {1, 0});
  
  std::string ip2_mov_b_var = "mlh_ip2_b";
  WeightsToNumpyArray(ip2_mov_b_var, weights.ip2_mov_b, {1});
  
  code.str("");
  code << "mlh_dense2_layer = layers.Dense(1, name='mlh_dense2', use_bias=True)";
  py_.AppendPython(code.str());
  
  code.str("");
  code << "mlh_dense2_layer.build([None, " << mlh_fc1_outputs << "])";
  py_.AppendPython(code.str());
  
  code.str("");
  code << "mlh_dense2_layer.set_weights([" << ip2_mov_w_var << ", " 
       << ip2_mov_b_var << "])";
  py_.AppendPython(code.str());
  
  code.str("");
  code << "mlh_output = mlh_dense2_layer(" << mlh_flow_var << ")";
  py_.AppendPython(code.str());
  
  // Use MakeActivation to apply the default activation (Mish, ReLU, etc.)
  // This matches the ONNX converter which uses MakeActivation for the final layer
  std::string output_mlh = MakeActivation("mlh_output", "output_mlh", default_activation_);
  
  output_vars_.push_back("output_mlh");
  return "output_mlh";
}

void KerasConverter::ConvertToKeras(const std::string& output_path) {
  MultiHeadWeights weights(src_.weights());
  
  // Build input layer (already done in constructor)
  // Keep input in NCHW format - transpose will happen in MakeAttentionBody if needed
  current_flow_var_ = options_.input_name;
  
  // Input convolution
  if (NumResBlocks() > 0) {
    current_flow_var_ = MakeConvLayer(
        current_flow_var_,
        weights.input,
        kInputPlanes,
        NumFilters(),
        "input_conv",
        3,
        true);
  }
  
  // Residual tower
  for (size_t i = 0; i < NumResBlocks(); ++i) {
    current_flow_var_ = MakeResidualBlock(
        current_flow_var_,
        weights.residual[i],
        "block_" + std::to_string(i));
  }
  
  // Attention/Encoder blocks
  if (NumEncBlocks() > 0) {
    current_flow_var_ = MakeAttentionBody(current_flow_var_, weights);
  }
  
  // Policy head
  std::string policy_out = MakePolicyHead(weights, current_flow_var_);
  
  // Value head
  std::string value_out = MakeValueHead(weights, current_flow_var_);
  
  // Moves left head (may be empty if not present)
  std::string mlh_out = MakeMovesLeftHead(weights, current_flow_var_);
  
  // Create Keras Model with all outputs
  std::ostringstream code;
  code << "outputs = [";
  bool first = true;
  for (const auto& out : output_vars_) {
    if (!first) code << ", ";
    code << out;
    first = false;
  }
  code << "]";
  py_.AppendPython(code.str());
  
  // Set model dtype if not float32
  std::string dtype_str = "float32";
  if (options_.data_type == WeightsToKerasConverterOptions::DataType::kFloat16) {
    dtype_str = "float16";
  } else if (options_.data_type == WeightsToKerasConverterOptions::DataType::kBFloat16) {
    dtype_str = "bfloat16";
  }
  
  code.str("");
  code << "model = keras.Model(inputs=" << options_.input_name 
       << ", outputs=outputs, name='lc0_model')";
  py_.AppendPython(code.str());
  
  // Convert model dtype if needed by casting layer weights with numpy
  if (options_.data_type != WeightsToKerasConverterOptions::DataType::kFloat32) {
    code.str("");
    code << "model = keras.models.clone_model(model)";
    py_.AppendPython(code.str());

    code.str("");
    code << "for layer in model.layers:";
    py_.AppendPython(code.str());
    code.str("");
    code << "    weights = layer.get_weights()";
    py_.AppendPython(code.str());
    code.str("");
    code << "    if weights:";
    py_.AppendPython(code.str());
    code.str("");
    code << "        converted = [np.asarray(w, dtype=np." << dtype_str << ") for w in weights]";
    py_.AppendPython(code.str());
    code.str("");
    code << "        layer.set_weights(converted)";
    py_.AppendPython(code.str());
  }
  
  // Save model to .keras format (format is auto-detected from extension)
  code.str("");
  code << "model.save('" << output_path << "')";
  py_.AppendPython(code.str());

  // Write Python code to file if requested (for debugging) - before execution
  if (!options_.python_output_file.empty()) {
    std::ofstream py_file(options_.python_output_file);
    if (!py_file.is_open()) {
      throw Exception("Failed to open Python output file: " +
                      options_.python_output_file);
    }
    py_file << py_.GetCodeLog();
    py_file.close();
  }

  // Execute the entire accumulated script
  py_.ExecuteAll();
  
  // Clean up weights directory if it was auto-generated
  if (!options_.weights_dir.empty() && options_.cleanup_weights_dir) {
    try {
      std::filesystem::remove_all(options_.weights_dir);
    } catch (const std::exception& e) {
      std::cerr << "Warning: Failed to clean up weights directory: " << e.what() << std::endl;
    }
  }
}

}  // namespace

WeightsToKerasConverterOptions::DataType
WeightsToKerasConverterOptions::StringToDataType(const std::string& s) {
  if (s == "f32") return DataType::kFloat32;
  if (s == "f16") return DataType::kFloat16;
  if (s == "bf16") return DataType::kBFloat16;
  throw Exception("Invalid data type: [" + s +
                  "]. Only f32, f16 and bf16 are supported.");
}

void ConvertWeightsToKeras(const pblczero::Net& net,
                          const WeightsToKerasConverterOptions& options,
                          const std::string& output_path) {
  KerasConverter converter(net, options);
  converter.ConvertToKeras(output_path);
}

}  // namespace lczero