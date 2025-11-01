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

#include <fstream>
#include <iostream>

#include "neural/keras/converter.h"
#include "neural/loader.h"
#include "tools/describenet.h"
#include "utils/files.h"
#include "utils/optionsparser.h"

namespace lczero {
namespace {

const OptionId kInputFilenameId{"input", "",
                                "Path of the input Lc0 weights file."};
const OptionId kOutputFilenameId{"output", "",
                                 "Path of the output Keras model file."};
const OptionId kKerasBatchSizeId{
    {.long_flag = "keras-batch-size",
     .uci_option = "",
     .help_text = "Batch size to use for Keras conversion (-1 for dynamic).",
     .visibility = OptionId::kProOnly}};
const OptionId kKerasDataTypeId{"keras-data-type", "",
                                "Data type to use in the Keras model (f32, f16, bf16)."};
const OptionId kValueHead{
    "value-head", "",
    "Value head to be used in the generated model. Typical values are "
    "'winner', 'q' or 'st', but only 'winner' is always available."};
const OptionId kPolicyHead{"policy-head", "",
                           "Policy head to be used in the generated model. "
                           "Typical values are 'vanilla', 'optimistic' or "
                           "'soft', but only 'vanilla' is always available."};
const OptionId kPythonOutputFileId{
    "python-output", "",
    "Optional path to output the generated Python code used to create the "
    "Keras model (useful for debugging)."};
const OptionId kWeightsDirId{
    "weights-dir", "",
    "Optional directory to save weight .npy files (for cleaner Python code). "
    "If not specified, defaults to a temporary directory based on output filename."};

bool ProcessParameters(OptionsParser* options) {
  options->Add<StringOption>(kInputFilenameId);
  options->Add<StringOption>(kOutputFilenameId);
  options->Add<IntOption>(kKerasBatchSizeId, -1, 2048) = -1;
  options->Add<ChoiceOption>(
      kKerasDataTypeId, std::vector<std::string>{"f32", "f16", "bf16"}) = "f32";
  options->Add<StringOption>(kValueHead) = "winner";
  options->Add<StringOption>(kPolicyHead) = "vanilla";
  options->Add<StringOption>(kPythonOutputFileId) = "";
  options->Add<StringOption>(kWeightsDirId) = "";
  if (!options->ProcessAllFlags()) return false;

  const OptionsDict& dict = options->GetOptionsDict();
  dict.EnsureExists<std::string>(kInputFilenameId);
  dict.EnsureExists<std::string>(kOutputFilenameId);
  return true;
}

}  // namespace

void ConvertLeelaToKeras() {
  OptionsParser options_parser;
  if (!ProcessParameters(&options_parser)) return;

  const OptionsDict& dict = options_parser.GetOptionsDict();
  auto weights_file =
      LoadWeightsFromFile(dict.Get<std::string>(kInputFilenameId));

  ShowNetworkFormatInfo(weights_file);
  if (!weights_file.has_weights()) {
    throw Exception("The network doesn't have weights.");
  }

  ShowNetworkWeightsInfo(weights_file);
  COUT << "Converting Leela network to Keras.";

  WeightsToKerasConverterOptions keras_options;
  keras_options.batch_size = dict.Get<int>(kKerasBatchSizeId);
  keras_options.data_type = WeightsToKerasConverterOptions::StringToDataType(
      dict.Get<std::string>(kKerasDataTypeId));
  keras_options.value_head = dict.Get<std::string>(kValueHead);
  keras_options.policy_head = dict.Get<std::string>(kPolicyHead);
  keras_options.python_output_file = dict.Get<std::string>(kPythonOutputFileId);
  
  // Default weights_dir to a directory based on output filename if not specified
  std::string weights_dir = dict.Get<std::string>(kWeightsDirId);
  if (weights_dir.empty()) {
    std::string output_file = dict.Get<std::string>(kOutputFilenameId);
    // Use output file directory + "_weights" suffix
    size_t last_slash = output_file.find_last_of("/\\");
    size_t last_dot = output_file.find_last_of(".");
    std::string base_name = (last_dot != std::string::npos && last_dot > last_slash) 
                            ? output_file.substr(0, last_dot) 
                            : output_file;
    weights_dir = base_name + "_weights";
  }
  keras_options.weights_dir = weights_dir;
  
  ConvertWeightsToKeras(weights_file, keras_options,
                       dict.Get<std::string>(kOutputFilenameId));

  COUT << "Done. Model saved to: " << dict.Get<std::string>(kOutputFilenameId);
  if (!keras_options.python_output_file.empty()) {
    COUT << "Python code saved to: " << keras_options.python_output_file;
  }
}

}  // namespace lczero

