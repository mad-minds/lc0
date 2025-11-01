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

#pragma once

#include <string>
#include "proto/net.pb.h"

namespace lczero {

// Options to use when converting weights to Keras format.
struct WeightsToKerasConverterOptions {
  enum class DataType { kFloat32, kFloat16, kBFloat16 };

  DataType data_type = DataType::kFloat32;
  int batch_size = -1;  // -1 for dynamic batch size
  std::string input_name = "input_planes";
  std::string policy_output_name = "policy";
  std::string value_output_name = "value";
  std::string wdl_output_name = "wdl";
  std::string mlh_output_name = "moves_left";
  std::string policy_head = "vanilla";
  std::string value_head = "winner";
  std::string python_output_file = "";  // Optional: path to output the generated Python code
  std::string weights_dir = "";  // Optional: directory to save weight .npy files (for cleaner Python code)
  bool cleanup_weights_dir = true;  // Whether to clean up the weights directory after conversion

  static DataType StringToDataType(const std::string&);
};

// Converts weights file to Keras model file (.keras format).
void ConvertWeightsToKeras(const pblczero::Net& net,
                          const WeightsToKerasConverterOptions& options,
                          const std::string& output_path);

}  // namespace lczero

