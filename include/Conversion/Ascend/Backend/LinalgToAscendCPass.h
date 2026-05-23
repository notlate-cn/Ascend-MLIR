/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it
 * under terms and conditions of the CANN Open Software License Agreement
 * Version 2.0 (the "License"). Please refer to LICENSE in the root of the
 * software repository for the full text of the License.
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
 * KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
 * NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the
 * License.
 */

#ifndef CONVERSION_LINALGTOASCENDC_PASS_H
#define CONVERSION_LINALGTOASCENDC_PASS_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace afir {

std::unique_ptr<Pass> createLinalgToAscendCPass();

} // namespace afir
} // namespace mlir

#endif // CONVERSION_LINALGTOASCENDC_PASS_H
