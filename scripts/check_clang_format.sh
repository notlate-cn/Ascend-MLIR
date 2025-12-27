#!/bin/bash
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

set -e

CLANG_FORMAT=$(which clang-format) || (echo "Please install 'clang-format' tool first"; exit 1)

version=$("${CLANG_FORMAT}" --version | sed -n "s/.*\ \([0-9]*\)\.[0-9]*\.[0-9]*.*/\1/p")
if [[ "${version}" -lt "8" ]]; then
  echo "clang-format's version must be at least 8.0.0"
  exit 1
fi

CURRENT_PATH=$(pwd)
PROJECT_HOME=${PROJECT_HOME:-$(dirname "$0")/../}

echo "CURRENT_PATH=$CURRENT_PATH"
echo "PROJECT_HOME=$PROJECT_HOME"

# print usage message
function usage()
{
  echo "Check or format source files using clang-format"
  echo "Usage:"
  echo "bash $0 [-a] [-c] [-l] [-f] [-h]"
  echo "e.g. $0 -a"
  echo ""
  echo "Options:"
  echo "    -a Check code format of all files, default case"
  echo "    -c Check code format of the files changed compared to last commit"
  echo "    -l Check code format of the files changed in last commit"
  echo "    -f Format files instead of checking (use with -a, -c, or -l)"
  echo "    -h Print usage"
}

# check and set options
function checkopts()
{
  # init variable
  mode="all"    # default check all files
  format_only=0  # 0 for check, 1 for format

  # Process the options
  while getopts 'aclfh' opt
  do
    case "${opt}" in
      a)
        mode="all"
        ;;
      c)
        mode="changed"
        ;;
      l)
        mode="lastcommit"
        ;;
      f)
        format_only=1
        ;;
      h)
        usage
        exit 0
        ;;
      *)
        echo "Unknown option ${opt}!"
        usage
        exit 1
    esac
  done
}

# init variable
# check options
checkopts "$@"

# switch to project root path, which contains clang-format config file '.clang-format'
pushd "${CURRENT_PATH}"
    CHECK_LIST_FILE='__checked_files_list__'
    CHECK_RESULT_FILE='__code_format_check_result__'
    cd "${PROJECT_HOME}" || exit 1

    if [ "X${mode}" == "Xall" ]; then
      find . -type f \( -name "*.h" -o -name "*.hpp" -o -name "*.cpp" -o -name "*.cc" \) \
        ! -path "./externals/*" ! -path "./.git/*" ! -path "./build/*" ! -path "./cmake-build-debug/*" > "${CHECK_LIST_FILE}" || true
    elif [ "X${mode}" == "Xchanged" ]; then
      # --diff-filter=ACMRTUXB will ignore deleted files in commit
      git diff --diff-filter=ACMRTUXB --name-only | grep -v "^externals/" | grep "\.h$\|\.hpp$\|\.cpp$\|\.cc$" > "${CHECK_LIST_FILE}" || true
    else  # "X${mode}" == "Xlastcommit"
      git diff --diff-filter=ACMRTUXB --name-only HEAD~ HEAD | grep -v "^externals/" | grep "\.h$\|\.hpp$\|\.cpp$\|\.cc$" > "${CHECK_LIST_FILE}" || true
    fi

    if [ "${format_only}" -eq 1 ]; then
      # Format mode: directly format all files
      echo "Formatting files..."
      while read line; do
        echo "Formatting: ${line}"
        ${CLANG_FORMAT} -i "${line}"
      done < "${CHECK_LIST_FILE}"
      echo "Format complete!"
      rm -f "${CHECK_LIST_FILE}"
      exit 0
    fi

    echo "0" > "$CHECK_RESULT_FILE"

    # check format of files modified in the lastest commit 
    while read line; do
      BASE_NAME=$(basename "${line}")
      TEMP_FILE="__TEMP__${BASE_NAME}"
      cp "${line}" "${TEMP_FILE}"
      ${CLANG_FORMAT} -i "${TEMP_FILE}"
      set +e
      diff "${TEMP_FILE}" "${line}"
      ret=$?
      set -e
      rm -f "${TEMP_FILE}"
      if [[ "${ret}" -ne 0 ]]; then
        echo "File ${line} is not formated, please format it."
        echo "1" > "${CHECK_RESULT_FILE}"
        break
      fi
    done < "${CHECK_LIST_FILE}"

    result=$(cat "${CHECK_RESULT_FILE}")
    rm -f "${CHECK_RESULT_FILE}"
    rm -f "${CHECK_LIST_FILE}"
popd

if [[ "X${result}" == "X0" ]]; then
  echo "Check PASS: specified files are well formated!"
fi
exit "${result}"
