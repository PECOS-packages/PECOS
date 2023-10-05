#!/bin/bash

#
# Copyright 2022 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
#

#
# Initial author: Tyson Lawrence
#

#
# This script needs to be sourced from within the
# base project directory
#


RAN_NAME=$( basename ${0#-} ) #- needed if sourced no path
SCRIPT_NAME=$( basename ${BASH_SOURCE} )

BUILD_DIR="./build"

USAGE_STR="Usage: source scripts/$SCRIPT_NAME"

if [[ $RAN_NAME == $SCRIPT_NAME ]]; then
  echo "Error: Script needs to be sourced, not ran directly"
  echo $USAGE_STR
  exit
fi

# Delete the build directory if it exists, so we start fresh
if [ -d "$BUILD_DIR" ]; then
  rm -r $BUILD_DIR
fi

mkdir $BUILD_DIR

cd $BUILD_DIR

# Just pass on all command line arguments to cmake
cmake .. -D CMAKE_EXPORT_COMPILE_COMMANDS=ON $@
RET=$?

# If we couldn't setup the build, return to the base
# project directory and remove the build directory
if [ $RET -ne 0 ]
then
  cd ..
  rm -r $BUILD_DIR
fi
