#!/bin/bash

# See this stackoverflow question
# http://stackoverflow.com/questions/59895/getting-the-source-directory-of-a-bash-script-from-within
# for the magic in this command and see
# https://stackoverflow.com/questions/39340169/dir-cd-dirname-bash-source0-pwd-how-does-that-work
# for a further explaination
SETUP_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo $SETUP_DIR

export HEXSAMPLE_ROOT=$SETUP_DIR/../hexsample

echo "HEXSAMPLE_ROOT set to " $HEXSAMPLE_ROOT

#
# Base package root. All the other relevant folders are relative to this
# location.
#
export XRAYRECO_ROOT=$SETUP_DIR

#
# Add the external package and the root folder to the $PYTHONPATH so that we can
# effectively import the relevant modules.
#
export PYTHONPATH=$XRAYRECO_ROOT:$HEXSAMPLE_ROOT:$PYTHONPATH
echo "PYTHONPATH set to " $PYTHONPATH
