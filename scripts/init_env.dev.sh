#!/bin/zsh
# This script initialises a python virtual environment.
# It expects you to have Python installed at the system-level uses it for the virtual env.
# It also checks whether your system's OS and Architecture to install the appropriate dependant extras.
# All other extras and groups are installed (e.g. viz, dev)
# For any issues:
# + try to run this script from the project root level.

set -e

HELP="./$(basename $0) <virutal-env-name>"


if [[ $(dirname $0) != "./scripts" ]]; then
  echo "-- Please run this script from project root."
  exit 1
fi

if [[ -z $1 ]]; then
  echo "-- $HELP" >&2
  exit 1
fi

VENV_DIR=$(realpath "$1")
POETRY_FILE=$(realpath "pyproject.toml")
OS=$(uname -o)
ARCH=$(uname -m)

echo "PYPROJECT=$POETRY_FILE"
echo "OS=$OS"
echo "ARCH=$ARCH"
echo "PYTHON=$(which python3) version=$(python3 --version | awk '{print $2}')"
printf "Confirm? (y/n) "
read x && [[ $x != 'y' ]] && { echo "Exited"; exit 0 }

if [[ ! -f $POETRY_FILE ]]; then
  echo "-- Missing $POETRY_FILE." >&2
  exit 1
fi

echo "++ Virtual environment will be installed at $VENV_DIR"
if [[ -d "$VENV_DIR" ]]; then
  printf "-- Virtual environment $VENV_DIR already exists. Replace? (y/n) "
  read x && [[ $x != 'y' ]] && { echo "Exited."; exit 0 }
  rm -rf "$VENV_DIR"
fi

echo "++ Creating virtual env at $VENV_DIR..."
python3 -m venv $VENV_DIR

echo "++ Activating virtual env..."
source $VENV_DIR/bin/activate

echo "++ Installing poetry..."
pip install --upgrade pip
pip install poetry
echo "++ Installing dependencies via poetry..."

echo "++ Removing poetry.lock for fresh install..."
rm -f poetry.lock
echo "++ Installing dependencies..."
if [[ ${ARCH:u} == ARM* && ${OS:u} == "DARWIN" ]]; then   #:u - uppercase (zsh only)
  poetry install --with "dev"
else
  poetry install --with "dev"
fi

echo "++ Done. Your virtual env is installed at $VENV_DIR"
echo "To activate your virtual env run: source $VENV_DIR/bin/activate"
exit 0
