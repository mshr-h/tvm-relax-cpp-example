#!/bin/bash
set -e
# llvm setup https://apt.llvm.org/
#   sudo bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)"
#   sudo apt install -y libzstd-dev libpolly-17-dev

source_dir="tvm"
branch="main"
build_options="-DUSE_MLIR=ON -DUSE_CPP_RPC=ON"
use_llvm="ON"
clean_build_dir=0
if command -v uv &> /dev/null; then
  PIP_COMMAND="uv pip"
else
  PIP_COMMAND="python -m pip"
fi

while test $# -gt 0; do
  case "$1" in
    -h|--help)
      echo "options:"
      echo "-h, --help  show brief help"
      echo "--clean     cleanup build directory"
      echo "--cuda      enable CUDA support"
      echo "--msc       enable Multi-System Compiler support"
      echo "--papi      enable PAPI support"
      echo "--llvm      option for USE_LLVM"
      exit 0
      ;;
    --clean)
      shift
      clean_build_dir=1
      ;;
    --cuda)
      shift
      build_options+=" -DUSE_CUDA=ON -DUSE_TENSORRT_CODEGEN=ON"
      ;;
    --msc)
      shift
      build_options+=" -DUSE_MSC=ON"
      ;;
    --papi)
      shift
      build_options+=" -DUSE_PAPI=ON"
      ;;
    --llvm)
      shift
      use_llvm=$1
      ;;
    *)
      break
      ;;
  esac
done

# llvm config
build_options+=" -DUSE_LLVM=$use_llvm"

# clone repo or pull
if [ -d "$source_dir" ]; then
  cd $source_dir
  #git pull
  cd ..
else
  git clone --recursive --branch $branch https://github.com/apache/tvm $source_dir
fi

# build
cd $source_dir
if [ "$clean_build_dir" -ne 0 ]; then
  echo "cleanning build directory..."
  rm -rf build
fi
echo "Build Options: $build_options"
git submodule sync && git submodule update --init --recursive
cmake -S . -B build -G Ninja $build_options
cmake --build build

# install tvm-ffi
pushd 3rdparty/tvm-ffi
$PIP_COMMAND install -e .
popd

# install python package
$PIP_COMMAND install -e python --config-setting editable=compat
