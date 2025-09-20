#include <iostream>
#include <optional>
#include <tvm/ffi/c_api.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/object.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/container/shape.h>
#include <tvm/runtime/vm/vm.h>

struct CPUNDAlloc
{
  void AllocData(DLTensor *tensor)
  {
    tensor->data = malloc(tvm::ffi::GetDataSize(*tensor));
  }
  void FreeData(DLTensor *tensor) { free(tensor->data); }
};

int main()
{
  std::string path = "./compiled_artifact.so";

  // Load the shared object
  tvm::ffi::Module m = tvm::ffi::Module::LoadFromFile(path);

  tvm::ffi::Optional<tvm::ffi::Function> vm_load_executable = m->GetFunction("vm_load_executable");
  CHECK(vm_load_executable != nullptr)
      << "Error: `vm_load_executable` does not exist in file `" << path << "`";
  std::cout << "Found vm_load_executable()" << std::endl;

  // Create a VM from the Executable
  tvm::ffi::Module mod = (*vm_load_executable)().cast<tvm::ffi::Module>();
  tvm::ffi::Optional<tvm::ffi::Function> vm_initialization = mod->GetFunction("vm_initialization");
  if (!vm_initialization.has_value()) {
    LOG(FATAL) << "Error: `vm_initialization` does not exist in file `" << path << "`";
  }
  std::cout << "Found vm_initialization()" << std::endl;

  // Initialize the VM
  tvm::Device device{kDLCPU, 0};
  (*vm_initialization)(static_cast<int>(device.device_type), static_cast<int>(device.device_id),
                       static_cast<int>(tvm::runtime::memory::AllocatorType::kPooled), static_cast<int>(kDLCPU), 0,
                       static_cast<int>(tvm::runtime::memory::AllocatorType::kPooled));
  std::cout << "vm initialized" << std::endl;

  tvm::ffi::Optional<tvm::ffi::Function> main = mod->GetFunction("main");
  CHECK(main != nullptr)
      << "Error: Entry function does not exist in file `" << path << "`";
  std::cout << "Found main()" << std::endl;

  // Create and initialize the input array
  tvm::ffi::Tensor input = tvm::ffi::Tensor::FromNDAlloc(CPUNDAlloc(), {3, 3}, {kDLInt, 32, 1}, device);
  int numel = input.shape()->Product();
  for (int i = 0; i < numel; ++i)
    static_cast<int *>(input->data)[i] = i;
  std::cout << "Input array initialized" << std::endl;

  // Run the main function
  tvm::ffi::Tensor output = (*main)(input).cast<tvm::ffi::Tensor>();
  std::cout << "output: " << std::endl;
  for (int i = 0; i < numel; ++i)
    std::cout << "  " << static_cast<int *>(output->data)[i] << std::endl;
}