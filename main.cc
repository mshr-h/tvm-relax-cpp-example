#include <iostream>
#include <tvm/ffi/function.h>
#include <tvm/runtime/vm/vm.h>

using tvm::ffi::Any;
using tvm::ffi::Function;
using tvm::runtime::Module;
using tvm::runtime::NDArray;
using tvm::runtime::memory::AllocatorType;

int main()
{
  std::string path = "./compiled_artifact.so";

  // Load the shared object
  Module m = Module::LoadFromFile(path);
  std::cout << m << std::endl;

  Function vm_load_executable = m.GetFunction("vm_load_executable");
  CHECK(vm_load_executable != nullptr)
      << "Error: `vm_load_executable` does not exist in file `" << path << "`";
  std::cout << "Found vm_load_executable()" << std::endl;

  // Create a VM from the Executable
  Module mod = vm_load_executable().cast<Module>();
  Function vm_initialization = mod.GetFunction("vm_initialization");
  CHECK(vm_initialization != nullptr)
      << "Error: `vm_initialization` does not exist in file `" << path << "`";
  std::cout << "Found vm_initialization()" << std::endl;

  // Initialize the VM
  tvm::Device device{kDLCPU, 0};
  vm_initialization(static_cast<int>(device.device_type), static_cast<int>(device.device_id),
                    static_cast<int>(AllocatorType::kPooled), static_cast<int>(kDLCPU), 0,
                    static_cast<int>(AllocatorType::kPooled));
  std::cout << "vm initialized" << std::endl;

  Function main = mod.GetFunction("main");
  CHECK(main != nullptr)
      << "Error: Entry function does not exist in file `" << path << "`";
  std::cout << "Found main()" << std::endl;

  // Create and initialize the input array
  auto i32 = tvm::runtime::DataType::Int(32);
  NDArray input = NDArray::Empty({3, 3}, i32, device);
  int numel = input.Shape()->Product();
  for (int i = 0; i < numel; ++i)
    static_cast<int *>(input->data)[i] = i;
  std::cout << "Input array initialized" << std::endl;

  // Run the main function
  NDArray output = main(input).cast<NDArray>();
  std::cout << "output: " << std::endl;
  for (int i = 0; i < numel; ++i)
    std::cout << "  " << static_cast<int *>(output->data)[i] << std::endl;
}