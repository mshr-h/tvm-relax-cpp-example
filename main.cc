#include <iostream>
#include <tvm/runtime/relax_vm/executable.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/memory/memory_manager.h>
#include <tvm/runtime/data_type.h>

using tvm::runtime::Module;
using tvm::runtime::PackedFunc;
using tvm::runtime::memory::AllocatorType;
using tvm::runtime::relax_vm::VMExecutable;

int main()
{
  std::string path = "./compiled_artifact.so";

  // Load the shared object
  Module m = Module::LoadFromFile(path);
  std::cout << m << std::endl;

  PackedFunc vm_load_executable = m.GetFunction("vm_load_executable");
  CHECK(vm_load_executable != nullptr)
      << "Error: `vm_load_executable` does not exist in file `" << path << "`";

  // Create a VM from the Executable
  Module mod = vm_load_executable();
  PackedFunc vm_initialization = mod.GetFunction("vm_initialization");
  CHECK(vm_initialization != nullptr)
      << "Error: `vm_initialization` does not exist in file `" << path << "`";

  // Initialize the VM
  tvm::Device device{kDLCPU, 0};
  vm_initialization(static_cast<int>(device.device_type), static_cast<int>(device.device_id),
                    static_cast<int>(AllocatorType::kPooled), static_cast<int>(kDLCPU), 0,
                    static_cast<int>(AllocatorType::kPooled));

  PackedFunc main = mod.GetFunction("main");
  CHECK(main != nullptr)
      << "Error: Entry function does not exist in file `" << path << "`";

  // Create and initialize the input array
  auto i32 = tvm::runtime::DataType::Int(32);
  tvm::runtime::NDArray input = tvm::runtime::NDArray::Empty({3, 3}, i32, device);
  int numel = input.Shape()->Product();
  for (int i = 0; i < numel; ++i)
    static_cast<int *>(input->data)[i] = i;

  // Run the main function
  tvm::runtime::NDArray output = main(input);
  for (int i = 0; i < numel; ++i)
    std::cout << static_cast<int *>(output->data)[i] << std::endl;
}