# C++ Implementation of the model
Here, you will find the pure C++ implementation of the learning model represented at `../python/question1v1.py`.
Just make sure that you have already run the python script, successfully trained the model on CIFAR-10, exported the weights, 
the inputs, and the outputs of the model.
## Binaries
This `CMake` project, produces these executable binaries:
- `Inference`: The main executable to run the model in the inference mode. Use `-h` to see the possible arguments.

## Build
To build it, make sure you have C++14 capable compiler. In CentOS7, you can use this command to enable it:
```
scl enable devtoolset-7 bash
```
Then, do as follows:
```
mkdir build
cd build
cmake ..
make -j8
```

## Does it work correctly?
Use the script at `../compare/compare.py` to compare the results of the C++ implementation stored at `../dumps/inference_outputs_cpp/` 
with the results of the Tensorflow implementation stored at `../dumps/inference_outputs/`. 
Just make sure to run `Inference` executable with `-k` argument to enable the tensor dumps.
