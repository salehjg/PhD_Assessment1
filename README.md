# PhD Position Assessment
Directories:
- `python`: Contains the working version of the given Tensorflow script with Keras interface.
- `cpp`: Contains the pure C++ implementation of the Tensorflow script.
- `dumps`: Contains the tensor dumps to allow the third-party python script to compare the results of C++ and Tensorflow implementations. 
- `compare`: Contains the python scripts to compare the results of the two implementations stored at `dumps/`

## How to use it

### Step 0. Install The Dependencies
Install these dependencies for `python3`:
```
h5py
tensorflow
keras
numpy
C++ 14 capable compiler (G++ 7+)
```


Then clone the repository with the sub-modules with:
```
git clone <GIT REPO ADDRESS> --recursive
cd <GIT REPO DIR>
```

### Step 1. Train The Model
To train the model with CIFAR-10 dataset, do the following:
```
cd python
python3 question1v1.py -t
```


### Step 2. Export The Weights
Export the trained weights, the biases, the test-set, and the layer outputs with: 
```
python3 question1v1.py -e
```
Now, we are ready to run the C++ implementation.


### Step 3. Compile And Run The C++ Implementation
Use the following commands to build the C++ implementation:
``` 
cd <GIT REPO DIR>
cd cpp
mkdir build
cd build
scl enable devtoolset-7 bash #Only if you are using CentOS-7 
cmake ..
make -j8
./Inference -d <GIT REPO DIR>/dumps -k
```


### Step 4. Compare The Results 
To make sure that the C++ implementation is working as it should, use the third party Python script at `compare/` to 
compare the results of Tensorflow at `dumps/inference_outputs` with the results of C++ at `dumps/inference_outputs_cpp`. 
```
cd <GIT REPO DIR>
cd compare
python3 compare.py
```

The correct results must should look like this:
```
1.output.tensor.npy : 
dif sum:  -0.03274338
dif sum abs:  0.69267327

8.output.tensor.npy : 
dif sum:  9.123866e-08
dif sum abs:  1.7652825e-05

0.output.tensor.npy : 
dif sum:  0.02628721
dif sum abs:  1.71718

3.output.tensor.npy : 
dif sum:  -0.010693033
dif sum abs:  0.3406419

7.output.tensor.npy : 
dif sum:  -0.0006910674
dif sum abs:  0.009442929

4.output.tensor.npy : 
dif sum:  -0.0029424913
dif sum abs:  0.06448954

2.output.tensor.npy : 
dif sum:  -0.012329868
dif sum abs:  0.29902413

6.output.tensor.npy : 
dif sum:  -0.0016113836
dif sum abs:  0.034442667

5.output.tensor.npy : 
dif sum:  -0.0016113836
dif sum abs:  0.034442667
```


## Refs
These open-source repositories have been used to implement the project:
| Repo | Description | License | 
|-|-|-| 
| [cnpy](https://github.com/rogersce/cnpy) | C++ Library for working with `*.npy` files | MIT |
| [argparse](https://github.com/jamolnng/argparse) | C++ Library for handling arguments | Apache-2.0-with-LLVM-Exception or GPL-3.0 |
| [spdlog](https://github.com/gabime/spdlog) | C++ Library for fast logging | MIT | 

