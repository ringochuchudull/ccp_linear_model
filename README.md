# attn_ccp

```
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH="/mnt/c/Users/CHURI/Downloads/cpp/libtorch" -Dpybind11_DIR=""/home/ringocsw/anaconda3/envs/tsl/lib/python3.8/site-packages/pybind11/share/cmake/pybind11"" --target clean ../cpp 
make
```

```
import sys
sys.path.append('/mnt/c/Users/CHURI/Desktop/Rin90/attn_ccp/build')
import inference
```