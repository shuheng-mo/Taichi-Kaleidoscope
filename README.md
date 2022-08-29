# About
Taichi + X ? What can we do with Taichi, Computer physics? Simulation? Mathematics? High-performance parallel computing? What about other area, is Taichi the ideal next-gen language for all industries? This is a innovative lab to explore the possibility of Taichi, let's see what the best we can do.  
```
# Install Taichi with 
$ python3 -m pip install taichi
```

- Topics included:

```bash
1. Numerical method
2. Computational Fluid Dynamics
```

- Tech included:
```bash
1. Taichi + Pytorch
2. Taichi + MPI
3. ...? What is your idea? 
```

- Trouble shooting & Issues (updating ...)
1. To ensure install mpi4py using conda correctly, use `conda create -n ENV_NAME -c conda-forge 'python=3.10.*' openmpi mpi4py`, you may specify your open-sourced MPI version as `mpich` or `openmpi` and Python version according to your needs. This should work for both `Windows` and `macOS`, for `Linux` please build from the source. Test the installation with: 

    ```bash
    conda activate YOUR_ENV_NAME
    mpiexec -n 2048 python -m mpi4py.bench helloworld
    # this should return hello from 2048 processes
    ```