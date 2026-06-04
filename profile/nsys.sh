nsys profile -t cuda,nvtx -o nsys_profile -f true python ../example.py
nsys stats nsys_profile.nsys-rep