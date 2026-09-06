# ozaki-scheme-ii-sme

*Just for fun*: running Ozaki Scheme II on ARM SME.
Built and tested on an Apple M5 Pro.

This repository is an independent experiment and is not affiliated with the official Ozaki Scheme II implementation.

## Project Structure
```
/
├── playground/ozaki-ii.py    # Python implementation used to verify the algorithm
├── main.cpp                  # Entry point: runs a 256x256x256 matmul demo
├── matmul.cpp                # int8 matmul kernel on the SME unit (SMOPA)
├── matmul.hpp                # Kernel interface
├── ozaki-ii.cpp              # Ozaki Scheme II in C++
├── ozaki-ii.hpp              # Scheme interface
└── Makefile
```

## How to Play
```
make clean && make
make run
```

You will see a report in the console and two binary result files,
`c_ref.bin` and `c_ozaki.bin`

## Future Work
- Performance engineering: panel packing, multi-tile kernels, OpenMP, etc.

## References
- K. Ozaki, Y. Uchino, T. Imamura: [Ozaki Scheme II: ...](https://arxiv.org/abs/2504.08009) — the algorithm implemented here
- [ARM SME programming (ACLE intrinsics)](https://developer.arm.com/...) — the kernel's instruction set

## See Also
- [GEMMul8](https://github.com/RIKEN-RCCS/GEMMul8) — the official implementation maintained by the Ozaki Scheme II developers

