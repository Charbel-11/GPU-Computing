# GPU Computing

This is a collection of algorithms written in CUDA and to be run on GPU-enabled computers. The aim is to improve the runtime of their typical sequential implementations.

## Problems
The following problems were tackled:
* Breadth-First Search (BFS)
* Convolution computation
* Histogram finding
* Merge algorithm
* Reduction algorithm
* Scan algorithms (inclusive and exclusive)
* Sort algorithms 
  * Radix sort 
  * Merge sort
* Sparse Matrix-Vector Multiplication where sparse matrices are represented using either COO, CSR or ELL formats
* Vector Operations

## Results
The following results were achieved when running the algorithms on Google Colab. The GPU kernel time only considers the time it took for the GPU to compute the answer. The total GPU time also considers the time needed to copy memory from/to the CPU.  
Note that the used CPU code is single-threaded and non-vectorized which might exaggerate the GPU speed-up.
![Screenshot (345)](https://user-images.githubusercontent.com/61922252/129423342-3cc9155f-ed8e-4895-99bf-a877bcf6e82d.png)
![Screenshot (346)](https://user-images.githubusercontent.com/61922252/129423396-375e2471-b836-49c6-80a6-17ec6adcaca5.png)

