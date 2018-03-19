
# signal_cross_correlation

This repo is for demonstration on how to use [pycu_interface](https://github.com/asuszko/pycu_interface) to access **GPU resource management, performance primitives, and custom CUDA kernel calls to accelerate Python code**. As the [pycu_interface](https://github.com/asuszko/pycu_interface) framework is flexible, this is just one of many ways a user can accelerate their Python code.

## Problem Description

This problem cross correlates a reference signal against a batch of received signals. [Cross correlation](https://en.wikipedia.org/wiki/Cross-correlation) is a measure of similarity between two signals. In this code, a reference signal is generated using [SciPy's chirp function](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.chirp.html). Any image array of return signals are generated with Gaussian noise added.

## Setup

To clone the repo and all subrepos with it, in your terminal or git console, run the following command:
> git clone --recursive https://github.com/asuszko/signal_cross_correlation.git

To compile the shared libraries needed, run the **setup.py** file found in the root folder of **pycu_interface**, with optional argument(s) -arch, and -cc_bin if on Windows. On Windows, the NVCC compiler looks for cl.exe to compile the C/C++ code. cl.exe comes with Visual Studio. On Linux, it uses the built in gcc compiler. An example of a command line run (on Windows) to compile the code is given below:
> python setup.py -arch=sm_50 -cc_bin="C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin"

After the libraries in pycu_interface are compiled, run the **setup.py** file in the root folder of **signal_cross_correlation*:
> python setup.py -arch=sm_50 -cc_bin="C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin"

On Linux, the commands would be the same, with the -cc_bin argument omitted. If you are unable to compile the libraries, you may [download the latest precompiled libraries here](https://github.com/asuszko/pycu_interface_libs).

## Testing

To verify the code is working, in the root folder:
> python run.py

![signal_cross_correlation result](https://lh3.googleusercontent.com/tLPTuFvWcqizc3w-hSiRkxjsWCOJauavWXHLn2lbnS3heECH4cmdTZ-PNJ5IFo3Uae-zKdRPlXMIgWQfURlB7X5T4CAg0pFm9_f4kMZkqzZW5VKVkEw42ocbg6Aq5_k4yiyJa0D66G-3dFv4BlA4lBd1tnAJy_U8ZDyIUlYIOEHao7ixisl3lxH1kK5L_6k5--AIQwur4LowH-IAc8RcAQ1oqjhE4iJJKjA39AspQ3-nz6l-5wSRj7AgAU_5mZ-_ru_8ku3JHhF88iN0wlrKx1i-sNhBkQuIs2_vc_ekFl_5musNRPIUSTX8G69D7n2I0yZAVQPXA5Zv-CerFeJbR4ESDhmvnuLrud5dQrTGWYGWFMN3uKTulFrmbBNLYEx3lhcfdZb-GtkW2Z7S7o5_6IwbJ3XTVP-0tGLa1DEmstG_Ky1icK2aOY5LNvl48ZZPxoYcN42gJp8XT5PBFJBwgj9rby7FkGp0vZabz3oRKlaKXn2Uuhur_YO1P7ynKOqIyFen6nnJpshfWCrme5tH5FsTT6_AXlWqG-enDVX2QOzC0GWwE5yOSoutkfwAKVJK03-oLBcdfBGbOqCHpvbqHjso6ueiHn4W4sICGVY=w640-h472-no)

## License
 
The MIT License (MIT)

Copyright (c) 2018 (Arthur Suszko (art.suszko@gmail.com) and contributors)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
