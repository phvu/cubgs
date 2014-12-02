cubgs
=====

CUDA implementation of Gaussian Mixture models for Background subtraction in video.

General information
--

[![Background subtraction in CUDA](http://img.youtube.com/vi/VYKTxS7boAQ/0.jpg)](http://www.youtube.com/watch?v=VYKTxS7boAQ)

Introduction
--

This is an implementation of Extended Gaussian mixture model (GMM) applying in Background subtraction for NVIDIA CUDA platform.
The original CPU idea and implementation can be found in Ref. 1 and 2.


Usage
--

<T.B.D>

Please see the sample code in the repository, read the article on 
[Codeproject](http://www.codeproject.com/KB/GPU-Programming/cubgs.aspx),
read the [paper](http://cubgs.googlecode.com/files/rivf10-cudaBgs.pdf), 
or the lengthy [technical report](http://cubgs.googlecode.com/files/report-cudaBgs.pdf).

A more verbose user guide might be added soon.

cubgs on Linux
--

~~Some of you might be interested in making cubgs work on Linux. 
[This](http://cubgs.googlecode.com/files/simple-cubgs.zip) is a package with Makefile for Linux, 
contributed by Apostolis Glenis.~~

*Update 22 December 2011*: [Apostolis](mailto:apostglen46@gmail.com) has joined this project and provided an 
almost fully working version of cubgs for Linux. You can download it from the 
[download page](http://code.google.com/p/cubgs/downloads/list). There is, however, a minor issue with this version, 
and we are still working on it.

License
--

The testing video clips and libraries (CUDA, OpenCV) belong to its authors.

The following conditions are derived from Zivkovic's original CPU implementation:

 This work may not be copied or reproduced in whole or in part for any commercial purpose. 
 Permission to copy in whole or in part without payment of fee is granted for nonprofit educational 
 and research purposes provided that all such whole or partial copies include the following: 
   * this notice,
   * an acknowledgment of the authors and individual contributions to the work; 

 Copying, reproduction, or republishing for any other purpose shall require a license. 
 Please contact the author in such cases.
 All the code is provided without any guarantee.

If you use this project in academic publications, please consider citing us:

[Vu Pham](http://phvu.net), [Phong Vo](http://www.fit.hcmus.edu.vn/~vdphong/), Hung Vu Thanh, Bac Le Hoai, 
    _"GPU Implementation of Extended Gaussian Mixture Model for Background Subtraction"_, 
    IEEE-RIVF 2010 International Conference on Computing and Telecommunication Technologies, 
    Vietnam National University, Nov. 01-04, 2010. 
    DOI: [10.1109/RIVF.2010.5634007](http://dx.doi.org/10.1109/RIVF.2010.5634007).

References
--
 1. [Z.Zivkovic](http://www.zoranz.net/), _"Improved adaptive Gausian mixture model for background subtraction"_, 
 International Conference Pattern Recognition (ICPR), Vol.2, pages: 28-31, 2004.
 1. [Z.Zivkovic](http://www.zoranz.net/), F. van der Heijden, _"Efficient adaptive density estimation 
 per image pixel for the task of background subtraction"_, Pattern Recognition Letters, vol. 27, no. 7, pages 773-780, 2006.

Last update: Dec. 2, 2014.
