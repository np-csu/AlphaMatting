This is the reference implementation for the paper:

  Shared Sampling for Real-Time Alpha Matting
  Eduardo S. L. Gastal and Manuel M. Oliveira
  Computer Graphics Forum. Volume 29 (2010), Number 2.
  Proceedings of Eurographics 2010, pp. 575-584.

Usage of this demo is free for research purposes only. This software
is provided "as is" without any expressed or implied warranties of any
kind. The user also agrees that the authors shall not be liable for
any loss or damage of any sort incurred as the result of using this
software.

Refer to the publication above if you use this demo. For an up-to-date
version go to:

           http://inf.ufrgs.br/~eslgastal/SharedMatting/


=== Introduction
================================================================================

This package contains an executable for the real-time Shared Matting technique
(SharedMatting) which also implements the matte optimization step described in
our paper [1]. Furthermore, we also provide a MATLAB implementation for this
optimization step (SharedMatting.m).

This demo should work on any Linux 64-bit system which includes an NVIDIA GPU
with compute capability 1.0 or higher. We have included in this package the
NVIDIA CUDA Runtime library (libcudart.so.3, version 3.2.16); thus, installing
the CUDA toolkit is not required.  However, if you are having trouble running
this demo, here are the specifications of the system where it was compiled:

  Ubuntu 10.04 64-bit.
  NVIDIA GPU supporting CUDA compute capability 1.0 or higher.
   - Tested with GTX 280 and 8800 GTS.
  NVIDIA cuda driver version 260.19.26 for Linux 64-bit, available from [4].
  CUDA Toolkit version 3.2.16 for Linux 64-bit, available from [5].
  Qt version 4 (modules Core, Gui and OpenGL) with its dependencies.
  Boost program options version 1.40.

To install the required libraries under Ubuntu, use the following command:

  sudo apt-get install libqt4-dev libboost-program-options1.40.0

The MATLAB code requires an implementation of the Matting Laplacian matrix [2],
where the original code by Levin et al. can be found at [3].

Comparison of our Shared Matting technique against the state of the art is
available at the matting evaluation website:

                   http://www.alphamatting.com/

A dataset containing some images with corresponding trimaps and ground-truth
mattes is also available at the above website. We have included in this packaged
one of the images from this dataset.


=== Usage
================================================================================

--- Real-Time Shared Matting
--------------------------------------------------------------------------------

  1. Make the demo executable using the command "chmod +x SharedMatting",
     without quotes.

  2. Run the demo without any command line options using the command
     "./SharedMatting", without quotes.

  3. A window will appear asking for an input image. Select the provided
     "GT04.png".

  4. A window will appear asking for an input trimap. Select the provided
     "GT04_trimap.png".

  5. A simple GUI will be presented alongside a window showing the selected
     image. This GUI allows control over many parameters, described below.

  == Matting Parameters:
    - Ki: The maximum imagespace distance used for expanding known regions (in
          pixels). Default value: 10.

    - Kc: The maximum colorspace distance used for expanding known regions (in RGB
          units). Default value: 5.

    - Search steps: Maximum number of steps used when searching for known samples.
                    Default value: 300.

  Refer to our paper [1] for details on how each of these parameters are used
  when computing the alpha matte. 

  == Display Options:
    - Image:     Shows the original input image. Shortcut: i.
    - Alpha:     Shows the extracted alpha matte. Shortcut: a.
    - Composite: Shows the input image composited over a new background, after
                 matte extraction. Shortcut: c.
    - Trimap:    Shows the trimap after known regions expansion. Shortcut: t.
    - FG Color:  Shows the estimated foreground color for each pixel, used to
                 compute the matte. Shortcut: f.
    - BG Color:  Shows the estimated background color for each pixel, used to
                 compute the matte. Shortcut: b.

  == Additional Options:
    - Continuously compute the matte:
      This option is provided for benchmark purposes only. When enabled, the
    system computes the alpha matte nonstop, while displaying an FPS (frames per
    second) counter on the window title.  Please note that continuously
    computing the matte will use a lot of CPU and GPU power, and might make your
    computer unresponsive.

    - Run matte optimization:
      When this button is clicked, the alpha matte quality is improved by an
    optimization process described in our paper [1]. This optimization is CUDA
    enabled and runs on the GPU, so it should be really fast.
      When the optimization button is clicked, if the selected display option is
    not "Alpha" or "Composite", the "Alpha" option will be selected automatically.
    When the optimized matte or optimized composite is being shown in the image
    window, a green checkmark will be displayed in the optimization button, and
    the image window title will include the suffix "(optimized)".  Changing any
    GUI parameter after the optimization will require a new optimization by
    re-clicking the button.
      If a ground-truth alpha matte is provided in the command line using the
    "-g" option (see below), the matte RMSE (root mean square error) will be
    recalculated for the optimized matte and displayed under the "Error
    Information" section.
      Finally, note that only the alpha values are optimized, while the
    estimated foreground and background are left as-is (i.e., the values
    computed in real-time are used). A latter version of this demo might include
    an optimization of the foreground and background colors as well, which might
    produce better results for compositing.


  The available command line options are:

    Input options:
      -i [ --image ] arg          Input image file path.
      -t [ --trimap ] arg         Input trimap file path.
      -g [ --ground-truth ] arg   Input ground truth file path.
      -b [ --new-background ] arg New background file path.

    Matting options:
      --ki arg (=10)            Maximum imagespace distance for expanding known 
                                regions (in pixels).
      --kc arg (=5)             Maximum colorspace distance for expanding known 
                                regions (in RGB units).
      -n [ --steps ] arg (=300) Maximum number of steps used when searching for 
                                samples.

    Other options:
      --help                        Print this help text and exit.
      -p [ --pack ]                 Save pack(alpha, confidence, trimap) to "shared
                                    _matting_pack.bmp" and exit.
      -a [ --save-alpha ]           Save alpha to "shared_matting_alpha.bmp" and 
                                    exit.
      -x [ --save-composite ]       Save composite to "shared_matting_composite.bmp
                                    " and exit.
      --disable-matte-optimization  Disables matte optimization. This prevents out 
                                    of memory errors for large images.

  One example of specifying the input images using the command line:

  ./SharedMatting -i GT04.png -t GT04_trimap.png -g GT04_gt.png -b moon.jpg

--- Matte Optimization in MATLAB
--------------------------------------------------------------------------------

  This package also includes a MATLAB implementation of the matte optimization
  process.  To run the matte optimization using MATLAB, using the output of our
  real-time shared matting technique, follow these steps:

  1. Copy the file getLaplacian1.m from [3] to the same folder as
     SharedMatting.m (this only needs to be done once)

  2. Save the Shared Matting output to "shared_matting_pack.bmp" with the
     following command:
       SharedMatting --pack -i GT04.png -t GT04_trimap.png

  3. Open MATLAB, go to the SharedMatting folder, and run the following commands:
       % Read inputs
       image = imread('GT04.png');
       pack  = imread('shared_matting_pack.bmp');

       % Show alpha generated by real-time Shared Matting
       figure, imshow(pack(:,:,1));
       title('Shared Matting (Real-Time)');

       % Run optimization
       new_alpha = SharedMatting(image, pack);

       % Show alpha after optimization
       figure, imshow(new_alpha);
       title('Shared Matting (Optimized)');

=== References
================================================================================

[1] Eduardo S. L. Gastal and Manuel M. Oliveira.  "Shared Sampling for Real-Time
    Alpha Matting".  Computer Graphics Forum. Volume 29 (2010), Number 2.
    Proceedings of Eurographics 2010, pp. 575-584.
[2] Anat Levin, Dani Lischinski and Yair Weiss. "A Closed Form Solution to Natural
    Image Matting". IEEE Trans. Pattern Analysis and Machine Intelligence, Feb 2008.
[3] http://people.csail.mit.edu/alevin/matting.tar.gz
[4] http://developer.download.nvidia.com/compute/cuda/3_2_prod/drivers/devdriver_3.2_linux_64_260.19.26.run
[5] http://developer.download.nvidia.com/compute/cuda/3_2_prod/toolkit/cudatoolkit_3.2.16_linux_64_ubuntu10.04.run

=== Changelog
================================================================================

December 2010 - Version 1.0: 
  - Initial release.
