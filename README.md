# sgp
Trying to enhance the undertext in the [SGP dataset](http://openn.library.upenn.edu/)

### Environment
Python 3.7.4 <br />
Pytorch 1.4.0
<br />


### Results preview (on cropped Tiff * rescale-0.25)
Images of 024r_029v
<br />
Original version, LDA version, NN-enhanced version, Conv-1d, Conv-2d, Conv-3d, 1x1-Conv, Conv-inception:
<br />
<img src='/figs/results_preview/scale25/024r_029v_orig_eval.png' width='100'>
<img src='/figs/results_preview/scale25/024r_029v_lda.png' width='100'>
<img src='/figs/results_preview/scale25/024r_029v_ae.png' width='100'>
<img src='/figs/results_preview/scale25/024r_029v_conv1d.png' width='100'>
<img src='/figs/results_preview/scale25/024r_029v_conv2d_eval.png' width='100'>
<img src='/figs/results_preview/scale25/024r_029v_conv3d_eval.png' width='100'>
<img src='/figs/results_preview/scale25/024r_029v_conv_patch.png' width='100'>
<img src='/figs/results_preview/scale25/024r_029v_conv_incep.png' width='100'>
<br />

Images of 102v_107r
<br />
Original version, LDA version, NN-enhanced version, Conv-1d, Conv-2d, Conv-3d, 1x1-Conv, Conv-inception:
<br />
<img src='/figs/results_preview/scale25/102v_107r_orig_eval.png' width='100'>
<img src='/figs/results_preview/scale25/102v_107r_lda.png' width='100'>
<img src='/figs/results_preview/scale25/102v_107r_ae.png' width='100'>
<img src='/figs/results_preview/scale25/102v_107r_conv1d.png' width='100'>
<img src='/figs/results_preview/scale25/102v_107r_conv2d_eval.png' width='100'>
<img src='/figs/results_preview/scale25/102v_107r_conv3d_eval.png' width='100'>
<img src='/figs/results_preview/scale25/102v_107r_conv_patch.png' width='100'>
<img src='/figs/results_preview/scale25/102v_107r_conv_incep.png' width='100'>
<br />

Images of 214v_221r
<br />
Original version, LDA version, NN-enhanced version, Conv-1d, Conv-2d, Conv-3d, 1x1-Conv, Conv-inception:
<br />
<img src='/figs/results_preview/scale25/214v_221r_orig_eval.png' width='100'>
<img src='/figs/results_preview/scale25/214v_221r_lda.png' width='100'>
<img src='/figs/results_preview/scale25/214v_221r_ae.png' width='100'>
<img src='/figs/results_preview/scale25/214v_221r_conv1d.png' width='100'>
<img src='/figs/results_preview/scale25/214v_221r_conv2d_eval.png' width='100'>
<img src='/figs/results_preview/scale25/214v_221r_conv3d_eval.png' width='100'>
<img src='/figs/results_preview/scale25/214v_221r_conv_patch.png' width='100'>
<img src='/figs/results_preview/scale25/214v_221r_conv_incep.png' width='100'>
<br />


### Network Reference
Denoise Autoencoder [[1]](#xing2015stacked)
<br />
<img src='/figs/networks/dae.png' width='360'>
<br />

Stacked Denoise Autoencoder [[1]](#xing2015stacked)
<br />
<img src='/figs/networks/sdae.png' width='900'>
<br />

Convolutional Neural Network (conv-1d) [[2]](#hu2015deep)
<br />
<img src='/figs/networks/conv1d.png' width='500'>
<br />

Convolutional Neural Network (conv-2d)
<br />
<img src='/figs/networks/conv2d.png' width='800'>
<br />

Convolutional Neural Network (1x1-conv) [[3]](#yu2017convolutional)
<br />
<img src='/figs/networks/1x1conv.png' width='450'>
<br />

Convnet + Inception (9 convolutional layers) [[4]](#lee2016contextual)
<br />
<img src='/figs/networks/conv_incep.png' width='450'>
<br />


### Reference
<a id="xing2015stacked">[1]</a> 
C. Xing, L. Ma, and X. Yang. Stacked denoise autoencoder based feature extraction and classification for hyperspectral images. Journal of Sensors, 2016:e3632943, 2015.

<a id="hu2015deep">[2]</a> 
Hu, W., Huang, Y., Wei, L., Zhang, F. and Li, H., 2015. Deep convolutional neural networks for hyperspectral image classification. Journal of Sensors, 2015.

<a id="yu2017convolutional">[3]</a> 
Yu, S., Jia, S. and Xu, C., 2017. Convolutional neural networks for hyperspectral image classification. Neurocomputing, 219, pp.88-98.

<a id="lee2016contextual">[4]</a> 
Lee, H. and Kwon, H., 2016, July. Contextual deep CNN based hyperspectral classification. In 2016 IEEE International Geoscience and Remote Sensing Symposium (IGARSS) (pp. 3322-3325). IEEE.
