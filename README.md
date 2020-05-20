# sgp
Trying to enhance the undertext in the [SGP dataset](http://openn.library.upenn.edu/)

### Environment
Python 3.7.4 <br />
Pytorch 1.4.0
<br />


### Results preview (on cropped Tiff)
Original image of 024r_029v, NN-enhanced version, LDA version:
<br />
<img src='/figs/results_preview/024r_029v_orig.png' width='150'>
<img src='/figs/results_preview/024r_029v_enh.png' width='150'>
<img src='/figs/results_preview/024r_029v_lda.png' width='150'>
<br />

Original image of 102v_107r, NN-enhanced version, LDA version:
<br />
<img src='/figs/results_preview/102v_107r_orig.png' width='150'>
<img src='/figs/results_preview/102v_107r_enh.png' width='150'>
<img src='/figs/results_preview/102v_107r_lda.png' width='150'>
<br />

Original image of 214v_221r, NN-enhanced version, LDA version:
<br />
<img src='/figs/results_preview/214v_221r_orig.png' width='150'>
<img src='/figs/results_preview/214v_221r_enh.png' width='150'>
<img src='/figs/results_preview/214v_221r_lda.png' width='150'>
<br />


### Network Reference
Denoise Autoencoder [[1]](#xing2015stacked)
<br />
<img src='/figs/networks/dae.png' width='150'>
<br />

Stacked Denoise Autoencoder [[1]](#xing2015stacked)
<br />
<img src='/figs/networks/sdae.png' width='150'>
<br />


### Reference
<a id="xing2015stacked">[1]</a> 
C. Xing, L. Ma, and X. Yang. Stacked denoise autoencoder based feature extraction and classification for hyperspectral images. Journal of Sensors, 2016:e3632943, 2015.
