# UTPB-COSC-6389-Project3
This repo contains the assignment and provided code base for Project 3 of the graduate Computational Biomimicry class.

The goals of this project are:
1) Understand how convolutional neural networks are constructed and used, and the particulars of their implementation.

Description:
Using the code from your Project 2, create an extension which implements convolutions in your networks.  This time, your goal is to create an image classifier network, which accepts single rectangular images as input and outputs which of the object classes the network believes the image depicts.  You are allowed to select "toy" problems for this, such as the famous handwritten numerical digit dataset (example available here: https://www.kaggle.com/datasets/jcprogjava/handwritten-digits-dataset-not-in-mnist).

This site (https://www.kaggle.com/datasets) seems to be a database of datasets that you will likely find useful for both projects 2 and 3.

As with Projects 1 and 2, your application must generate the neural networks, display them on a canvas, and update them in real time as the weight values change.

You are not allowed to make use of any libraries related to neural networks in the code for this project.  The implementation of the network construction, operation, forward and backward propagation, training, and testing must all be your own.

Grading criteria:
1) If the code submitted via your pull request does not compile, the grade is zero.
2) If the code crashes due to forseeable unhandled exceptions, the grade is zero.
3) For full points, the code must correctly perform the relevant algorithms and display the network in real time, via the UI.

Deliverables:
A Python application which provides a correct implementation of a neural network generation and training system and is capable of training an image classifier which has good accuracy for the problem set selected.

Pooling
Forward:
foreach input neuron:
foreach input neuron:
get some input array with dimensions
in_M, inh


backward!
have some kernel (can be on a per-input basis) array with dimensions
kernel_w, kernel h where k_w and k h are equal t o in w and inch respectively
convolve the kernel over the input aray
the output dimensions are
out w = in_w - k_w+ 1,
outh- inh-kh + 1
i n t h i s case with in_w - k_ etc, our output dimensions are 1x1
we accumulate error a s a scalar float value i n the same way as a hiden neuron
29
301
311
foreach input neuron:
copy in_kernel
copy k e r n e l
create i n e r r with dimensions - in_kernel
fi l l i n ert with the values o f kernel * e r r o r
r e t r i e v e t h e o u t p u t e r r matrix from t h e i n p u t
i t s dimensions a r e equal t o t h e s i z e o f i t s kernel a n d it w i l l i n i t i a l l y we add the value in each cell of the i n err matrix t o the corresponding c e l l i n the matrix
contain a l l 0 . 0 v a l u e s
output_er
since each neuron t o which i t outputs w i l l d o t h e same thing, i t s t o t a l e r r o r w i l l a c c u m u l a t e
during t h e backprop operation

to modify the values o f kernel t o minimize error, w e perform a convop o f t h e error v a l u e o v e r
the original input matrix, multiplying t h e output o f each step o f t h e c o n v o p against the
learning r a t e and subtracting t h e result from t h e corresponding c e l l i n this kernel
works because the dimensions o f t h e kernel match t h e dimensions o f t h e input image, eror for a pooling neuron i s a s i n g l e s c a l a r v a l u e

 o n v neuron:
Forward:
Coreach input:
get some input aray with dimensions
in_v, in_h
have some kernel (can be on a per-input basis) aray w i t h dimensions
kernel_w, kernel h where kw and kh are equal t o i n w and in h respectively
convolve the kernel over the input array
t h e output dimensions a r e
out w - i n w - k w + 1 ,
o u t h = i n h - k h + 1
add t h e values i n t h e output m a t r i x t o o u r final once we complete t h e convops f o r each o f t h e Input output
matrices, w e u s u a l l y w a n t t o perform a
n o r m a l i z a t i o n s t e p
we can divide the values in t h e output matrix by t h e number o f inputs t o g e t a v e r a g e s f o r t h e m , o r
apply t h e a c t i v a t i o n f u n c t i o n t o e a c h c e l l i n t h e m a t r i x
we p a s s t h e normalized o u t p u t forward
backward:
error i s computed by each o f o u r output neurons ( o t h e r convolutions o r p o o l i n g n e u r o n s )
neurons during backprop
conv neurons have an error matrix which h a s values accumulated b y t h e c o m p u t a t i o n s o f t h e o u t p u t
i f you u s e a n activation i n t o normalize t h e o u t p u t s , fi r s t