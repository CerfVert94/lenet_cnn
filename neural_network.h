#ifndef _NEURAL_NETWORK_H_
#define _NEURAL_NETWORK_H_

#define IMG_WIDTH 29
#define IMG_HEIGHT 29

#define SIGMOID(x) (1.7159*tanh(0.66666667*x))
#define DSIGMOID(S) (0.66666667/1.7159*(1.7159+(S))*(1.7159-(S)))
	#if defined(_CHAR_WEIGHT_)
	#define CONV(n) ((n) / 10.0)
	typedef char WEIGHT_TYPE;
	#elif defined(_SHORT_WEIGHT_)
	#define CONV(n) ((n) / 100.0)
	typedef short WEIGHT_TYPE;	
	#else
	#define CONV(n) (n)
	typedef float WEIGHT_TYPE;
	#endif
	void calculateLayer1(char* input, float* Layer1_Neurons_CPU);
	void calculateLayer2(float* Layer1_Neurons_CPU, WEIGHT_TYPE* Layer1_Weights_CPU, float* Layer2_Neurons_CPU);
	void calculateLayer3(float* Layer2_Neurons_CPU, WEIGHT_TYPE* Layer2_Weights_CPU, float* Layer3_Neurons_CPU);
	void calculateLayer4(float* Layer3_Neurons_CPU, WEIGHT_TYPE* Layer3_Weights_CPU, float* Layer4_Neurons_CPU);
	void calculateLayer5(float* Layer4_Neurons_CPU, WEIGHT_TYPE* Layer4_Weights_CPU, double* Layer5_Neurons_CPU);
#endif