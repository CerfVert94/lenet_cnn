#include "neural_network.h"
#include <math.h>
#include <stdio.h>

void calculateLayer1(char* input, float* Layer1_Neurons_CPU){
	int i, j;
	for (i = 0; i < IMG_HEIGHT; i++) 
		for (j = 0; j < IMG_WIDTH; j++) 
			Layer1_Neurons_CPU[i * IMG_HEIGHT + j] = input[i * IMG_HEIGHT + j];
		
}

void calculateLayer2(float* Layer1_Neurons_CPU, WEIGHT_TYPE* Layer1_Weights_CPU, float* Layer2_Neurons_CPU){
	float somme;
	int i,j,k,m,n;
	for(i=0;i<6;i++)
		for(j=0;j<13;j++)
			for(k=0;k<13;k++){
				somme = Layer1_Weights_CPU[26*i];
				for(m=0;m<5;m++)
					for(n=0;n<5;n++)
						somme += Layer1_Weights_CPU[26*i+5*m+n+1] * Layer1_Neurons_CPU[29*(m+2*j)+n+2*k];
				Layer2_Neurons_CPU[13*13*i+13*j+k] = SIGMOID(CONV(somme));
			}
}

void calculateLayer3(float* Layer2_Neurons_CPU, WEIGHT_TYPE* Layer2_Weights_CPU, float* Layer3_Neurons_CPU){
	float somme;
	int i,j,k,m,n;
	for( i=0;i<50;i++)
		for(j=0;j<5;j++)
			for(k=0;k<5;k++){
				somme = Layer2_Weights_CPU[26*6*i];

				for( m=0;m<5;m++)
					for( n=0;n<5;n++){
						somme += Layer2_Weights_CPU[26*6*i+1+6*(n+5*m)	] * Layer2_Neurons_CPU[13*13*0+13*(2*j+m)+(2*k+n)];
						somme += Layer2_Weights_CPU[26*6*i+1+6*(n+5*m)+1] * Layer2_Neurons_CPU[13*13*1+13*(2*j+m)+(2*k+n)];
						somme += Layer2_Weights_CPU[26*6*i+1+6*(n+5*m)+2] * Layer2_Neurons_CPU[13*13*2+13*(2*j+m)+(2*k+n)];
						somme += Layer2_Weights_CPU[26*6*i+1+6*(n+5*m)+3] * Layer2_Neurons_CPU[13*13*3+13*(2*j+m)+(2*k+n)];
						somme += Layer2_Weights_CPU[26*6*i+1+6*(n+5*m)+4] * Layer2_Neurons_CPU[13*13*4+13*(2*j+m)+(2*k+n)];
						somme += Layer2_Weights_CPU[26*6*i+1+6*(n+5*m)+5] * Layer2_Neurons_CPU[13*13*5+13*(2*j+m)+(2*k+n)];

					}
				Layer3_Neurons_CPU[5*5*i+5*j+k] = SIGMOID(CONV(somme));
				
			}
}

void calculateLayer4(float* Layer3_Neurons_CPU, WEIGHT_TYPE* Layer3_Weights_CPU, float* Layer4_Neurons_CPU){
	float somme;
	int i, j, k, m;
	for( i=0;i<100;i++){
		somme = Layer3_Weights_CPU[i*(1+50*25)];
		for( j=0;j<50;j++)
			for( k=0;k<5;k++)
				for ( m=0;m<5;m++)
					somme += Layer3_Weights_CPU[i*(1+50*25)+1 + m + k*5 + j*25] * Layer3_Neurons_CPU[m+5*k+25*j];

		Layer4_Neurons_CPU[i] = SIGMOID(CONV(somme));
	}

}

void calculateLayer5(float* Layer4_Neurons_CPU, WEIGHT_TYPE* Layer4_Weights_CPU, double* Layer5_Neurons_CPU){
	float somme;
	int i, j;
	for( i=0;i<10;i++){
		somme = Layer4_Weights_CPU[101*i];
		for( j=0;j<100;j++)
			somme += Layer4_Weights_CPU[1+101*i+j] * Layer4_Neurons_CPU[j];
		Layer5_Neurons_CPU[i] = SIGMOID(CONV(somme));
		
	}
}
