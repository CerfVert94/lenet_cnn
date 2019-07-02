
/* Include Files */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <malloc.h>
#include "neural_network.h"
//#include "data.h"

#if defined(_CHAR_WEIGHT_)
	#include "char_weight.h"
#elif defined(_SHORT_WEIGHT_)
	#include "short_weight.h"
#else	
	#include "float_weight.h"
#endif


void write_weights(FILE *ptr_file1, FILE *ptr_file2, const char* layer_name, float *Layer_Weights_CPU, int size);
void read_data(const char *, char *Input);
void write_data(char (*)[841], int size);

#ifndef _CONV_WEIGHTS_
int evaluate(char *Input)
{
	
	float	Layer1_Neurons_CPU[IMG_WIDTH*IMG_HEIGHT],
				Layer2_Neurons_CPU[6*13*13],
				Layer3_Neurons_CPU[50*5*5],
				Layer4_Neurons_CPU[100];
	double Layer5_Neurons_CPU[10];
	double scoremax = 0.0;
	int i;

	int indexmax = -1;
	calculateLayer1(Input, Layer1_Neurons_CPU);
	calculateLayer2(Layer1_Neurons_CPU, Layer1_Weights_CPU, Layer2_Neurons_CPU);
	calculateLayer3(Layer2_Neurons_CPU, Layer2_Weights_CPU, Layer3_Neurons_CPU);
	calculateLayer4(Layer3_Neurons_CPU, Layer3_Weights_CPU, Layer4_Neurons_CPU);
	calculateLayer5(Layer4_Neurons_CPU, Layer4_Weights_CPU, Layer5_Neurons_CPU);

	scoremax = -999;
	indexmax = -1;
	for(i=0;i<10;i++)
	{
		if(Layer5_Neurons_CPU[i] > scoremax)
		{
			scoremax = Layer5_Neurons_CPU[i];
			indexmax = i;
		}
	}
	return indexmax;
}
#endif
/* Main function. */
int main(void){
	#if !defined(_CONV_WEIGHTS_)
	char Input[10][29*29] = {};

	//Load gray scale image.
	const char *bmp_path[10] = {"./Data/0.bmp",
								"./Data/1.bmp",
								"./Data/2.bmp",
								"./Data/3.bmp",
								"./Data/4.bmp",
								"./Data/5.bmp",
								"./Data/6.bmp",
								"./Data/7.bmp",
								"./Data/8.bmp",
								"./Data/9.bmp"};
	int nb_idx, input_idx, result;
	
	for (nb_idx = 0; nb_idx < 10; nb_idx++) {
		// Read the image
		read_data(bmp_path[nb_idx], Input[nb_idx]);
		// Print the handwritten digit : 
		for(input_idx = 0; input_idx < IMG_WIDTH * IMG_HEIGHT; input_idx++) {
			if((input_idx % IMG_WIDTH) == 0)
				fputc('\n',stdout);
			
			fputc(Input[nb_idx][input_idx] + '0',stdout);
		}
		fputc('\n',stdout);
		
		// Run the neural network to recognized the digit : 
		result = evaluate(Input[nb_idx]);
		printf("The recognized digit is %d\n", result);	
		if (nb_idx != result) {
			fprintf(stderr, "Recognized digit is a mismatch : %d =/= %d\n", nb_idx, result);
			exit(EXIT_FAILURE);
		}
		
	}
	write_data(Input, 10);
	fprintf(stdout, "Success\n");
	#else
	#define NB_WEIGHTS_LAYER1 ((5 * 5 + 1) * 6)
	#define NB_WEIGHTS_LAYER2 (((5 * 5 + 1) * 6) * 50)
	#define NB_WEIGHTS_LAYER3 ((5 * 5 * 50 + 1) * 100)
	#define NB_WEIGHTS_LAYER4 ((100 + 1) * 10)
	
	FILE *ptr_file1, *ptr_file2;
	// Create new header files.
	ptr_file1 = fopen("short_weight.h", "wb");
	ptr_file2 = fopen("char_weight.h", "wb");
	// Null pointer signifies an error :
	if (!ptr_file1 || !ptr_file2) {
		fprintf(stdout, "Failed creating new header files\n");
		exit(EXIT_FAILURE);
	}
	
	fprintf(ptr_file1, "#ifndef _SHORT_WEIGHT_H_\n#define _SHORT_WEIGHT_H_\n");
	fprintf(ptr_file2, "#ifndef _CHAR_WEIGHT_H_\n#define _CHAR_WEIGHT_H_\n");
	write_weights(ptr_file1, ptr_file2, "Layer1_Weights_CPU", Layer1_Weights_CPU, NB_WEIGHTS_LAYER1);
	write_weights(ptr_file1, ptr_file2, "Layer2_Weights_CPU", Layer2_Weights_CPU, NB_WEIGHTS_LAYER2);
	write_weights(ptr_file1, ptr_file2, "Layer3_Weights_CPU", Layer3_Weights_CPU, NB_WEIGHTS_LAYER3);
	write_weights(ptr_file1, ptr_file2, "Layer4_Weights_CPU", Layer4_Weights_CPU, NB_WEIGHTS_LAYER4);
	fprintf(ptr_file1, "#endif\n");
	fprintf(ptr_file2, "#endif\n");
	fclose(ptr_file1);
	fclose(ptr_file2);
	#endif
	return 0;
}
void write_weights(FILE *ptr_file1, FILE *ptr_file2, const char* layer_name, float *Layer_Weights_CPU, int size) {
	
	#define fprintfs(file1, file2, str) 	fprintf(file1, str);\
											fprintf(file2, str)
	int i;
	fprintf(ptr_file1, "short %s[%d] = {", layer_name, size);
	fprintf(ptr_file2, "char %s[%d] = {", layer_name, size);
	for (i = 0; i < size; i++) {
		fprintf(ptr_file1, "%d", (short)(Layer_Weights_CPU[i] * 100));
		fprintf(ptr_file2, "%d", (char)(Layer_Weights_CPU[i] * 10));
		if (i < (size - 1))
			fprintfs(ptr_file1, ptr_file2, ", ");
	}
	fprintfs(ptr_file1, ptr_file2, "};\n");
	
	#undef fprintfs
}

#define BYTE_TO_BITS(n) (n * 8)
#define Y_POS(idx, width, height) (((height - 1) - idx / width) * height)
void read_data(const char * path, char *Input) {
	
	FILE *ptr_file;
	int32_t size;
	int i, j, idx;
	const int pd_offset = 62; //offset to pd (pixel data)
	const int bpl = BYTE_TO_BITS(4); // bits_per_line = bpl.
	char ch;
	
	// Open the B&W bitmap file of the given path.
	ptr_file = fopen(path, "rb");
	
	// Null pointer signifies an error :
	if (!ptr_file) {
		fprintf(stdout, "Failed opening the data.\n");
		exit(EXIT_FAILURE);
	}
	// Get the size of the bitmap file.
	fseek(ptr_file, 0, SEEK_END);
	size = ftell(ptr_file);
	
	// Skips header info.
	fseek(ptr_file, pd_offset, SEEK_SET);
	
	// Read bitmap : 
	idx = 0;
	for (i = 0; i < size - pd_offset; i++) {
		ch = fgetc(ptr_file);
		// Each byte represents 8 pixels in reverse order.
		for (j = BYTE_TO_BITS(1) - 1; j >= 0 ; j--) {
			/* 
			Consider zero padding : 
			3 bytes are padded.
			
			Store only significant pixel data.
			*/
			if((idx % bpl) < IMG_WIDTH) {
				Input[Y_POS(idx, bpl, IMG_HEIGHT) + (idx % bpl)] = (ch >> j & 0x1);
			}
			idx++;
		}
	}
	fclose(ptr_file);
}

void write_data(char (*Input)[841], int size) 
{
	FILE *ptr_file;
	int i, j, k;
	// Create a new data header file.
	ptr_file = fopen("data.h", "wb");
	// Null pointer signifies an error :
	if (!ptr_file) {
		fprintf(stdout, "Failed creating new header files\n");
		exit(EXIT_FAILURE);
	}
	fprintf(ptr_file,	"#ifndef _DATA_H_\n"\
						"#define	_DATA_H_\n"\
						"\tchar Input[10][29*29] = {");
	for (i = 0; i < size; i++) {
		fputs("{\n\t\t", ptr_file);
		for (j = 0; j < IMG_HEIGHT; j++) {
			for (k = 0; k < IMG_WIDTH; k++) {				
				fputc('0' + Input[i][j * IMG_HEIGHT + k], ptr_file);
				if (j == (IMG_HEIGHT - 1) && k == (IMG_WIDTH - 1)) 
					fputs("}", ptr_file);
				else {
					fputc(',', ptr_file);
				}
				
			}
			if (j < (IMG_HEIGHT - 1))
				fputs("\n\t\t", ptr_file);
		}
		if (i < (size - 1))
			fputs(",\n\t\t", ptr_file);
		
	}					
	fputs("};\n", ptr_file);
	fprintf(ptr_file, "#endif /* DATA_H_ */");
	fclose(ptr_file);
}