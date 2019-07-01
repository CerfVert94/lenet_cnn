all : float 
 

float :
	gcc -Wall -g main.c neural_network.c -o main -lm
short : 
	gcc -Wall -g main.c neural_network.c -o main -lm -D_SHORT_WEIGHT_
char : 
	gcc -Wall -g main.c neural_network.c -o main -lm -D_CHAR_WEIGHT_
conv : 
	gcc -Wall -g main.c -o main -lm -D_CONV_WEIGHTS_
	./main
	rm main
	
.PHONY: clean

clean : 
	rm main short_weight.h char_weight.h
	