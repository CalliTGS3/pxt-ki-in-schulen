#include "Mat.h"
#include "Vect.h"
#include "NNLayer.h"
#include "NN.h"
#include "../platform/Utils.h"
#include <stdio.h>



NN::NN(int num_input) {
	this->num_input = num_input;
	this->num_layers = 0;
	this->layers = 0;
}

int NN::getNumLayers() const {
	return num_layers;
}

NNLayer* NN::getLayer(int n) const {
	return layers[n];
}

void NN::addLayer(int num_output) {
	int old_output = getNumOutput();
	if (num_layers == 0) {
		layers = new NNLayer*[1];
	}
	else {
		NNLayer **old_layers = layers;
		layers = new NNLayer*[num_layers+1];
		for (int i=0; i<num_layers; i++) {
			layers[i] = old_layers[i];
		}
		delete old_layers;
	}
	layers[num_layers] = new NNLayer(old_output, num_output);
	num_layers = num_layers + 1;
}


int NN::getNumInput() const {
	return num_input;
}

int NN::getNumOutput() const {
	if (num_layers == 0) {
		return getNumInput();
	}
	return layers[num_layers-1]->getNumOutput();
}

Vect* NN::forwardPropagate(const Vect *input) {
	if (num_layers == 0) {
		Vect *result = new Vect(input);
		return result;
	}
	Vect *result = (Vect *)input;
	Vect* tempResult = 0;
	for (int i=0; i<num_layers; i++) {
		result = layers[i]->forwardPropagate(result);
		if (tempResult != 0) {
			delete tempResult;
		}
		tempResult = result;
	}
	return result;
}

Vect* NN::backwardPropagate(const Vect *expected_output, const Vect *last_output, float learning_rate) {
	Vect *err = new Vect(last_output);
	err->sub(expected_output);
	for (int n=num_layers-1; n>=0; n--) {
		NNLayer *layer = layers[n];
		Vect *err_in = layer->backwardPropagate(err, learning_rate);
		delete err;
		err = err_in;
	}
	return err;
}



void NN::print() const {
	char buf[32];
	sprintf(buf, "NN(%d)\r\n", num_layers);
	log(buf);
	for (int i=0; i<num_layers; i++) {
		layers[i]->print();
	}
}


void set_weights(Mat* weights, float* data) {
	for (int r=0; r<weights->getRows(); r++) {
		for (int c=0; c<weights->getCols(); c++) {
			weights->set(r, c, *data);
			data++;
		}
	}
}

void set_biases(Vect* biases, float* data) {
	for (int i=0; i<biases->getLength(); i++) {
		biases->set(i, *data);
		data++;
	}
}

NN::~NN() {
	for (int i=0; i<num_layers; i++) {
		delete layers[i];
	}
	num_layers = 0;
	delete layers;
	layers = 0;
}

