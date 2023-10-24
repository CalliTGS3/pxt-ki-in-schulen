/**
* Andy England @ SparkFun Electronics
* September 6, 2018
* https://github.com/sparkfun/pxt-light-bit
*
* Development environment specifics:
* Written in Microsoft PXT
* Tested with a SparkFun temt6000 sensor and micro:bit
*
* This code is released under the [MIT License](http://opensource.org/licenses/MIT).
* Please review the LICENSE.md file included with this example. If you have any questions
* or concerns with licensing, please contact techsupport@sparkfun.com.
* Distributed as-is; no warranty is given.
*/


#include <pxt.h>
#include <MicroBit.h>
#include <cstdint>
#include <math.h>

#include "platform/Utils.h"
#include "common/logUtils.h"

#include "neuralnets/NN.h"
#include "neuralnets/NNLayer.h"
#include "neuralnets/Vect.h"

#include "json/Parser.h"
#include "json/NNJsonParser.h"


using namespace pxt;

namespace nnpredict {

static NN *brain = 0;

Vect* toVect(RefCollection &param) {
    int len = param.length();
    Vect *result = new Vect(len);
    for (int i=0; i<len; i++) {
	    TNumber tn = param.getAt(i);
		float f = toFloat(tn);
		result->set(i, f);
    }
    return result;
}

RefCollection *toRefCollection(Vect *vect) {
    int len = vect->getLength();
    RefCollection *result = Array_::mk();
    for (int i=0; i<len; i++) {
    	float v = vect->get(i);
	    Array_::insertAt(result, i, fromFloat(v));
    }
    return result;
}

//% blockId=nntest_fcnnfromjson
//% block="Json Brain|string %json"
//% shim=nntest::fcnnfromjson
void fcnnfromjson(String json) {

	if (brain != 0) {
		delete brain;
		brain = 0;
	}
	const char *jsonNN = PXT_STRING_DATA(json);
	NNJsonParser nnParser;
	Parser parser(&nnParser);

	parser.parse(jsonNN);
	brain = (NN*) nnParser.getResult();

/*
	if (brain != 0) {
		brain->print();
	}
*/	
}

//% blockId=nntest_predict
//% block="Predict|number[] %input|number[] %output"
//% shim=nntest::predict
void predict(RefCollection &input, RefCollection &output) {
    Vect *x = toVect(input);
	Vect *y_hat = brain->forwardPropagate(x);
	output.setLength(y_hat->getLength());
	for (int i=0; i<y_hat->getLength(); i++) {
		output.head.set(i, fromFloat(y_hat->get(i)));
	}
	delete x;
	delete y_hat;
}


}
