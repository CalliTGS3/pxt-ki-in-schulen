/**
 * enum for activation function
 */

enum activationFunctionType{
	 None=0,
	 RELU=1,
	 SIGMOID=2,
     SOFTMAX=3,
     TANHYP=4
}

//% weight=70 icon="\u237E" color=#74DF00 block="AI"
namespace nnpredict {

    //% weight=90 
    //% blockId=nntest_fcnnfromjson
    //% block="Json Brain|string %json"
    //% shim=nnpredict::fcnnfromjson
    export function fcnnfromjson(json: string): void {
    	basic.showString("sim:json")
    }

    //% weight=50 
    //% blockId=nntest_predict
    //% block="Predict|number[] %input|number[] %output"
    //% shim=nnpredict::predict
    export function predict(input: number[], output: number[]): void {
    	basic.showString("sim-predict")
    }

 }
