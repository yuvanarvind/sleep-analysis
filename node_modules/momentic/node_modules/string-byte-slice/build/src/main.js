import{bufferSlice}from"./buffer.js";
import{charCodeSlice}from"./char_code/main.js";
import{textEncoderSlice}from"./encoder.js";
import{normalizeByteEnd,normalizeByteIndex}from"./normalize.js";
import{replaceInvalidSurrogate}from"./surrogate.js";
import{validateInput}from"./validate.js";
import{estimateCharWidth}from"./width.js";


const stringByteSlice=(input,byteStart,byteEnd)=>{
validateInput(input,byteStart,byteEnd);

if(input===""){
return input
}

const byteStartA=normalizeByteIndex(input,byteStart);
const byteEndA=normalizeByteEnd(input,byteEnd);

if(byteEndA===undefined&&Object.is(byteStartA,0)){
return replaceInvalidSurrogate(input)
}

return useBestSlice(input,byteStartA,byteEndA)
};

export default stringByteSlice;









const useBestSlice=(input,byteStart,byteEnd)=>{
if(input.length<=CHAR_CODE_MIN_LENGTH){
return charCodeSlice(input,byteStart,byteEnd)
}

const{asciiOnly,longCharsPercentage}=estimateCharWidth(input);

if(asciiOnly){
return tryBufferSlice(input,byteStart,byteEnd)
}

return longCharsPercentage>=CHAR_CODE_MIN_PERC?
charCodeSlice(input,byteStart,byteEnd):
tryTextEncoderSlice(input,byteStart,byteEnd)
};




const CHAR_CODE_MIN_LENGTH=200;

const CHAR_CODE_MIN_PERC=0.4;


const tryBufferSlice=(input,byteStart,byteEnd)=>

"Buffer"in globalThis&&"from"in globalThis.Buffer?
bufferSlice(input,byteStart,byteEnd):
/* c8 ignore next */
tryTextEncoderSlice(input,byteStart,byteEnd);


const tryTextEncoderSlice=(input,byteStart,byteEnd)=>
"TextEncoder"in globalThis?
textEncoderSlice(input,byteStart,byteEnd):
/* c8 ignore next */
charCodeSlice(input,byteStart,byteEnd);