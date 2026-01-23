
export const createTextEncoderFunc=()=>
getTextEncoderByteLength.bind(undefined,new TextEncoder);


const getTextEncoderByteLength=(textEncoder,string)=>{
const encoderBuffer=getBuffer(string);
return textEncoder.encodeInto(string,encoderBuffer).written
};


const getBuffer=(string)=>{
const size=string.length*3;

if(size>CACHE_MAX_MEMORY){
return new Uint8Array(size)
}

if(cachedEncoderBuffer===undefined||cachedEncoderBuffer.length<size){

cachedEncoderBuffer=new Uint8Array(size)
}

return cachedEncoderBuffer
};


export const CACHE_MAX_MEMORY=1e5;

let cachedEncoderBuffer;


export const TEXT_ENCODER_MIN_LENGTH=100;