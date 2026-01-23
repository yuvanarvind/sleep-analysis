
export const getByteStart=(buffer,bufferLength,byteStart)=>{
const byteStartA=convertNegativeIndex(bufferLength,byteStart);
return findByteStart(buffer,bufferLength,byteStartA)
};


const findByteStart=(buffer,bufferLength,byteStart)=>{
if(byteStart>=bufferLength){
return byteStart
}

const byte=buffer[byteStart];
return byte>=NEXT_BYTES_START&&byte<=NEXT_BYTES_END?
findByteStart(buffer,bufferLength,byteStart+1):
byteStart
};


export const getByteEnd=(buffer,bufferLength,byteEnd)=>{
if(byteEnd===undefined){
return byteEnd
}

const byteEndA=convertNegativeIndex(bufferLength,byteEnd);
return findByteEnd(buffer,byteEndA)
};


const findByteEnd=(buffer,byteEndA)=>{
if(isInvalid4Sequence(buffer,byteEndA)){
return byteEndA-3
}

if(isInvalid3Sequence(buffer,byteEndA)){
return byteEndA-2
}

if(isInvalid2Sequence(buffer,byteEndA)){
return byteEndA-1
}

return byteEndA
};

const isInvalid4Sequence=(buffer,byteEnd)=>
byteEnd>=3&&
buffer[byteEnd-3]>=FIRST_BYTE_4_START&&
buffer[byteEnd-3]<=FIRST_BYTE_4_END;

const isInvalid3Sequence=(buffer,byteEnd)=>
byteEnd>=2&&buffer[byteEnd-2]>=FIRST_BYTE_3_START;

const isInvalid2Sequence=(buffer,byteEnd)=>
byteEnd>=1&&buffer[byteEnd-1]>=FIRST_BYTE_2_START;

const convertNegativeIndex=(bufferLength,byteIndex)=>
byteIndex<0||Object.is(byteIndex,-0)?
Math.max(bufferLength+byteIndex,0):
byteIndex;


const FIRST_BYTE_4_START=240;
const FIRST_BYTE_4_END=244;

const FIRST_BYTE_3_START=224;

const FIRST_BYTE_2_START=194;





const NEXT_BYTES_START=128;
const NEXT_BYTES_END=191;