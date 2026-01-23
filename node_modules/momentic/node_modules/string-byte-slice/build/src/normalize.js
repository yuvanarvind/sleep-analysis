


export const normalizeByteEnd=(input,byteEnd)=>{
if(byteEnd===undefined){
return byteEnd
}

const byteEndA=normalizeByteIndex(input,byteEnd);
return byteEndA>=input.length*MAX_UTF8_CHAR_LENGTH?undefined:byteEndA
};

export const normalizeByteIndex=(input,byteIndex)=>
byteIndex<=input.length*-MAX_UTF8_CHAR_LENGTH?0:byteIndex;

const MAX_UTF8_CHAR_LENGTH=4;