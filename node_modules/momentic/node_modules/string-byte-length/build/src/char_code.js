



export const getCharCodeByteLength=(string)=>{
const charLength=string.length;
let byteLength=charLength;

for(let charIndex=0;charIndex<charLength;charIndex+=1){
const codepoint=string.charCodeAt(charIndex);

if(codepoint<=LAST_ASCII_CODEPOINT){
continue
}

if(codepoint<=LAST_TWO_BYTES_CODEPOINT){
byteLength+=1;
continue
}

byteLength+=2;

if(codepoint<FIRST_HIGH_SURROGATE||codepoint>LAST_HIGH_SURROGATE){
continue
}



const nextCodepoint=string.charCodeAt(charIndex+1);





if(
nextCodepoint<FIRST_LOW_SURROGATE||
nextCodepoint>LAST_LOW_SURROGATE)
{
continue
}

charIndex+=1
}

return byteLength
};


const LAST_ASCII_CODEPOINT=127;

const LAST_TWO_BYTES_CODEPOINT=2047;




const FIRST_HIGH_SURROGATE=55296;
const LAST_HIGH_SURROGATE=56319;
const FIRST_LOW_SURROGATE=56320;
const LAST_LOW_SURROGATE=57343;