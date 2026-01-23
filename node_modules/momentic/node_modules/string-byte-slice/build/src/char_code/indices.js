import{
LAST_ASCII_CODEPOINT,
LAST_TWO_BYTES_CODEPOINT}from
"../codepoints.js";









export const findCharIndex=({
input,
targetByteCount,
firstStartSurrogate,
lastStartSurrogate,
firstEndSurrogate,
lastEndSurrogate,
increment,
canBacktrack,
shift,
charIndexInit
})=>{
let charIndex=charIndexInit;
let previousCharIndex=charIndex;
let byteCount=0;

for(;byteCount<targetByteCount;charIndex+=increment){
previousCharIndex=charIndex;
const codepoint=input.charCodeAt(charIndex);

if(Number.isNaN(codepoint)){
break
}

if(codepoint<=LAST_ASCII_CODEPOINT){
byteCount+=1;
continue
}

if(codepoint<=LAST_TWO_BYTES_CODEPOINT){
byteCount+=2;
continue
}

byteCount+=3;

if(codepoint<firstStartSurrogate||codepoint>lastStartSurrogate){
continue
}

const nextCodepoint=input.charCodeAt(charIndex+increment);





if(
Number.isNaN(nextCodepoint)||
nextCodepoint<firstEndSurrogate||
nextCodepoint>lastEndSurrogate)
{
continue
}

byteCount+=1;
charIndex+=increment
}

const finalCharIndex=
canBacktrack&&byteCount>targetByteCount?previousCharIndex:charIndex;
return finalCharIndex+shift
};