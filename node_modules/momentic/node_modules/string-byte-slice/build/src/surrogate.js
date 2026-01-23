import{
FIRST_HIGH_SURROGATE,
LAST_LOW_SURROGATE,
SURROGATE_REGEXP,
SURROGATE_REPLACE_CHAR}from
"./codepoints.js";







export const replaceInvalidSurrogate=(input)=>
hasSurrogates(input)?
input.replace(SURROGATE_REGEXP,SURROGATE_REPLACE_CHAR):
input;




const hasSurrogates=(input)=>{

for(let index=0;index<input.length;index+=1){
const codepoint=input.codePointAt(index);


if(codepoint>=FIRST_HIGH_SURROGATE&&codepoint<=LAST_LOW_SURROGATE){
return true
}
}

return false
};