import guessJsonIndent from"guess-json-indent";

import{truncateNumber}from"./number.js";
import{validateOptions}from"./options.js";
import{truncateString}from"./string.js";
import{truncateValue}from"./value.js";


const truncateJson=(jsonString,maxSize)=>{
validateOptions(jsonString,maxSize);
const indent=getIndent(jsonString);
const value=parseJson(jsonString);
const{value:newValue,truncatedProps}=truncateValue({
value,
truncatedProps:[],
path:[],
size:0,
maxSize,
indent,
depth:0
});
const newJsonString=serializeJson({newValue,value,maxSize,indent});
return{jsonString:newJsonString,truncatedProps}
};

export default truncateJson;






const getIndent=(jsonString)=>{
const indent=guessJsonIndent(jsonString);
return typeof indent==="string"?indent.length:indent
};

const parseJson=(jsonString)=>{
try{
return JSON.parse(jsonString)
}catch(error){
throw new TypeError(
`Invalid JSON string: "${jsonString}"\n${error.message}`
)
}
};









const serializeJson=({newValue,value,maxSize,indent})=>{
if(newValue!==undefined){
return JSON.stringify(newValue,undefined,indent)
}

return typeof value==="number"?
truncateNumber(value,maxSize):
truncateString(value,maxSize)
};