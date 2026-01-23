import{truncateArray}from"./array.js";
import{truncateObject}from"./object.js";
import{addSize,getValueSize}from"./size.js";


export const truncateValue=({
value,
truncatedProps,
path,
size,
maxSize,
indent,
depth
})=>{
const increment=getValueSize(value);
const{
size:sizeA,
stop,
truncatedProps:truncatedPropsA
}=addSize({
size,
increment,
maxSize,
truncatedProps,
path,
value
});
return stop?
{value:undefined,size:sizeA,truncatedProps:truncatedPropsA}:
recurseValue({
value,
truncatedProps:truncatedPropsA,
path,
size:sizeA,
maxSize,
indent,
depth
})
};










const recurseValue=({
value,
truncatedProps,
path,
size,
maxSize,
indent,
depth
})=>{
if(typeof value!=="object"||value===null){
return{value,size,truncatedProps}
}

return Array.isArray(value)?
truncateArray({
array:value,
truncatedProps,
path,
size,
maxSize,
truncateValue,
indent,
depth
}):
truncateObject({
object:value,
truncatedProps,
path,
size,
maxSize,
truncateValue,
indent,
depth
})
};