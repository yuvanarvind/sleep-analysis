import{truncateProp}from"./prop.js";
import{getObjectPropSize}from"./size.js";




export const truncateObject=({
object,
truncatedProps,
path,
size,
maxSize,
truncateValue,
indent,
depth
})=>{
const newObject={};
let state={empty:true,size,truncatedProps};


for(const key in object){
const increment=getObjectPropSize({
key,
empty:state.empty,
indent,
depth
});

state=truncateProp({
parent:object,
truncatedProps:state.truncatedProps,
path,
increment,
maxSize,
key,
empty:state.empty,
size:state.size,
truncateValue,
indent,
depth
});

if(state.value!==undefined){
newObject[key]=state.value
}
}

return{
value:newObject,
size:state.size,
truncatedProps:state.truncatedProps
}
};