import{getJsonLength,getJsonStringLength}from"./length.js";











export const addSize=({
size,
increment,
maxSize,
truncatedProps,
path,
value
})=>{
const newSize=size+increment;
const stop=newSize>maxSize;
return stop?
{size,stop,truncatedProps:[...truncatedProps,{path,value}]}:
{size:newSize,stop,truncatedProps}
};


export const getValueSize=(value)=>getJsonLength(value);


export const getArrayItemSize=(empty,indent,depth)=>{
const indentSize=getIndentSize({empty,indent,depth,keySpaceSize:0});
const commaSize=getCommaSize(empty);
return indentSize+commaSize
};


export const getObjectPropSize=({key,empty,indent,depth})=>{
const indentSize=getIndentSize({empty,indent,depth,keySpaceSize:1});
const keySize=getJsonStringLength(key);
const commaSize=getCommaSize(empty);
return indentSize+keySize+COLON_SIZE+commaSize
};

const COLON_SIZE=1;



const getIndentSize=({empty,indent,depth,keySpaceSize})=>{
if(indent===undefined){
return 0
}

const propSpaces=NEWLINE_SIZE+indent*(depth+1);
const parentSpaces=empty?NEWLINE_SIZE+indent*depth:0;
return keySpaceSize+propSpaces+parentSpaces
};

const NEWLINE_SIZE=1;


const getCommaSize=(empty)=>empty?0:COMMA_SIZE;

const COMMA_SIZE=1;