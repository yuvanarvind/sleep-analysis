import stringByteLength from"string-byte-length";



export const getJsonLength=(value)=>{
if(value===null){
return NULL_LENGTH
}

if(value===true){
return TRUE_LENGTH
}

if(value===false){
return FALSE_LENGTH
}

const type=typeof value;

if(type==="object"){
return OBJ_ARR_LENGTH
}

if(type==="number"){
return JSON.stringify(value).length
}

return getJsonStringLength(value)
};

const NULL_LENGTH=4;
const TRUE_LENGTH=4;
const FALSE_LENGTH=5;
const OBJ_ARR_LENGTH=2;





export const getJsonStringLength=(string)=>
stringByteLength(JSON.stringify(string));