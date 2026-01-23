
export const validateOptions=(jsonString,maxSize)=>{
if(typeof jsonString!=="string"){
throw new TypeError(`Input must be a JSON string: ${jsonString}`)
}

validateMaxSize(maxSize)
};

const validateMaxSize=(maxSize)=>{
checkMaxSizeType(maxSize);

if(maxSize<0){
throw new TypeError(`"maxSize" argument must be positive: ${maxSize}`)
}

if(maxSize<MIN_MAX_SIZE){
throw new TypeError(
`"maxSize" argument must be at least ${MIN_MAX_SIZE}: ${maxSize}`
)
}
};

const checkMaxSizeType=(maxSize)=>{
if(maxSize===undefined){
throw new TypeError("\"maxSize\" argument must be defined")
}

if(!Number.isInteger(maxSize)){
throw new TypeError(`"maxSize" argument must be an integer: ${maxSize}`)
}
};





export const MIN_MAX_SIZE=7;