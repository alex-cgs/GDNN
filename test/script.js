const fs = require("fs");
const path = require("path");
const convert = require("gd-level-to-json");

console.log(convert(fs.readFileSync(path.resolve(__dirname, "level.txt")).toString())); // { properties: { ... }, objects: [ ... ] }