function reverseSubstring(input) {
    let stack = [];
    for (let i = 0; i<input.length; i++) {
        if (input[i] === '(') {
            stack.push(i);
        } else if (input[i] === ')') {
            let start = stack.pop();
            let reversedString = input.substring(start+1,i).split("").reverse().join("");
            input = input.substring(0,start) + reversedString + input.substring(i+1);
            i-=2;
        }
    }
    return input;
}

let input = "foo(bar(baz))blim";
console.log(reverseSubstring(input));