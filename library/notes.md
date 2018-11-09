if you are compiling the code yourself using GCC (or Clang), you will need to use the -shared and -fPIC options

``` bash
gcc -c -Wall -Werror -fpic foo.c 
gcc -shared -o libfoo.so foo.o
```

``` julia
function getenv(var::AbstractString)
    val = ccall((:getenv, "libc.so.6"),
                Cstring, (Cstring,), var)
    if val == C_NULL
        error("getenv: undefined variable: ", var)
    end
    unsafe_string(val)
end
```
