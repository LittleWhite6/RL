def fib(n):
    results=list(range(n+1))
    for i in range(n+1):
        if i<2:
            results[i]=i
        else:
            results[i]=results[i-1]+results[i-2]
    return results[-1]

if __name__=='__main__':
    print(fib(100))