def count_paths(m,n):
    results=[[1]*n]*m
    #初始化方法2：
    #results=[[1 for _ in range(n)]for _ in range(m)] 空间复杂度O(nm)，上面方法空间复杂度仅O(n)
    for i in range(1,m):
        for j in range(1,n):
            results[i][j]=results[i-1][j]+results[i][j-1]
    return results[-1][-1]

if __name__=='__main__':
    result=count_paths(1,1)
    print(result)

'''
leetcode 62. 使用动态规划(DP)解决不同路径问题
'''