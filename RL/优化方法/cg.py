import numpy as np
def cg(f_Ax,b,cg_iters=10,callback=None,verbose=False,residual_tol=1e-10):
    p=b.copy()  #真正的优化方向p
    r=b.copy()  #残差r
    x=np.zeros_like(b)
    rdotr=r.dot(r)
    for i in range(cg_iters):
        z=f_Ax(p)
        v=rdotr/p.dot(z)    #优化步长v
        x+= v*p
        r-= v*z
        newrdotr=r.dot(r)
        mu=newrdotr/rdotr
        p=r+mu*p
        rdotr=newrdotr
        if rdotr<residual_tol:
            break
    return x