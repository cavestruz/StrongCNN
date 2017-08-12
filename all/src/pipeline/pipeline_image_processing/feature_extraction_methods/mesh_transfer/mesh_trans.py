import numpy as np
import pylab as pl
#--------------------------------------------------------------------
def lens_equation_sie(x1,x2,lpar):
    xc1 = lpar[0]   #x coordinate of the center of the lens (in units of Einstein radius).
    xc2 = lpar[1]   #y coordinate of the center of the lens (in units of Einstein radius).
    q   = lpar[2]   #Ellipticity of the lens.
    rc  = lpar[3]   #Core size of the lens (in units of Einstein radius).
    re  = lpar[4]   #Einstein radius of the lens.
    pha = lpar[5]   #Orientation of lens.

    phirad = np.deg2rad(pha)
    cosa = np.cos(phirad)
    sina = np.sin(phirad)

    xt1 = (x1-xc1)*cosa+(x2-xc2)*sina
    xt2 = (x2-xc2)*cosa-(x1-xc1)*sina

    phi = np.sqrt(xt2*xt2+xt1*q*xt1*q+rc*rc)
    sq = np.sqrt(1.0-q*q)
    pd1 = phi+rc/q
    pd2 = phi+rc*q
    fx1 = sq*xt1/pd1
    fx2 = sq*xt2/pd2
    qs = np.sqrt(q)

    a1 = qs/sq*np.arctan(fx1)
    a2 = qs/sq*np.arctanh(fx2)

    xt11 = cosa
    xt22 = cosa
    xt12 = sina
    xt21 =-sina

    fx11 = xt11/pd1-xt1*(xt1*q*q*xt11+xt2*xt21)/(phi*pd1*pd1)
    fx22 = xt22/pd2-xt2*(xt1*q*q*xt12+xt2*xt22)/(phi*pd2*pd2)
    fx12 = xt12/pd1-xt1*(xt1*q*q*xt12+xt2*xt22)/(phi*pd1*pd1)
    fx21 = xt21/pd2-xt2*(xt1*q*q*xt11+xt2*xt21)/(phi*pd2*pd2)

    a11 = qs/(1.0+fx1*fx1)*fx11
    a22 = qs/(1.0-fx2*fx2)*fx22
    a12 = qs/(1.0+fx1*fx1)*fx12
    a21 = qs/(1.0-fx2*fx2)*fx21

    rea11 = (a11*cosa-a21*sina)*re
    rea22 = (a22*cosa+a12*sina)*re
    rea12 = (a12*cosa-a22*sina)*re
    rea21 = (a21*cosa+a11*sina)*re

    y11 = 1.0-rea11
    y22 = 1.0-rea22
    y12 = 0.0-rea12
    y21 = 0.0-rea21

    jacobian = y11*y22-y12*y21
    mu = 1.0/jacobian

    res1 = (a1*cosa-a2*sina)*re
    res2 = (a2*cosa+a1*sina)*re
    return res1,res2,mu
#--------------------------------------------------------------------
def xy_rotate(x, y, xcen, ycen, phi):
	phirad = np.deg2rad(phi)
	xnew = (x-xcen)*np.cos(phirad)+(y-ycen)*np.sin(phirad)
	ynew = (y-ycen)*np.cos(phirad)-(x-xcen)*np.sin(phirad)
	return (xnew,ynew)
def gauss_2d(x, y, par):
	(xnew,ynew) = xy_rotate(x, y, par[2], par[3], par[5])
	res0 = ((xnew**2)*par[4]+(ynew**2)/par[4])/np.abs(par[1])**2
	res = par[0]*np.exp(-0.5*res0)
	return res
#--------------------------------------------------------------------
# This function shows how to move points on grids to the local peaks of the luminosity map.
#--------------------------------------------------------------------
def pixels_trans(x1,x2,xx1,xx2,matrix,ntmp):

    ntmp = ntmp*1.0
    dr = 1000
    nnx,nny = np.shape(matrix)
    dsx = xx1[1,1]-xx1[0,0]
    bsz = nnx*dsx

    buf = np.zeros((2.0*ntmp,2.0*ntmp))
    buf_x1 = np.zeros((2.0*ntmp,2.0*ntmp))
    buf_x2 = np.zeros((2.0*ntmp,2.0*ntmp))

    kk = 0

    while (dr >dsx) :
        i = int((x2+bsz*0.5)/dsx)
        j = int((x1+bsz*0.5)/dsx)

        idx1 = i-ntmp;sidx1 = 0
        idx2 = i+ntmp;sidx2 = 2*ntmp

        idy1 = j-ntmp;sidy1 = 0
        idy2 = j+ntmp;sidy2 = 2*ntmp

        if ((idx1<0)|(idx2>=nnx)|(idy1<0)|(idy2>=nny)):
            break

        buf[sidx1:sidx2,sidy1:sidy2] = matrix[idx1:idx2,idy1:idy2]
        buf_x1[sidx1:sidx2,sidy1:sidy2] = xx1[idx1:idx2,idy1:idy2]
        buf_x2[sidx1:sidx2,sidy1:sidy2] = xx2[idx1:idx2,idy1:idy2]
        buf_tot = np.sum(buf)
        buf_bar = buf_tot/(4.0*ntmp*ntmp)

        buf_nrm = buf-buf_bar

        buf_nrm[buf_nrm <=0] = 0
        buf_nrm_tot = np.sum(buf_nrm)

        if buf_nrm_tot == 0.0 :
            break

        xc2 = np.sum(buf_nrm*buf_x1)/buf_nrm_tot
        xc1 = np.sum(buf_nrm*buf_x2)/buf_nrm_tot

        dx1 = x1-xc1
        dx2 = x2-xc2

        dr = np.sqrt(dx1*dx1+dx2*dx2)
        x1 = xc1
        x2 = xc2

        kk = kk+1

        if kk > 100 :
            break

    return x1,x2

#------------------------------------------------------------------------------
if __name__ == '__main__':
    re = 1.0 # in units of arcsec
    boxsize = 6.0*re # in the units of arcsec
    nnn = 1024
    dsx = boxsize/nnn # arcsec

    xx01 = np.linspace(-boxsize/2.0,boxsize/2.0,nnn)+0.5*dsx
    xx02 = np.linspace(-boxsize/2.0,boxsize/2.0,nnn)+0.5*dsx
    xi2,xi1 = np.meshgrid(xx01,xx02)
    #----------------------------------------------------------------------
    # Set the parameters of the image of the source galaxy
    #
    g_amp = 1.0   	# peak brightness value
    g_sig = 0.02  	# Gaussian "sigma" (i.e., size)
    g_xcen = 0.03  	# x position of the center of the source (also try (0.0,0.14)
    g_ycen = 0.1  	# y position of the center of the source
    g_axrat = 1.0 	# minor-to-major axis ratio
    g_pa = 0.0    	# major-axis position angle (degrees) c.c.w. from x axis
    gpar = np.asarray([g_amp,g_sig,g_xcen,g_ycen,g_axrat,g_pa])
    #----------------------------------------------------------------------
    # Set the parameters of the lens galaxy
    #
    xc1 = 0.0       #x coordinate of the center of the lens (in units of Einstein radius).
    xc2 = 0.0       #y coordinate of the center of the lens (in units of Einstein radius).
    q   = 0.7       #Ellipticity of the lens.
    rc  = 0.0       #Core size of the lens (in units of Einstein radius).
    re  = 1.0       #Einstein radius of lens.
    pha = 45.0      #Orientation of lens.
    lpar = np.asarray([xc1,xc2,q,rc,re,pha])
    #----------------------------------------------------------------------
    # Calculate deflection angles
    #
    ai1,ai2,mua = lens_equation_sie(xi1,xi2,lpar)
    yi1 = xi1-ai1
    yi2 = xi2-ai2
    #----------------------------------------------------------------------
    # Produce Lensed images
    #
    gpar = np.asarray([g_amp,g_sig,g_xcen,g_ycen,g_axrat,g_pa])
    g_lensimage = gauss_2d(yi1,yi2,gpar)
    #----------------------------------------------------------------------
    # Set the parameters of the image of the lens galaxy
    #
    g_amp = 5.0   	# peak brightness value
    g_sig = 0.5  	# Gaussian "sigma" (i.e., size)
    g_xcen = 0.0  	# x position of center (also try (0.0,0.14)
    g_ycen = 0.0  	# y position of center
    g_axrat = 0.7 	# minor-to-major axis ratio
    g_pa = 45.0    	# major-axis position angle (degrees) c.c.w. from x axis
    gpar = np.asarray([g_amp,g_sig,g_xcen,g_ycen,g_axrat,g_pa])

    g_lens = gauss_2d(xi1,xi2,gpar)
    g_lensimage = g_lensimage+g_lens
    #----------------------------------------------------------------------
    # Generate noises
    #
    g_noise = np.random.normal(0,1,[nnn,nnn])*1.0
    #----------------------------------------------------------------------
    # Generate the final mock images of the lensing system
    #
    g_lensimage = g_lensimage+g_noise
    #----------------------------------------------------------------------
    # Plot the final images
    #
    levels = [0.0,1.0,2.0,3.0,4.0,5.0,6.0]
    pl.figure(figsize=(10,10))
    pl.contourf(g_lensimage,levels)
    #----------------------------------------------------------------------
    # Sample the positions of the points on grids
    #
    nns = 16

    xp01 = np.linspace(-boxsize/2.0,boxsize/2.0,nnn/nns)+0.5*dsx*nns
    xp02 = np.linspace(-boxsize/2.0,boxsize/2.0,nnn/nns)+0.5*dsx*nns
    xp2,xp1 = np.meshgrid(xp01,xp02)

    xp2 = xp2.reshape((nnn/nns*nnn/nns))
    xp1 = xp1.reshape((nnn/nns*nnn/nns))

    #----------------------------------------------------------------------
    # Draw the initial position of points on grids
    #
    pl.figure(figsize=(10,10))
    pl.xlim(-3,3)
    pl.ylim(-3,3)
    pl.plot(xp1,xp2,'ko')

    #----------------------------------------------------------------------
    # Move the points to the local peaks
    #
    xr1 = xp1*0.0
    xr2 = xp2*0.0

    for i in xrange(len(xp2)):
        xr1[i],xr2[i] = pixels_trans(xp1[i],xp2[i],xi1,xi2,g_lensimage,nns)

    X = np.vstack((xr1,xr2)).T
    #----------------------------------------------------------------------
    # Plot final positions of the points
    #
    pl.figure(figsize=(10,10))
    pl.xlim(-3,3)
    pl.ylim(-3,3)
    pl.plot(X[:, 0], X[:, 1], 'bo')

    #----------------------------------------------------------------------
    # Colorize different structures of the points
    #
    from sklearn.cluster import DBSCAN
    colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
    colors = np.hstack([colors] * 20)
    dbscan = DBSCAN(eps=0.14,min_samples=6)
    dbscan.fit(X)
    y_pred = dbscan.labels_.astype(np.int)
    pl.figure(figsize=(10,10))
    pl.xlim(-3,3)
    pl.ylim(-3,3)
    pl.scatter(X[:, 0], X[:, 1], color=colors[y_pred].tolist(), s=22)

    pl.show()
